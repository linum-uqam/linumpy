#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a quality report from pipeline metrics.

This script aggregates metrics from various pipeline steps and generates
a comprehensive report in HTML or text format to help identify potential
issues in the 3D reconstruction pipeline.
"""

# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

import argparse
import base64
import io as _io
import re
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

import numpy as np

from linumpy.utils.metrics import aggregate_metrics, compute_summary_statistics


# Logical pipeline step ordering
STEP_ORDER = [
    'stitch_3d',
    'xy_transform_estimation',
    'create_masks',
    'normalize_intensities',
    'psf_compensation',
    'crop_interface',
    'pairwise_registration',
    'stack_slices',
]

# Human-readable display names (step_name → display label)
STEP_DISPLAY_NAMES = {
    'stitch_3d': 'Stitch 3D',
    'xy_transform_estimation': 'XY Transform Estimation',
    'create_masks': 'Create Masks',
    'normalize_intensities': 'Normalize Intensities',
    'psf_compensation': 'PSF Compensation',
    'crop_interface': 'Crop Interface',
    'pairwise_registration': 'Pairwise Registration',
    'stack_slices': 'Stack Slices',
}

# Human-readable descriptions for pipeline steps
STEP_DESCRIPTIONS = {
    'stitch_3d': 'Stitches individual mosaic tiles into a single 2D slice.',
    'xy_transform_estimation': 'Estimates the affine transformation for tile overlap correction.',
    'create_masks': 'Creates binary tissue masks to guide registration and normalization.',
    'normalize_intensities': 'Normalizes per-slice intensities using agarose background.',
    'psf_compensation': 'Compensates for beam profile / PSF attenuation along the optical axis.',
    'crop_interface': 'Detects and crops the tissue-agarose interface.',
    'pairwise_registration': 'Registers consecutive serial sections to align the 3D volume.',
    'stack_slices': 'Stacks registered slices into the final 3D volume.',
}

# Maps pipeline step_name → image category shown in that step section
STEP_PREVIEW_CATEGORY = {
    'stitch_3d': 'stitch_preview',
    'create_masks': 'mask_preview',
    'pairwise_registration': 'common_space_preview',
}


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_dir",
                   help="Input directory containing pipeline output with metrics files.")
    p.add_argument("output_report",
                   help="Output report file path (.html, .zip, or .txt)")
    p.add_argument("--format", choices=['html', 'text', 'zip', 'auto'], default='auto',
                   help="Output format. 'auto' infers from extension. (default=%(default)s)")
    p.add_argument("--title", default="Pipeline Quality Report",
                   help="Report title. (default=%(default)s)")
    p.add_argument("--verbose", action='store_true',
                   help="Include all metric details in the report.")
    p.add_argument("--overview_png", type=Path, default=None,
                   help="Path to the main volume PNG screenshot (embedded in summary).")
    p.add_argument("--annotated_png", type=Path, default=None,
                   help="Path to the annotated volume PNG screenshot (embedded in summary).")
    p.add_argument("--max_overview_width", type=int, default=900,
                   help="Max pixel width for overview images. (default=%(default)s)")
    p.add_argument("--max_thumb_width", type=int, default=380,
                   help="Max pixel width for gallery thumbnails. (default=%(default)s)")
    p.add_argument("--no_images", action='store_true',
                   help="Disable image discovery for zip bundles.")
    return p


def get_status_color(status: str) -> str:
    """Get HTML color for status."""
    colors = {
        'ok': '#28a745',      # green
        'warning': '#ffc107', # yellow/amber
        'error': '#dc3545',   # red
        'info': '#17a2b8',    # blue
        'unknown': '#6c757d'  # gray
    }
    return colors.get(status, colors['unknown'])


def get_status_emoji(status: str) -> str:
    """Get emoji for status in text format."""
    emojis = {
        'ok': '✓',
        'warning': '⚠',
        'error': '✗',
        'info': 'ℹ',
        'unknown': '?'
    }
    return emojis.get(status, '?')


def format_value(value, precision: int = 4) -> str:
    """Format a value for display."""
    if isinstance(value, float):
        if abs(value) < 0.0001 or abs(value) > 10000:
            return f"{value:.{precision}e}"
        return f"{value:.{precision}f}"
    elif isinstance(value, list) and len(value) > 5:
        return f"[{len(value)} items]"
    return str(value)


def sort_steps(aggregated: Dict) -> Dict:
    """Sort pipeline steps in logical execution order."""
    def step_key(step_name):
        try:
            return (0, STEP_ORDER.index(step_name))
        except ValueError:
            return (1, step_name)
    return dict(sorted(aggregated.items(), key=lambda x: step_key(x[0])))


def extract_slice_id(source_file: str) -> str:
    """Extract a meaningful slice identifier from a source file path."""
    path = Path(source_file)
    # Search path components for a slice pattern like z01, z002, slice_3
    for part in reversed(path.parts):
        m = re.search(r'(z\d+|slice_z?\d+)', part, re.IGNORECASE)
        if m:
            return m.group(1)
    return path.stem


def parse_issue(issue_str: str) -> dict:
    """Parse an issue string of the form 'source: metric: value op threshold (level)'."""
    parts = issue_str.split(': ', 2)
    if len(parts) < 3:
        return {'source': parts[0] if parts else '', 'metric': '', 'raw': issue_str,
                'value': None, 'threshold': None}
    source, metric, rest = parts[0], parts[1], parts[2]
    m = re.match(r'([+-]?[\d.e+-]+)\s*([><]=?)\s*([+-]?[\d.e+-]+)', rest)
    if m:
        return {'source': source, 'metric': metric, 'raw': issue_str,
                'value': float(m.group(1)), 'op': m.group(2), 'threshold': float(m.group(3))}
    return {'source': source, 'metric': metric, 'raw': issue_str, 'value': None, 'threshold': None}


def group_issues(issues: List[str]) -> List[dict]:
    """
    Group issues by metric name.

    Returns a list of dicts with keys: metric, count, values, threshold, details.
    """
    groups = defaultdict(list)
    for issue in issues:
        parsed = parse_issue(issue)
        key = parsed['metric'] if parsed['metric'] else '__other__'
        groups[key].append(parsed)

    result = []
    for metric, items in groups.items():
        values = [i['value'] for i in items if i.get('value') is not None]
        threshold = items[0].get('threshold') if items else None
        op = items[0].get('op', '>') if items else '>'
        result.append({
            'metric': metric if metric != '__other__' else '',
            'count': len(items),
            'values': values,
            'threshold': threshold,
            'op': op,
            'details': [i['raw'] for i in items],
        })
    return result


def separate_metrics_by_type(metrics_list: List[Dict]) -> Tuple[dict, dict]:
    """
    Separate metrics into quality metrics and info/parameter fields.

    Returns
    -------
    tuple
        quality_metrics: {name: {'entries': [{value, status}], 'unit': str}}
        info_fields: {name: {'values': [v], 'description': str, 'is_constant': bool, 'display_value': any}}
    """
    quality_metrics: dict = {}
    info_fields: dict = {}

    for m in metrics_list:
        for name, data in m.get('metrics', {}).items():
            if not isinstance(data, dict):
                continue
            status = data.get('status', 'ok')
            value = data.get('value')
            unit = data.get('unit') or ''
            desc = data.get('description') or ''

            if status == 'info':
                if name not in info_fields:
                    info_fields[name] = {'values': [], 'description': desc, 'unit': unit}
                info_fields[name]['values'].append(value)
            else:
                if name not in quality_metrics:
                    quality_metrics[name] = {'entries': [], 'unit': unit, 'description': desc}
                quality_metrics[name]['entries'].append({'value': value, 'status': status})

    # Determine if each info field is constant across all files
    for name, info in info_fields.items():
        vals = info['values']
        try:
            numeric = [v for v in vals if isinstance(v, (int, float))]
            if numeric and len(numeric) == len(vals):
                is_const = float(np.std(numeric)) < 1e-10
            else:
                is_const = len(set(str(v) for v in vals)) <= 1
        except Exception:
            is_const = len(set(str(v) for v in vals)) <= 1
        info['is_constant'] = is_const
        info['display_value'] = vals[0] if vals else None

    return quality_metrics, info_fields


def generate_sparkline_svg(values: list, statuses: Optional[List[str]] = None,
                           width: int = 160, height: int = 36) -> str:
    """Generate an inline SVG bar-chart sparkline for a list of values."""
    numeric = [(i, v) for i, v in enumerate(values) if isinstance(v, (int, float))]
    if len(numeric) < 2:
        return ''

    all_vals = [v for _, v in numeric]
    min_val, max_val = min(all_vals), max(all_vals)
    val_range = max_val - min_val or 1.0

    if statuses is None:
        statuses = ['ok'] * len(values)

    n = len(values)
    bar_w = width / n
    rects = []
    for i, v in numeric:
        h = max(2.0, (v - min_val) / val_range * (height - 4))
        y = height - h
        color = get_status_color(statuses[i]) if i < len(statuses) else get_status_color('ok')
        rects.append(
            f'<rect x="{i * bar_w:.1f}" y="{y:.1f}" '
            f'width="{max(1.0, bar_w - 0.5):.1f}" height="{h:.1f}" '
            f'fill="{color}" opacity="0.85" rx="1"/>'
        )

    title = f"Min: {min_val:.3g}  Max: {max_val:.3g}  n={len(numeric)}"
    return (
        f'<svg width="{width}" height="{height}" '
        f'style="display:block;vertical-align:middle;" title="{title}">'
        + ''.join(rects) + '</svg>'
    )


def generate_trend_line_svg(values: list, labels: Optional[List[str]] = None,
                            width: int = 420, height: int = 90,
                            show_trend: bool = True,
                            color: str = '#4a90d9') -> str:
    """Generate an inline SVG line chart for cross-slice trend visualisation."""
    numeric = [(i, float(v)) for i, v in enumerate(values) if isinstance(v, (int, float))]
    if len(numeric) < 2:
        return ''

    xs = [p[0] for p in numeric]
    ys = [p[1] for p in numeric]
    min_y, max_y = min(ys), max(ys)
    y_range = max_y - min_y or 1.0
    pad_x, pad_y = 30, 10

    def to_svg_x(i):
        return pad_x + (i / (len(values) - 1)) * (width - 2 * pad_x)

    def to_svg_y(v):
        return height - pad_y - ((v - min_y) / y_range) * (height - 2 * pad_y)

    # Build polyline points
    pts = ' '.join(f"{to_svg_x(i):.1f},{to_svg_y(v):.1f}" for i, v in numeric)

    elements = [
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5" stroke-opacity="0.85"/>',
    ]

    # Dots at each data point
    for i, v in numeric:
        elements.append(
            f'<circle cx="{to_svg_x(i):.1f}" cy="{to_svg_y(v):.1f}" r="2" '
            f'fill="{color}" opacity="0.7"/>'
        )

    # Trend line (least squares)
    if show_trend and len(xs) >= 3:
        x_arr = np.array(xs, dtype=float)
        y_arr = np.array(ys, dtype=float)
        slope = (np.mean(x_arr * y_arr) - np.mean(x_arr) * np.mean(y_arr)) / (
            np.mean(x_arr ** 2) - np.mean(x_arr) ** 2 + 1e-12)
        intercept = np.mean(y_arr) - slope * np.mean(x_arr)
        x0, x1 = xs[0], xs[-1]
        y0, y1 = slope * x0 + intercept, slope * x1 + intercept
        elements.append(
            f'<line x1="{to_svg_x(x0):.1f}" y1="{to_svg_y(y0):.1f}" '
            f'x2="{to_svg_x(x1):.1f}" y2="{to_svg_y(y1):.1f}" '
            f'stroke="#e74c3c" stroke-width="1" stroke-dasharray="4 2" opacity="0.7"/>'
        )

    # Y-axis labels
    elements.append(
        f'<text x="{pad_x - 3}" y="{to_svg_y(max_y):.1f}" '
        f'text-anchor="end" font-size="8" fill="#888">{max_y:.3g}</text>'
    )
    elements.append(
        f'<text x="{pad_x - 3}" y="{to_svg_y(min_y):.1f}" '
        f'text-anchor="end" font-size="8" fill="#888">{min_y:.3g}</text>'
    )

    title_text = f"n={len(numeric)}, range [{min_y:.3g}, {max_y:.3g}]"
    return (
        f'<svg width="{width}" height="{height}" '
        f'style="display:block;" title="{title_text}">'
        + ''.join(elements) + '</svg>'
    )


def compute_cross_slice_trends(aggregated: Dict[str, List[Dict]]) -> Dict:
    """
    Compute cross-slice aggregate trends from aggregated metrics.

    Returns a dict with trend groups, each containing:
      'label', 'description', 'series': [{name, values, unit}]
    """
    trends = {}

    def _extract(metrics_list, key):
        """Extract sorted numerical values for a given metric key."""
        pairs = []
        for m in metrics_list:
            src = m.get('source_file', '')
            val = m.get('metrics', {}).get(key, {}).get('value')
            if isinstance(val, (int, float)):
                pairs.append((src, val))
        pairs.sort(key=lambda p: p[0])  # sort by source file path
        return [v for _, v in pairs]

    # XY tile transform: scale and shear across slices
    if 'xy_transform_estimation' in aggregated:
        ml = aggregated['xy_transform_estimation']
        t00 = _extract(ml, 'transform_00')
        t11 = _extract(ml, 'transform_11')
        rms = _extract(ml, 'rms_residual')
        acc_sys = _extract(ml, 'accumulated_systematic_error_px')
        acc_rnd = _extract(ml, 'accumulated_random_error_px')
        series = []
        if t00:
            series.append({'name': 'Step Y (px)', 'values': t00, 'unit': 'px'})
        if t11:
            series.append({'name': 'Step X (px)', 'values': t11, 'unit': 'px'})
        if rms:
            series.append({'name': 'RMS residual (px)', 'values': rms, 'unit': 'px'})
        if acc_sys:
            series.append({'name': 'Accum. systematic error (px)', 'values': acc_sys, 'unit': 'px'})
        if acc_rnd:
            series.append({'name': 'Accum. random error (px)', 'values': acc_rnd, 'unit': 'px'})
        if series:
            trends['xy_transform'] = {
                'label': 'XY Tile Transform Consistency',
                'description': ('Tile step sizes and fitting residuals across slices. '
                                'Large variation indicates unstable tile positioning.'),
                'series': series,
            }

    # Pairwise registration: cumulative drift
    if 'pairwise_registration' in aggregated:
        ml = aggregated['pairwise_registration']
        tx = _extract(ml, 'translation_x')
        ty = _extract(ml, 'translation_y')
        rot = _extract(ml, 'rotation')
        series = []
        if tx:
            cum_tx = list(np.cumsum(tx))
            series.append({'name': 'Cumulative tx (px)', 'values': cum_tx, 'unit': 'px'})
        if ty:
            cum_ty = list(np.cumsum(ty))
            series.append({'name': 'Cumulative ty (px)', 'values': cum_ty, 'unit': 'px'})
        if rot:
            cum_rot = list(np.cumsum(rot))
            series.append({'name': 'Cumulative rotation (deg)', 'values': cum_rot, 'unit': 'deg'})
        if series:
            trends['registration_drift'] = {
                'label': 'Cumulative Registration Drift',
                'description': ('Accumulated translation and rotation across all slices. '
                                'A large net drift indicates systematic 3D volume distortion.'),
                'series': series,
            }

    # Interface depth trend
    if 'crop_interface' in aggregated:
        ml = aggregated['crop_interface']
        depth = _extract(ml, 'detected_interface_depth_um')
        if depth:
            trends['interface_depth'] = {
                'label': 'Interface Depth Trend',
                'description': ('Detected tissue-agarose interface depth across slices. '
                                'A systematic slope may indicate progressive tissue deformation.'),
                'series': [{'name': 'Interface depth (µm)', 'values': depth, 'unit': 'µm'}],
            }

    # Background normalization drift
    if 'normalize_intensities' in aggregated:
        ml = aggregated['normalize_intensities']
        bg = _extract(ml, 'mean_background')
        if bg:
            trends['background_drift'] = {
                'label': 'Background Level Trend',
                'description': ('Mean agarose background level across slices. '
                                'A strong trend indicates illumination drift during acquisition.'),
                'series': [{'name': 'Mean background', 'values': bg, 'unit': ''}],
            }

    return trends


# =============================================================================
# Diagnostic data discovery
# =============================================================================

def discover_diagnostic_data(input_dir: Path) -> Dict[str, dict]:
    """
    Discover diagnostic outputs in the pipeline output directory.

    Looks for known diagnostic subdirectories and reads their JSON data.

    Returns
    -------
    dict
        Maps diagnostic_name → {'label', 'description', 'json_data': [...], 'images': [Path]}
    """
    import json as _json
    diagnostics: Dict[str, dict] = {}

    diag_dir = input_dir / 'diagnostics'
    if not diag_dir.exists():
        return diagnostics

    # Define known diagnostics: (subdir, label, description)
    known = [
        ('dilation_analysis', 'Tile Dilation Analysis',
         'Per-slice scale factors and mosaic positioning accuracy.'),
        ('aggregated_dilation', 'Aggregated Dilation Analysis',
         'Cross-slice tile dilation summary.'),
        ('rotation_analysis', 'Rotation Drift Analysis',
         'Rotation angle drift across slices.'),
        ('acquisition_rotation', 'Acquisition Rotation Analysis',
         'In-plane rotation estimated from acquisition metadata.'),
        ('motor_only_stitch', 'Motor-Only Stitching (comparison)',
         'Stitched mosaic using motor positions only (no registration correction).'),
        ('motor_only_stack', 'Motor-Only Stack (comparison)',
         'Volume stacked without pairwise registration (motor positions only).'),
        ('stitch_comparison', 'Stitching Comparison',
         'Side-by-side comparison of registration-based vs motor-based stitching.'),
    ]

    for subdir_name, label, description in known:
        subdir = diag_dir / subdir_name
        if not subdir.exists():
            continue

        json_data = []
        images = []

        # Collect all JSON files (recursively for per-slice diagnostics)
        for json_file in sorted(subdir.rglob('*.json')):
            try:
                with open(json_file) as f:
                    data = _json.load(f)
                data['_source'] = str(json_file)
                json_data.append(data)
            except Exception:
                pass

        # Collect PNG images
        for png_file in sorted(subdir.rglob('*.png')):
            images.append(png_file)

        if json_data or images:
            diagnostics[subdir_name] = {
                'label': label,
                'description': description,
                'json_data': json_data,
                'images': images,
            }

    return diagnostics

def discover_images(input_dir: Path,
                    overview_png: Optional[Path] = None,
                    annotated_png: Optional[Path] = None) -> Dict[str, List[Path]]:
    """
    Discover preview images in the pipeline output directory.

    Returns a dict mapping category → sorted list of image paths:
      'overview'              – main volume screenshots (up to 2)
      'stitch_preview'        – per-slice stitched previews
      'mask_preview'          – per-slice mask previews
      'common_space_preview'  – common-space alignment previews
      'diag_*'                – images found in diagnostics/ subdirs
    """
    images: Dict[str, List[Path]] = {
        'overview': [],
        'stitch_preview': [],
        'mask_preview': [],
        'common_space_preview': [],
    }

    # Overview images from CLI (staged in Nextflow work dir)
    for p in [overview_png, annotated_png]:
        if p and Path(p).exists():
            images['overview'].append(Path(p))

    # Stitched slice previews
    stitch_dir = input_dir / 'previews' / 'stitched_slices'
    if stitch_dir.exists():
        images['stitch_preview'] = sorted(stitch_dir.glob('*.png'))

    # Mask previews (published alongside masks)
    mask_dir = input_dir / 'create_registration_masks'
    if mask_dir.exists():
        images['mask_preview'] = sorted(mask_dir.glob('*_preview.png'))

    # Common-space alignment previews
    cs_dir = input_dir / 'common_space_previews'
    if cs_dir.exists():
        images['common_space_preview'] = sorted(cs_dir.glob('*.png'))

    # Auto-detect overview from stack output directories if not provided via CLI
    if not images['overview']:
        for stack_dir_name in ('stack_motor', 'stack', 'normalize_z_intensity'):
            d = input_dir / stack_dir_name
            if d.exists():
                pngs = sorted(d.glob('*.png'))
                if pngs:
                    images['overview'] = pngs[:2]   # at most overview + annotated
                    break

    # Diagnostic images: add one category per diagnostics subdir
    diag_dir = input_dir / 'diagnostics'
    if diag_dir.exists():
        for subdir in sorted(diag_dir.iterdir()):
            if subdir.is_dir():
                pngs = sorted(subdir.rglob('*.png'))
                if pngs:
                    cat_key = f'diag_{subdir.name}'
                    images[cat_key] = pngs

    return images


def image_to_data_uri(path: Path, max_width: Optional[int] = None) -> str:
    """Encode a PNG image as a base64 data URI, optionally resizing."""
    if max_width and _PIL_AVAILABLE:
        with _PILImage.open(path) as img:
            if img.width > max_width:
                ratio = max_width / img.width
                new_size = (max_width, int(img.height * ratio))
                img = img.resize(new_size, _PILImage.LANCZOS)
            buf = _io.BytesIO()
            img.save(buf, format='PNG', optimize=True)
            data_bytes = buf.getvalue()
    else:
        data_bytes = path.read_bytes()
    b64 = base64.b64encode(data_bytes).decode('ascii')
    return f"data:image/png;base64,{b64}"


def render_image_gallery_html(images: List[Path],
                              mode: str = 'embed',
                              category: str = 'images',
                              label: str = 'Preview Images',
                              max_width: int = 380) -> str:
    """
    Render a collapsible image gallery section.

    Parameters
    ----------
    mode : 'embed' (base64 in HTML) or 'link' (relative path for zip mode)
    category : image category name, used as subfolder in zip mode
    """
    if not images:
        return ''

    items = []
    for p in images:
        if mode == 'embed':
            src = image_to_data_uri(p, max_width=max_width)
        else:
            src = f"previews/{category}/{p.name}"
        name = p.stem
        items.append(
            f'<figure class="gallery-item">'
            f'<a href="{src}" target="_blank">'
            f'<img src="{src}" alt="{name}" title="{name}" loading="lazy">'
            f'</a>'
            f'<figcaption>{name}</figcaption>'
            f'</figure>'
        )

    return f"""
        <details class="gallery-details">
            <summary class="gallery-summary">Preview Images ({len(images)})</summary>
            <div class="image-gallery">
                {''.join(items)}
            </div>
        </details>
"""


def generate_zip_bundle(html: str,
                        images: Dict[str, List[Path]],
                        output_path: Path) -> None:
    """Bundle the HTML report and all image files into a zip archive."""
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('index.html', html)
        for category, paths in images.items():
            for p in paths:
                zf.write(p, f'previews/{category}/{p.name}')


def compute_overall_status(aggregated: Dict[str, List[Dict]]) -> tuple:
    """
    Compute overall status counts from aggregated metrics.

    Returns
    -------
    tuple
        (all_statuses, error_count, warning_count, ok_count)
    """
    all_statuses = []
    for step_metrics in aggregated.values():
        for m in step_metrics:
            all_statuses.append(m.get('overall_status', 'unknown'))

    error_count = all_statuses.count('error')
    warning_count = all_statuses.count('warning')
    ok_count = all_statuses.count('ok')

    return all_statuses, error_count, warning_count, ok_count


def get_step_status(metrics_list: List[Dict]) -> str:
    """Get the overall status for a step based on its metrics."""
    step_statuses = [m.get('overall_status', 'unknown') for m in metrics_list]
    if 'error' in step_statuses:
        return 'error'
    elif 'warning' in step_statuses:
        return 'warning'
    return 'ok'


def collect_issues(metrics_list: List[Dict]) -> tuple:
    """
    Collect all warnings and errors from a metrics list.

    Returns
    -------
    tuple
        (all_warnings, all_errors)
    """
    all_warnings = []
    all_errors = []
    for m in metrics_list:
        source = Path(m.get('source_file', 'unknown')).stem
        for w in m.get('warnings', []):
            all_warnings.append(f"{source}: {w}")
        for e in m.get('errors', []):
            all_errors.append(f"{source}: {e}")
    return all_warnings, all_errors


def _render_grouped_issues_html(grouped: List[dict], color_class: str, label: str) -> str:
    """Render a collapsible grouped-issues section in HTML."""
    total = sum(g['count'] for g in grouped)
    html = f"""
        <details class="issues-details {color_class}-details">
            <summary class="issues-summary {color_class}-summary">
                <strong>{label}</strong>
                <span class="issue-count-badge">{total}</span>
            </summary>
            <div class="issues-body">
"""
    for g in grouped:
        if g['count'] == 1:
            html += f'                <div class="issue-item">{g["details"][0]}</div>\n'
        else:
            vals = g['values']
            if vals:
                val_str = f"range {min(vals):.3g} – {max(vals):.3g}"
            else:
                val_str = f"{g['count']} occurrences"
            thresh_str = (f", threshold: {g['threshold']:.3g}" if g['threshold'] is not None else "")
            summary_line = (
                f"<strong>{g['metric']}</strong>: {g['count']} slices affected "
                f"({val_str}{thresh_str})"
            )
            html += f'                <details class="sub-issue">\n'
            html += f'                    <summary class="issue-item">{summary_line}</summary>\n'
            html += '                    <div class="sub-issue-list">\n'
            for detail in g['details']:
                html += f'                        <div class="sub-issue-item">{detail}</div>\n'
            html += '                    </div>\n'
            html += '                </details>\n'
    html += "            </div>\n        </details>\n"
    return html


def generate_html_report(aggregated: Dict[str, List[Dict]],
                         title: str,
                         verbose: bool = False,
                         images: Optional[Dict[str, List[Path]]] = None,
                         image_mode: str = 'embed',
                         max_overview_width: int = 900,
                         max_thumb_width: int = 380,
                         trends: Optional[Dict] = None,
                         diagnostics: Optional[Dict] = None) -> str:
    """Generate an HTML report from aggregated metrics."""
    aggregated = sort_steps(aggregated)
    images = images or {}

    _, error_count, warning_count, ok_count = compute_overall_status(aggregated)

    if error_count > 0:
        overall_status = 'error'
        overall_message = f"{error_count} error(s), {warning_count} warning(s)"
    elif warning_count > 0:
        overall_status = 'warning'
        overall_message = f"{warning_count} warning(s)"
    else:
        overall_status = 'ok'
        overall_message = "All checks passed"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{ margin: 0; }}
        .header .timestamp {{ opacity: 0.8; font-size: 0.9em; }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-status {{
            display: inline-block;
            padding: 10px 20px;
            border-radius: 5px;
            color: white;
            font-weight: bold;
            font-size: 1.2em;
            margin-bottom: 15px;
        }}
        .step-section {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .step-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
            margin-bottom: 10px;
        }}
        .step-title {{ font-size: 1.3em; font-weight: bold; }}
        .step-description {{ color: #666; font-size: 0.9em; margin-bottom: 15px; }}
        .status-badge {{
            padding: 5px 15px;
            border-radius: 20px;
            color: white;
            font-size: 0.85em;
            font-weight: bold;
            white-space: nowrap;
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .metrics-table th, .metrics-table td {{
            padding: 8px 10px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        .metrics-table th {{
            background: #f8f9fa;
            font-weight: 600;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .metrics-table td.sparkline-cell {{
            padding: 4px 10px;
        }}
        .metric-status {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
            flex-shrink: 0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }}
        .stat-box {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }}
        .stat-value {{ font-size: 1.4em; font-weight: bold; color: #333; }}
        .stat-label {{ font-size: 0.8em; color: #666; }}

        /* Issues sections */
        .issues-details {{
            border-radius: 5px;
            margin-top: 12px;
            overflow: hidden;
        }}
        .issues-summary {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 15px;
            cursor: pointer;
            user-select: none;
            list-style: none;
            font-weight: 600;
        }}
        .issues-summary::-webkit-details-marker {{ display: none; }}
        .issues-summary::before {{
            content: "▶";
            font-size: 0.7em;
            transition: transform 0.2s;
        }}
        details[open] > .issues-summary::before {{ transform: rotate(90deg); }}
        .error-details   {{ border: 1px solid #dc3545; }}
        .error-summary   {{ background: #f8d7da; color: #842029; }}
        .warning-details {{ border: 1px solid #ffc107; }}
        .warning-summary {{ background: #fff3cd; color: #664d03; }}
        .issue-count-badge {{
            background: rgba(0,0,0,0.15);
            border-radius: 20px;
            padding: 1px 8px;
            font-size: 0.85em;
        }}
        .issues-body {{
            padding: 8px 15px 12px;
        }}
        .issue-item {{
            padding: 4px 0;
            border-bottom: 1px solid rgba(0,0,0,0.06);
            font-size: 0.9em;
        }}
        .issue-item:last-child {{ border-bottom: none; }}
        .sub-issue > summary {{ cursor: pointer; }}
        .sub-issue-list {{
            padding-left: 20px;
            border-left: 3px solid #dee2e6;
            margin: 4px 0 4px 8px;
        }}
        .sub-issue-item {{
            padding: 2px 0;
            font-size: 0.85em;
            color: #555;
        }}

        /* Info/params section */
        .params-details {{
            border: 1px solid #dee2e6;
            border-radius: 5px;
            margin-top: 12px;
        }}
        .params-summary {{
            padding: 8px 15px;
            cursor: pointer;
            background: #f8f9fa;
            color: #495057;
            font-size: 0.9em;
            font-weight: 500;
            user-select: none;
            list-style: none;
        }}
        .params-summary::-webkit-details-marker {{ display: none; }}
        .params-summary::before {{
            content: "▶";
            font-size: 0.65em;
            margin-right: 6px;
            transition: transform 0.2s;
        }}
        details[open] > .params-summary::before {{ transform: rotate(90deg); }}
        .params-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85em;
        }}
        .params-table td {{
            padding: 5px 15px;
            border-bottom: 1px solid #f0f0f0;
            vertical-align: top;
        }}
        .params-table td:first-child {{ color: #555; font-weight: 500; width: 35%; }}
        .params-table td:last-child  {{ color: #333; font-family: monospace; }}

        /* Verbose individual results */
        .collapsible {{ cursor: pointer; user-select: none; }}
        .collapsible:hover {{ background: #f0f0f0; }}
        .content {{ display: none; padding: 10px; background: #fafafa; }}
        .show {{ display: block; }}
        .section-label {{
            font-size: 0.8em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #666;
            margin: 18px 0 6px;
        }}

        /* Image galleries */
        .gallery-details {{
            border: 1px solid #dee2e6;
            border-radius: 5px;
            margin-top: 12px;
            overflow: hidden;
        }}
        .gallery-summary {{
            padding: 8px 15px;
            cursor: pointer;
            background: #f0f4ff;
            color: #3d4db7;
            font-size: 0.9em;
            font-weight: 500;
            user-select: none;
            list-style: none;
        }}
        .gallery-summary::-webkit-details-marker {{ display: none; }}
        .gallery-summary::before {{
            content: "▶";
            font-size: 0.65em;
            margin-right: 6px;
            transition: transform 0.2s;
        }}
        details[open] > .gallery-summary::before {{ transform: rotate(90deg); }}
        .image-gallery {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            padding: 12px;
            background: #fafbff;
            max-height: 520px;
            overflow-y: auto;
        }}
        .gallery-item {{
            flex: 0 0 auto;
            text-align: center;
        }}
        .gallery-item img {{
            display: block;
            max-width: 100%;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            transition: box-shadow 0.15s;
        }}
        .gallery-item img:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.2); }}
        .gallery-item figcaption {{
            font-size: 0.7em;
            color: #666;
            margin-top: 3px;
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        /* Overview image in summary */
        .overview-container {{
            margin-top: 16px;
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .overview-container figure {{
            margin: 0;
            flex: 1 1 45%;
        }}
        .overview-container img {{
            width: 100%;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        .overview-container figcaption {{
            font-size: 0.8em;
            color: #666;
            margin-top: 4px;
            text-align: center;
        }}

        /* Cross-slice trends section */
        .trends-section {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .trends-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(460px, 1fr));
            gap: 14px;
            margin-top: 12px;
        }}
        .trend-card {{
            border: 1px solid #dee2e6;
            border-radius: 6px;
            overflow: hidden;
        }}
        .trend-card-header {{
            background: #f8f9fa;
            padding: 8px 12px;
            font-weight: 600;
            font-size: 0.9em;
            color: #333;
        }}
        .trend-card-desc {{
            font-size: 0.8em;
            color: #666;
            padding: 4px 12px 8px;
        }}
        .trend-series {{
            padding: 4px 12px 8px;
        }}
        .trend-series-label {{
            font-size: 0.78em;
            color: #555;
            margin-bottom: 2px;
        }}

        /* Diagnostics section */
        .diag-section {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .diag-item {{
            border: 1px solid #dee2e6;
            border-radius: 6px;
            margin-top: 12px;
            overflow: hidden;
        }}
        .diag-item-header {{
            background: #f0f4ff;
            padding: 10px 15px;
            font-weight: 600;
            font-size: 0.95em;
            color: #3d4db7;
        }}
        .diag-item-desc {{
            font-size: 0.85em;
            color: #555;
            padding: 6px 15px 8px;
        }}
        .diag-kv-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82em;
        }}
        .diag-kv-table td {{
            padding: 3px 15px;
            border-bottom: 1px solid #f0f0f0;
        }}
        .diag-kv-table td:first-child {{ color: #555; font-weight: 500; width: 40%; }}
        .diag-kv-table td:last-child  {{ color: #333; font-family: monospace; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>

    <div class="summary">
        <h2>Summary</h2>
        <div class="summary-status" style="background-color: {get_status_color(overall_status)};">
            {overall_message}
        </div>
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value">{len(aggregated)}</div>
                <div class="stat-label">Pipeline Steps</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{sum(len(v) for v in aggregated.values())}</div>
                <div class="stat-label">Total Metrics Files</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" style="color: {get_status_color('ok')};">{ok_count}</div>
                <div class="stat-label">OK</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" style="color: {get_status_color('warning')};">{warning_count}</div>
                <div class="stat-label">Warnings</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" style="color: {get_status_color('error')};">{error_count}</div>
                <div class="stat-label">Errors</div>
            </div>
        </div>
    </div>
"""

    # Overview images in the summary section
    overview_imgs = images.get('overview', [])
    if overview_imgs:
        html += '    <div class="summary" style="padding-top:10px;">\n'
        html += '        <div class="section-label">Volume Overview</div>\n'
        html += '        <div class="overview-container">\n'
        for p in overview_imgs:
            if image_mode == 'embed':
                src = image_to_data_uri(p, max_width=max_overview_width)
            else:
                src = f"previews/overview/{p.name}"
            html += (
                f'            <figure>'
                f'<a href="{src}" target="_blank">'
                f'<img src="{src}" alt="{p.stem}"></a>'
                f'<figcaption>{p.stem}</figcaption></figure>\n'
            )
        html += '        </div>\n    </div>\n'

    # Cross-slice trends section
    if trends:
        colors = ['#4a90d9', '#e67e22', '#27ae60', '#8e44ad', '#c0392b']
        html += '\n    <div class="trends-section">\n'
        html += '        <h2>Cross-Slice Trends</h2>\n'
        html += ('        <p style="color:#555;font-size:0.9em;">'
                 'Aggregate quality indicators computed across all slices. '
                 'Red dashed lines show the linear trend.</p>\n')
        html += '        <div class="trends-grid">\n'
        for trend_key, trend in trends.items():
            html += '            <div class="trend-card">\n'
            html += f'                <div class="trend-card-header">{trend["label"]}</div>\n'
            html += f'                <div class="trend-card-desc">{trend["description"]}</div>\n'
            for ci, series in enumerate(trend['series']):
                col = colors[ci % len(colors)]
                svg = generate_trend_line_svg(series['values'], color=col)
                html += '                <div class="trend-series">\n'
                html += f'                    <div class="trend-series-label">{series["name"]}</div>\n'
                html += f'                    {svg}\n'
                html += '                </div>\n'
            html += '            </div>\n'
        html += '        </div>\n    </div>\n'

    # Generate section for each step
    for step_name, metrics_list in aggregated.items():
        summary = compute_summary_statistics(metrics_list)
        step_status = get_step_status(metrics_list)
        description = STEP_DESCRIPTIONS.get(step_name, '')

        # Separate quality metrics from info/parameter fields
        quality_metrics, info_fields = separate_metrics_by_type(metrics_list)

        html += f"""
    <div class="step-section">
        <div class="step-header">
            <span class="step-title">{STEP_DISPLAY_NAMES.get(step_name, step_name.replace('_', ' ').title())}</span>
            <span class="status-badge" style="background-color: {get_status_color(step_status)};">
                {summary['count']} items &mdash; {step_status.upper()}
            </span>
        </div>
"""
        if description:
            html += f'        <div class="step-description">{description}</div>\n'

        # --- Quality metrics stats table with sparklines ---
        if quality_metrics:
            html += '        <div class="section-label">Quality Metrics</div>\n'
            html += """        <table class="metrics-table">
            <tr>
                <th>Metric</th>
                <th>Mean</th>
                <th>Median</th>
                <th>Std</th>
                <th>Min</th>
                <th>Max</th>
                <th>Distribution</th>
            </tr>
"""
            for metric_name, mdata in quality_metrics.items():
                entries = mdata['entries']
                numeric_vals = [e['value'] for e in entries
                                if isinstance(e.get('value'), (int, float))]
                if not numeric_vals:
                    continue
                arr = np.array(numeric_vals)
                mean_v = float(np.mean(arr))
                median_v = float(np.median(arr))
                std_v = float(np.std(arr))
                min_v = float(np.min(arr))
                max_v = float(np.max(arr))
                statuses = [e.get('status', 'ok') for e in entries]
                unit = mdata.get('unit', '')
                unit_str = f' {unit}' if unit else ''

                # Worst status in this metric
                if 'error' in statuses:
                    metric_status = 'error'
                elif 'warning' in statuses:
                    metric_status = 'warning'
                else:
                    metric_status = 'ok'

                sparkline = generate_sparkline_svg(
                    [e.get('value') for e in entries], statuses)

                html += f"""            <tr>
                <td>
                    <span class="metric-status" style="background-color:{get_status_color(metric_status)};"></span>
                    {metric_name}{unit_str}
                </td>
                <td>{format_value(mean_v)}</td>
                <td>{format_value(median_v)}</td>
                <td>{format_value(std_v)}</td>
                <td>{format_value(min_v)}</td>
                <td>{format_value(max_v)}</td>
                <td class="sparkline-cell">{sparkline}</td>
            </tr>
"""
            html += "        </table>\n"

        # --- Errors and warnings (grouped, collapsible) ---
        all_warnings, all_errors = collect_issues(metrics_list)

        if all_errors:
            grouped_errors = group_issues(all_errors)
            html += _render_grouped_issues_html(grouped_errors, 'error', 'Errors')

        if all_warnings:
            grouped_warnings = group_issues(all_warnings)
            html += _render_grouped_issues_html(grouped_warnings, 'warning', 'Warnings')

        # --- Info / parameter fields (collapsed) ---
        if info_fields:
            constant_params = {k: v for k, v in info_fields.items() if v['is_constant']}
            variable_infos = {k: v for k, v in info_fields.items() if not v['is_constant']}

            if constant_params:
                html += """
        <details class="params-details">
            <summary class="params-summary">Pipeline Parameters</summary>
            <table class="params-table">
"""
                for name, info in constant_params.items():
                    val = info['display_value']
                    unit = info.get('unit', '')
                    unit_str = f' {unit}' if unit else ''
                    html += f"""                <tr>
                    <td>{name}</td>
                    <td>{format_value(val)}{unit_str}</td>
                </tr>
"""
                html += "            </table>\n        </details>\n"

            if variable_infos:
                html += """
        <details class="params-details">
            <summary class="params-summary">Variable Info Fields (per-slice)</summary>
            <table class="metrics-table" style="font-size:0.85em;">
                <tr>
                    <th>Field</th>
                    <th>Mean</th>
                    <th>Std</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
"""
                for name, info in variable_infos.items():
                    numeric = [v for v in info['values'] if isinstance(v, (int, float))]
                    if not numeric:
                        continue
                    arr = np.array(numeric)
                    unit = info.get('unit', '')
                    unit_str = f' {unit}' if unit else ''
                    html += f"""                <tr>
                    <td>{name}{unit_str}</td>
                    <td>{format_value(float(np.mean(arr)))}</td>
                    <td>{format_value(float(np.std(arr)))}</td>
                    <td>{format_value(float(np.min(arr)))}</td>
                    <td>{format_value(float(np.max(arr)))}</td>
                </tr>
"""
                html += "            </table>\n        </details>\n"

        # --- Verbose: individual per-slice results (collapsible as a unit) ---
        if verbose:
            n_items = len(metrics_list)
            html += f"""
        <details class="params-details">
            <summary class="params-summary">Individual Results ({n_items} slices)</summary>
"""
            for m in metrics_list:
                source = extract_slice_id(m.get('source_file', 'unknown'))
                m_status = m.get('overall_status', 'unknown')
                html += f"""
        <details>
            <summary style="cursor:pointer; padding:5px; background:#f8f9fa; border-radius:3px; margin:5px 0;">
                <span class="metric-status" style="background-color: {get_status_color(m_status)};"></span>
                {source}
            </summary>
            <table class="metrics-table" style="margin:10px 0;">
"""
                for name, data in m.get('metrics', {}).items():
                    if isinstance(data, dict):
                        value = data.get('value', 'N/A')
                        unit = data.get('unit', '') or ''
                        status = data.get('status', 'info')
                        html += f"""
                <tr>
                    <td>
                        <span class="metric-status" style="background-color: {get_status_color(status)};"></span>
                        {name}
                    </td>
                    <td>{format_value(value)}{(' ' + unit) if unit else ''}</td>
                </tr>
"""
                html += """            </table>
        </details>
"""
            html += "        </details>\n"

        # --- Per-step preview image gallery ---
        preview_category = STEP_PREVIEW_CATEGORY.get(step_name)
        if preview_category:
            step_imgs = images.get(preview_category, [])
            if step_imgs:
                html += render_image_gallery_html(
                    step_imgs, mode=image_mode,
                    category=preview_category,
                    max_width=max_thumb_width)

        html += "    </div>\n"

    # Diagnostics section (only if diagnostic data was found)
    if diagnostics:
        html += '\n    <div class="diag-section">\n'
        html += '        <h2>Diagnostic Outputs</h2>\n'
        html += ('        <p style="color:#555;font-size:0.9em;">'
                 'Additional diagnostic analyses enabled in the pipeline configuration.</p>\n')
        for diag_key, diag in diagnostics.items():
            label = diag['label']
            description = diag['description']
            json_data = diag.get('json_data', [])
            diag_images = diag.get('images', [])

            html += f'        <div class="diag-item">\n'
            html += f'            <div class="diag-item-header">{label}</div>\n'
            html += f'            <div class="diag-item-desc">{description}</div>\n'

            # Render key JSON fields
            if json_data:
                # Collect interesting numeric/scalar fields from first entry
                first = json_data[0]
                numeric_fields = {}
                for k, v in first.items():
                    if k.startswith('_') or k == 'slice_id':
                        continue
                    if isinstance(v, (int, float, str, bool)):
                        numeric_fields[k] = v
                    elif isinstance(v, dict):
                        # like scale_factors / residuals / distortions sub-dicts
                        for sk, sv in v.items():
                            if isinstance(sv, (int, float, str, bool)):
                                numeric_fields[f'{k}.{sk}'] = sv

                if numeric_fields:
                    html += '            <table class="diag-kv-table">\n'
                    for k, v in list(numeric_fields.items())[:20]:
                        html += (f'                <tr><td>{k}</td>'
                                 f'<td>{format_value(v) if isinstance(v, (int, float)) else v}</td></tr>\n')
                    html += '            </table>\n'

            # Render diagnostic image gallery
            if diag_images:
                # In zip mode images are referenced via relative paths; in embed mode as data URIs
                cat_key = f'diag_{diag_key}'
                gallery = render_image_gallery_html(
                    diag_images, mode=image_mode,
                    category=cat_key,
                    label=f'{label} Images',
                    max_width=max_thumb_width)
                html += gallery

            html += '        </div>\n'
        html += '    </div>\n'

    html += """
</body>
</html>
"""
    return html


def generate_text_report(aggregated: Dict[str, List[Dict]],
                         title: str,
                         verbose: bool = False) -> str:
    """Generate a plain text report from aggregated metrics."""
    aggregated = sort_steps(aggregated)

    lines = []
    lines.append("=" * 70)
    lines.append(title.center(70))
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(70))
    lines.append("=" * 70)
    lines.append("")

    _, error_count, warning_count, ok_count = compute_overall_status(aggregated)

    lines.append("SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Pipeline Steps: {len(aggregated)}")
    lines.append(f"  Total Metrics Files: {sum(len(v) for v in aggregated.values())}")
    lines.append(f"  Status: {get_status_emoji('ok')} OK: {ok_count}  "
                f"{get_status_emoji('warning')} Warnings: {warning_count}  "
                f"{get_status_emoji('error')} Errors: {error_count}")
    lines.append("")

    for step_name, metrics_list in aggregated.items():
        summary = compute_summary_statistics(metrics_list)
        step_status = get_step_status(metrics_list)

        lines.append("")
        lines.append(f"{get_status_emoji(step_status)} {step_name.replace('_', ' ').upper()}")
        lines.append("-" * 70)
        lines.append(f"  Items: {summary['count']} | Status: {step_status.upper()}")

        # Quality metrics stats
        quality_metrics, _ = separate_metrics_by_type(metrics_list)
        if quality_metrics:
            lines.append("")
            lines.append("  Quality Metrics:")
            lines.append(f"  {'Metric':<25} {'Mean':>12} {'Median':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
            lines.append("  " + "-" * 77)
            for metric_name, mdata in quality_metrics.items():
                entries = mdata['entries']
                numeric_vals = [e['value'] for e in entries
                                if isinstance(e.get('value'), (int, float))]
                if not numeric_vals:
                    continue
                arr = np.array(numeric_vals)
                name = metric_name[:25]
                lines.append(
                    f"  {name:<25} {format_value(float(np.mean(arr))):>12} "
                    f"{format_value(float(np.median(arr))):>12} "
                    f"{format_value(float(np.std(arr))):>12} "
                    f"{format_value(float(np.min(arr))):>12} "
                    f"{format_value(float(np.max(arr))):>12}"
                )

        all_warnings, all_errors = collect_issues(metrics_list)

        if all_errors:
            lines.append("")
            lines.append(f"  {get_status_emoji('error')} ERRORS:")
            for g in group_issues(all_errors):
                if g['count'] == 1:
                    lines.append(f"    - {g['details'][0]}")
                else:
                    vals = g['values']
                    val_str = f"range {min(vals):.3g}–{max(vals):.3g}" if vals else f"{g['count']} occurrences"
                    lines.append(f"    - {g['metric']}: {g['count']} slices ({val_str})")

        if all_warnings:
            lines.append("")
            lines.append(f"  {get_status_emoji('warning')} WARNINGS:")
            for g in group_issues(all_warnings):
                if g['count'] == 1:
                    lines.append(f"    - {g['details'][0]}")
                else:
                    vals = g['values']
                    val_str = f"range {min(vals):.3g}–{max(vals):.3g}" if vals else f"{g['count']} occurrences"
                    lines.append(f"    - {g['metric']}: {g['count']} slices ({val_str})")

        if verbose:
            lines.append("")
            lines.append("  Individual Results:")
            for m in metrics_list:
                source = extract_slice_id(m.get('source_file', 'unknown'))
                m_status = m.get('overall_status', 'unknown')
                lines.append(f"    {get_status_emoji(m_status)} {source}")
                for name, data in m.get('metrics', {}).items():
                    if isinstance(data, dict):
                        value = data.get('value', 'N/A')
                        unit = data.get('unit', '') or ''
                        lines.append(f"       {name}: {format_value(value)}{(' ' + unit) if unit else ''}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("End of Report".center(70))
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_report)

    if not input_dir.exists():
        parser.error(f"Input directory does not exist: {input_dir}")

    # Determine format
    if args.format == 'auto':
        suffix = output_file.suffix.lower()
        if suffix == '.html':
            output_format = 'html'
        elif suffix == '.zip':
            output_format = 'zip'
        else:
            output_format = 'text'
    else:
        output_format = args.format

    # Aggregate metrics from all subdirectories
    print(f"Scanning for metrics files in: {input_dir}")
    aggregated = aggregate_metrics(input_dir)

    if not aggregated:
        print("No metrics files found. Checking for process subdirectories...")
        for subdir in input_dir.iterdir():
            if subdir.is_dir():
                sub_aggregated = aggregate_metrics(subdir)
                for step, metrics in sub_aggregated.items():
                    if step not in aggregated:
                        aggregated[step] = []
                    aggregated[step].extend(metrics)

    if not aggregated:
        print("Warning: No metrics files found in the input directory.")
        print("Make sure the pipeline has been run with metrics collection enabled.")
        aggregated = {}

    print(f"Found {sum(len(v) for v in aggregated.values())} metrics files "
          f"across {len(aggregated)} pipeline steps")

    # Discover preview images — only for zip bundles; HTML is always image-free
    images: Dict[str, List[Path]] = {}
    if output_format == 'zip' and not args.no_images:
        images = discover_images(input_dir,
                                 overview_png=args.overview_png,
                                 annotated_png=args.annotated_png)
        total_imgs = sum(len(v) for v in images.values())
        if total_imgs:
            print(f"Found {total_imgs} preview image(s) to bundle in zip")

    # Zip bundles use relative image links; standalone HTML has no images
    image_mode = 'link'

    # Compute cross-slice aggregate trends
    trends = compute_cross_slice_trends(aggregated)
    if trends:
        n_trend_groups = len(trends)
        print(f"Computed {n_trend_groups} cross-slice trend group(s)")

    # Discover diagnostic outputs
    diagnostics = discover_diagnostic_data(input_dir)
    if diagnostics:
        print(f"Found {len(diagnostics)} diagnostic output(s): {', '.join(diagnostics.keys())}")
        # In zip mode, include diagnostic images in the bundle
        if output_format == 'zip' and not args.no_images:
            for diag_key, diag in diagnostics.items():
                cat_key = f'diag_{diag_key}'
                diag_imgs = diag.get('images', [])
                if diag_imgs:
                    images[cat_key] = diag_imgs

    # Generate report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_format in ('html', 'zip'):
        report = generate_html_report(
            aggregated, args.title, args.verbose,
            images=images,
            image_mode=image_mode,
            max_overview_width=args.max_overview_width,
            max_thumb_width=args.max_thumb_width,
            trends=trends if trends else None,
            diagnostics=diagnostics if diagnostics else None,
        )
        if output_format == 'zip':
            if output_file.suffix.lower() != '.zip':
                output_file = output_file.with_suffix('.zip')
            generate_zip_bundle(report, images, output_file)
        else:
            with open(output_file, 'w') as f:
                f.write(report)
    else:
        report = generate_text_report(aggregated, args.title, args.verbose)
        with open(output_file, 'w') as f:
            f.write(report)

    print(f"Report saved to: {output_file}")

    _, error_count, warning_count, _ = compute_overall_status(aggregated)

    if error_count > 0:
        print(f"\n{get_status_emoji('error')} {error_count} error(s) found - please review the report")
    elif warning_count > 0:
        print(f"\n{get_status_emoji('warning')} {warning_count} warning(s) found - please review the report")
    else:
        print(f"\n{get_status_emoji('ok')} All checks passed")


if __name__ == "__main__":
    main()
