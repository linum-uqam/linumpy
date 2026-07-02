#!/usr/bin/env python3
"""
Generate a quality report from pipeline metrics.

This script aggregates metrics from various pipeline steps and generates
a comprehensive report in HTML or text format to help identify potential
issues in the 3D reconstruction pipeline.
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
from pathlib import Path

from linumpy.diagnostics.pipeline_report import (
    STEP_DESCRIPTIONS,
    STEP_DISPLAY_NAMES,
    STEP_ORDER,
    STEP_PREVIEW_CATEGORY,
    collect_issues,
    compute_cross_slice_trends,
    compute_overall_status,
    discover_diagnostic_data,
    discover_images,
    discover_interpolation_data,
    discover_slice_config_summary,
    extract_slice_id,
    format_value,
    generate_html_report,
    generate_sparkline_svg,
    generate_text_report,
    generate_trend_line_svg,
    generate_zip_bundle,
    get_status_color,
    get_status_emoji,
    get_step_status,
    group_issues,
    image_to_data_uri,
    parse_issue,
    render_image_gallery_html,
    separate_metrics_by_type,
    slug,
    sort_steps,
)
from linumpy.metrics import aggregate_metrics

__all__ = [
    "STEP_DESCRIPTIONS",
    "STEP_DISPLAY_NAMES",
    "STEP_ORDER",
    "STEP_PREVIEW_CATEGORY",
    "collect_issues",
    "compute_cross_slice_trends",
    "compute_overall_status",
    "discover_diagnostic_data",
    "discover_images",
    "discover_interpolation_data",
    "discover_slice_config_summary",
    "extract_slice_id",
    "format_value",
    "generate_html_report",
    "generate_sparkline_svg",
    "generate_text_report",
    "generate_trend_line_svg",
    "generate_zip_bundle",
    "get_status_color",
    "get_status_emoji",
    "get_step_status",
    "group_issues",
    "image_to_data_uri",
    "main",
    "parse_issue",
    "render_image_gallery_html",
    "separate_metrics_by_type",
    "slug",
    "sort_steps",
]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_dir", help="Input directory containing pipeline output with metrics files.")
    p.add_argument("output_report", help="Output report file path (.html, .zip, or .txt)")
    p.add_argument(
        "--format",
        choices=["html", "text", "zip", "auto"],
        default="auto",
        help="Output format. 'auto' infers from extension. [%(default)s]",
    )
    p.add_argument("--title", default="Pipeline Quality Report", help="Report title. [%(default)s]")
    p.add_argument("--verbose", action="store_true", help="Include all metric details in the report.")
    p.add_argument(
        "--overview_png", type=Path, default=None, help="Path to the main volume PNG screenshot (embedded in summary)."
    )
    p.add_argument(
        "--annotated_png", type=Path, default=None, help="Path to the annotated volume PNG screenshot (embedded in summary)."
    )
    p.add_argument("--max_overview_width", type=int, default=900, help="Max pixel width for overview images. [%(default)s]")
    p.add_argument("--max_thumb_width", type=int, default=380, help="Max pixel width for gallery thumbnails. [%(default)s]")
    p.add_argument("--no_images", action="store_true", help="Disable image discovery for zip bundles.")
    return p


def main() -> None:
    """Run function."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_report)

    if not input_dir.exists():
        parser.error(f"Input directory does not exist: {input_dir}")

    # Determine format
    if args.format == "auto":
        suffix = output_file.suffix.lower()
        if suffix == ".html":
            output_format = "html"
        elif suffix == ".zip":
            output_format = "zip"
        else:
            output_format = "text"
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

    print(f"Found {sum(len(v) for v in aggregated.values())} metrics files across {len(aggregated)} pipeline steps")

    # Discover preview images -- only for zip bundles; HTML is always image-free
    images: dict[str, list[Path]] = {}
    if output_format == "zip" and not args.no_images:
        images = discover_images(input_dir, overview_png=args.overview_png, annotated_png=args.annotated_png)
        total_imgs = sum(len(v) for v in images.values())
        if total_imgs:
            print(f"Found {total_imgs} preview image(s) to bundle in zip")

    # Zip bundles use relative image links; standalone HTML has no images
    image_mode = "link"

    # Compute cross-slice aggregate trends
    trends = compute_cross_slice_trends(aggregated)
    if trends:
        n_trend_groups = len(trends)
        print(f"Computed {n_trend_groups} cross-slice trend group(s)")

    # Discover slice-interpolation outputs
    interpolation = discover_interpolation_data(input_dir)
    if interpolation:
        s = interpolation["summary"]
        print(f"Found interpolation output(s): {s['count']} slice(s), {s['n_with_fallback']} with fallback")
        if output_format == "zip" and not args.no_images and interpolation.get("images"):
            images["diag_interpolate_missing_slice"] = list(interpolation["images"])

    # Discover slice_config summary (deepest-enriched slice_config CSV)
    slice_config_summary = discover_slice_config_summary(input_dir)
    if slice_config_summary:
        print(f"Found slice_config summary: {slice_config_summary['source']}")

    # Discover diagnostic outputs
    diagnostics = discover_diagnostic_data(input_dir)
    if diagnostics:
        print(f"Found {len(diagnostics)} diagnostic output(s): {', '.join(diagnostics.keys())}")
        # In zip mode, include diagnostic images in the bundle
        if output_format == "zip" and not args.no_images:
            for diag_key, diag in diagnostics.items():
                cat_key = f"diag_{diag_key}"
                diag_imgs = diag.get("images", [])
                if diag_imgs:
                    images[cat_key] = diag_imgs

    # Generate report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_format in ("html", "zip"):
        report = generate_html_report(
            aggregated,
            args.title,
            args.verbose,
            images=images,
            image_mode=image_mode,
            max_overview_width=args.max_overview_width,
            max_thumb_width=args.max_thumb_width,
            trends=trends or None,
            diagnostics=diagnostics or None,
            interpolation=interpolation,
            slice_config_summary=slice_config_summary,
        )
        if output_format == "zip":
            if output_file.suffix.lower() != ".zip":
                output_file = output_file.with_suffix(".zip")
            generate_zip_bundle(report, images, output_file)
        else:
            Path(output_file).write_text(report)
    else:
        report = generate_text_report(
            aggregated,
            args.title,
            args.verbose,
            interpolation=interpolation,
            slice_config_summary=slice_config_summary,
        )
        Path(output_file).write_text(report)

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
