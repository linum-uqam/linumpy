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
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from linumpy.utils.metrics import aggregate_metrics, compute_summary_statistics


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_dir",
                   help="Input directory containing pipeline output with metrics files.")
    p.add_argument("output_report",
                   help="Output report file path (.html or .txt)")
    p.add_argument("--format", choices=['html', 'text', 'auto'], default='auto',
                   help="Output format. 'auto' infers from extension. (default=%(default)s)")
    p.add_argument("--title", default="Pipeline Quality Report",
                   help="Report title. (default=%(default)s)")
    p.add_argument("--verbose", action='store_true',
                   help="Include all metric details in the report.")
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


def generate_html_report(aggregated: Dict[str, List[Dict]],
                         title: str,
                         verbose: bool = False) -> str:
    """Generate an HTML report from aggregated metrics."""

    # Calculate overall status using helper
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
            margin-bottom: 15px;
        }}
        .step-title {{ font-size: 1.3em; font-weight: bold; }}
        .status-badge {{
            padding: 5px 15px;
            border-radius: 20px;
            color: white;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .metrics-table th, .metrics-table td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        .metrics-table th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        .metric-status {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
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
        .warnings-section {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 15px;
            border-radius: 5px;
            margin-top: 15px;
        }}
        .errors-section {{
            background: #f8d7da;
            border: 1px solid #dc3545;
            padding: 15px;
            border-radius: 5px;
            margin-top: 15px;
        }}
        .issue-item {{
            padding: 5px 0;
            border-bottom: 1px solid rgba(0,0,0,0.1);
        }}
        .issue-item:last-child {{ border-bottom: none; }}
        .collapsible {{
            cursor: pointer;
            user-select: none;
        }}
        .collapsible:hover {{ background: #f0f0f0; }}
        .content {{ display: none; padding: 10px; background: #fafafa; }}
        .show {{ display: block; }}
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

    # Generate section for each step
    for step_name, metrics_list in sorted(aggregated.items()):
        summary = compute_summary_statistics(metrics_list)
        step_status = get_step_status(metrics_list)

        html += f"""
    <div class="step-section">
        <div class="step-header">
            <span class="step-title">{step_name.replace('_', ' ').title()}</span>
            <span class="status-badge" style="background-color: {get_status_color(step_status)};">
                {summary['count']} items - {step_status.upper()}
            </span>
        </div>
"""

        # Show statistics for numerical metrics
        numerical_stats = {k: v for k, v in summary.items()
                          if isinstance(v, dict) and 'mean' in v}

        if numerical_stats:
            html += """
        <h4>Statistics</h4>
        <table class="metrics-table">
            <tr>
                <th>Metric</th>
                <th>Mean</th>
                <th>Std</th>
                <th>Min</th>
                <th>Max</th>
            </tr>
"""
            for metric_name, stats in numerical_stats.items():
                html += f"""
            <tr>
                <td>{metric_name}</td>
                <td>{format_value(stats['mean'])}</td>
                <td>{format_value(stats['std'])}</td>
                <td>{format_value(stats['min'])}</td>
                <td>{format_value(stats['max'])}</td>
            </tr>
"""
            html += "        </table>\n"

        # Collect all warnings and errors using helper
        all_warnings, all_errors = collect_issues(metrics_list)

        if all_errors:
            html += """
        <div class="errors-section">
            <strong>Errors:</strong>
"""
            for error in all_errors:
                html += f'            <div class="issue-item">{error}</div>\n'
            html += "        </div>\n"

        if all_warnings:
            html += """
        <div class="warnings-section">
            <strong>Warnings:</strong>
"""
            for warning in all_warnings:
                html += f'            <div class="issue-item">{warning}</div>\n'
            html += "        </div>\n"

        # Show individual metrics if verbose
        if verbose:
            html += """
        <h4>Individual Results</h4>
"""
            for m in metrics_list:
                source = Path(m.get('source_file', 'unknown')).name
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
                        unit = data.get('unit', '')
                        status = data.get('status', 'info')
                        html += f"""
                <tr>
                    <td>
                        <span class="metric-status" style="background-color: {get_status_color(status)};"></span>
                        {name}
                    </td>
                    <td>{format_value(value)} {unit}</td>
                </tr>
"""
                html += """            </table>
        </details>
"""

        html += "    </div>\n"

    html += """
    <script>
        // Collapsible sections
        document.querySelectorAll('.collapsible').forEach(item => {
            item.addEventListener('click', () => {
                item.nextElementSibling.classList.toggle('show');
            });
        });
    </script>
</body>
</html>
"""
    return html


def generate_text_report(aggregated: Dict[str, List[Dict]],
                         title: str,
                         verbose: bool = False) -> str:
    """Generate a plain text report from aggregated metrics."""

    lines = []
    lines.append("=" * 70)
    lines.append(title.center(70))
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(70))
    lines.append("=" * 70)
    lines.append("")

    # Calculate overall status using helper
    _, error_count, warning_count, ok_count = compute_overall_status(aggregated)

    lines.append("SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Pipeline Steps: {len(aggregated)}")
    lines.append(f"  Total Metrics Files: {sum(len(v) for v in aggregated.values())}")
    lines.append(f"  Status: {get_status_emoji('ok')} OK: {ok_count}  "
                f"{get_status_emoji('warning')} Warnings: {warning_count}  "
                f"{get_status_emoji('error')} Errors: {error_count}")
    lines.append("")

    # Generate section for each step
    for step_name, metrics_list in sorted(aggregated.items()):
        summary = compute_summary_statistics(metrics_list)
        step_status = get_step_status(metrics_list)

        lines.append("")
        lines.append(f"{get_status_emoji(step_status)} {step_name.replace('_', ' ').upper()}")
        lines.append("-" * 70)
        lines.append(f"  Items: {summary['count']} | Status: {step_status.upper()}")

        # Show statistics for numerical metrics
        numerical_stats = {k: v for k, v in summary.items()
                          if isinstance(v, dict) and 'mean' in v}

        if numerical_stats:
            lines.append("")
            lines.append("  Statistics:")
            lines.append(f"  {'Metric':<25} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
            lines.append("  " + "-" * 65)
            for metric_name, stats in numerical_stats.items():
                name = metric_name[:25]
                lines.append(f"  {name:<25} {format_value(stats['mean']):>12} "
                           f"{format_value(stats['std']):>12} "
                           f"{format_value(stats['min']):>12} "
                           f"{format_value(stats['max']):>12}")

        # Collect all warnings and errors using helper
        all_warnings, all_errors = collect_issues(metrics_list)

        if all_errors:
            lines.append("")
            lines.append(f"  {get_status_emoji('error')} ERRORS:")
            for error in all_errors:
                lines.append(f"    - {error}")

        if all_warnings:
            lines.append("")
            lines.append(f"  {get_status_emoji('warning')} WARNINGS:")
            for warning in all_warnings:
                lines.append(f"    - {warning}")

        if verbose:
            lines.append("")
            lines.append("  Individual Results:")
            for m in metrics_list:
                source = Path(m.get('source_file', 'unknown')).name
                m_status = m.get('overall_status', 'unknown')
                lines.append(f"    {get_status_emoji(m_status)} {source}")
                for name, data in m.get('metrics', {}).items():
                    if isinstance(data, dict):
                        value = data.get('value', 'N/A')
                        unit = data.get('unit', '')
                        lines.append(f"       {name}: {format_value(value)} {unit}")

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
        if output_file.suffix.lower() == '.html':
            output_format = 'html'
        else:
            output_format = 'text'
    else:
        output_format = args.format

    # Aggregate metrics from all subdirectories
    print(f"Scanning for metrics files in: {input_dir}")
    aggregated = aggregate_metrics(input_dir)

    if not aggregated:
        print("No metrics files found. Checking for process subdirectories...")
        # Try looking in common Nextflow output structure
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
        # Generate empty report
        aggregated = {}

    print(f"Found {sum(len(v) for v in aggregated.values())} metrics files "
          f"across {len(aggregated)} pipeline steps")

    # Generate report
    if output_format == 'html':
        report = generate_html_report(aggregated, args.title, args.verbose)
    else:
        report = generate_text_report(aggregated, args.title, args.verbose)

    # Save report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"Report saved to: {output_file}")

    # Print summary to console using helper
    _, error_count, warning_count, _ = compute_overall_status(aggregated)

    if error_count > 0:
        print(f"\n{get_status_emoji('error')} {error_count} error(s) found - please review the report")
    elif warning_count > 0:
        print(f"\n{get_status_emoji('warning')} {warning_count} warning(s) found - please review the report")
    else:
        print(f"\n{get_status_emoji('ok')} All checks passed")


if __name__ == "__main__":
    main()
