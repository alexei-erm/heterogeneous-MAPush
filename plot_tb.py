#!/usr/bin/env python3
"""
Plot all TensorBoard metrics from a run directory as PNG files.

Usage:
    python plot_tb.py <run_dir> [run_dir2 ...]
    python plot_tb.py results/models/baseline_mappo/mid
    python plot_tb.py results/models/baseline_mappo/mid results/models/baseline_happo/mid --labels "MAPPO" "HAPPO"

Options:
    --labels NAME [NAME ...]   Custom legend labels (one per run_dir)
    --smooth FLOAT             Exponential moving average weight (0=none, 0.9=heavy). Default: 0.6
    --dpi INT                  DPI for saved images. Default: 150
    --format EXT               Image format: png, pdf, svg. Default: png
    --no-show                  Don't display info about saved plots
    --tags TAG [TAG ...]       Only plot these tags (substring match)
    --exclude TAG [TAG ...]    Exclude tags containing these substrings
    --plot_agent_together      Overlay agent0/X and agent1/X on the same plot

Single run:  saves to <run_dir>/plots/<tag_path>.png
Multi run:   saves to plots_comparison/<tag_path>.png (in cwd)
"""

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from tbparse import SummaryReader


def ema_smooth(values, weight=0.6):
    """Exponential moving average smoothing."""
    if weight <= 0:
        return values
    smoothed = []
    last = values.iloc[0] if len(values) > 0 else 0
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return pd.Series(smoothed, index=values.index)


def format_step_label(x, _):
    """Format step numbers as 10M, 100M, etc."""
    if x >= 1e6:
        return f'{x/1e6:.0f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}K'
    return f'{x:.0f}'


def load_run(log_dir):
    """Load all scalar data from a TensorBoard log directory."""
    reader = SummaryReader(log_dir, pivot=False)
    df = reader.scalars
    if df.empty:
        print(f"  WARNING: No scalar data found in {log_dir}")
    return df


def tag_to_path(tag):
    """Convert a TensorBoard tag to a file path.
    'rewards/push_reward' -> 'rewards/push_reward'
    'average_step_reward' -> 'average_step_reward'
    'agent0/policy_loss'  -> 'agent0/policy_loss'
    """
    # Replace any characters that are problematic in filenames
    return tag.replace(' ', '_')


def plot_single_run(df, tag, out_path, smooth_weight=0.6, dpi=150, fmt='png'):
    """Plot a single metric from one run."""
    tag_df = df[df['tag'] == tag].sort_values('step')
    if tag_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    steps = tag_df['step'].values
    values = tag_df['value']

    # Raw data (faint)
    ax.plot(steps, values, alpha=0.25, color='#1f77b4', linewidth=0.8)

    # Smoothed data
    if smooth_weight > 0 and len(values) > 3:
        smoothed = ema_smooth(values.reset_index(drop=True), smooth_weight)
        ax.plot(steps, smoothed.values, color='#1f77b4', linewidth=2.0)

    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel(tag.split('/')[-1], fontsize=12)
    ax.set_title(tag, fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_step_label))
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', format=fmt)
    plt.close(fig)


def plot_agents_together(df, agent_tags, base_metric, out_path, smooth_weight=0.6, dpi=150, fmt='png'):
    """Plot multiple agent tags (e.g. agent0/X, agent1/X) on the same figure."""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    fig, ax = plt.subplots(figsize=(10, 5))
    has_data = False

    for i, tag in enumerate(sorted(agent_tags)):
        tag_df = df[df['tag'] == tag].sort_values('step')
        if tag_df.empty:
            continue
        has_data = True

        color = colors[i % len(colors)]
        steps = tag_df['step'].values
        values = tag_df['value']
        label = tag.rsplit('/', 1)[0] if '/' in tag else tag  # e.g. "agent0"

        # Raw data (faint)
        ax.plot(steps, values, alpha=0.15, color=color, linewidth=0.8)

        # Smoothed data
        if smooth_weight > 0 and len(values) > 3:
            smoothed = ema_smooth(values.reset_index(drop=True), smooth_weight)
            ax.plot(steps, smoothed.values, color=color, linewidth=2.0, label=label)
        else:
            ax.plot(steps, values, color=color, linewidth=1.5, label=label)

    if not has_data:
        plt.close(fig)
        return

    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel(base_metric, fontsize=12)
    ax.set_title(f'agents/{base_metric}', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_step_label))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', format=fmt)
    plt.close(fig)


def plot_agents_together_multi_run(runs_data, agent_tags_per_run, base_metric, out_path, labels,
                                   smooth_weight=0.6, dpi=150, fmt='png'):
    """Plot agent tags from multiple runs overlaid on the same figure."""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    fig, ax = plt.subplots(figsize=(10, 5))
    has_data = False
    color_idx = 0

    for run_idx, (df, run_label) in enumerate(zip(runs_data, labels)):
        agent_tags = agent_tags_per_run[run_idx]
        for tag in sorted(agent_tags):
            tag_df = df[df['tag'] == tag].sort_values('step')
            if tag_df.empty:
                continue
            has_data = True

            color = colors[color_idx % len(colors)]
            color_idx += 1
            steps = tag_df['step'].values
            values = tag_df['value']
            agent_name = tag.rsplit('/', 1)[0] if '/' in tag else tag
            label = f'{run_label} / {agent_name}'

            ax.plot(steps, values, alpha=0.15, color=color, linewidth=0.8)

            if smooth_weight > 0 and len(values) > 3:
                smoothed = ema_smooth(values.reset_index(drop=True), smooth_weight)
                ax.plot(steps, smoothed.values, color=color, linewidth=2.0, label=label)
            else:
                ax.plot(steps, values, color=color, linewidth=1.5, label=label)

    if not has_data:
        plt.close(fig)
        return

    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel(base_metric, fontsize=12)
    ax.set_title(f'agents/{base_metric}', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_step_label))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', format=fmt)
    plt.close(fig)


def plot_multi_run(runs_data, tag, out_path, labels, smooth_weight=0.6, dpi=150, fmt='png'):
    """Plot a single metric from multiple runs overlaid."""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    fig, ax = plt.subplots(figsize=(10, 5))
    has_data = False

    for i, (df, label) in enumerate(zip(runs_data, labels)):
        tag_df = df[df['tag'] == tag].sort_values('step')
        if tag_df.empty:
            continue
        has_data = True

        color = colors[i % len(colors)]
        steps = tag_df['step'].values
        values = tag_df['value']

        # Raw data (faint)
        ax.plot(steps, values, alpha=0.15, color=color, linewidth=0.8)

        # Smoothed data
        if smooth_weight > 0 and len(values) > 3:
            smoothed = ema_smooth(values.reset_index(drop=True), smooth_weight)
            ax.plot(steps, smoothed.values, color=color, linewidth=2.0, label=label)
        else:
            ax.plot(steps, values, color=color, linewidth=1.5, label=label)

    if not has_data:
        plt.close(fig)
        return

    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel(tag.split('/')[-1], fontsize=12)
    ax.set_title(tag, fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_step_label))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', format=fmt)
    plt.close(fig)


def find_logs_dir(run_dir):
    """Find the TensorBoard logs directory within a run directory."""
    run_dir = Path(run_dir)

    # Direct logs/ subdirectory
    if (run_dir / 'logs').is_dir():
        return str(run_dir / 'logs')

    # Maybe the path itself IS the logs dir
    events = list(run_dir.rglob('events.out.tfevents.*'))
    if events:
        return str(run_dir)

    return None


def main():
    parser = argparse.ArgumentParser(
        description='Plot TensorBoard metrics as PNG files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('run_dirs', nargs='+', help='Run directory (containing logs/ folder)')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Legend labels for multi-run comparison')
    parser.add_argument('--smooth', type=float, default=0.6,
                        help='EMA smoothing weight (0=none, 0.9=heavy). Default: 0.6')
    parser.add_argument('--dpi', type=int, default=150, help='Image DPI. Default: 150')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'pdf', 'svg'],
                        help='Image format. Default: png')
    parser.add_argument('--no-show', action='store_true', help='Suppress output')
    parser.add_argument('--tags', nargs='+', default=None,
                        help='Only plot tags containing these substrings')
    parser.add_argument('--exclude', nargs='+', default=None,
                        help='Exclude tags containing these substrings')
    parser.add_argument('--plot_agent_together', action='store_true',
                        help='Overlay per-agent metrics (e.g. agent0/X, agent1/X) on the same plot')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output directory override')

    args = parser.parse_args()

    multi_run = len(args.run_dirs) > 1

    # Load all runs
    runs_data = []
    run_dirs_resolved = []

    for rd in args.run_dirs:
        logs_dir = find_logs_dir(rd)
        if logs_dir is None:
            print(f"ERROR: No TensorBoard logs found in {rd}")
            sys.exit(1)

        if not args.no_show:
            print(f"Loading: {rd}")
        df = load_run(logs_dir)
        if df.empty:
            print(f"  WARNING: Empty data in {rd}, skipping")
            continue
        runs_data.append(df)
        run_dirs_resolved.append(rd)

    if not runs_data:
        print("ERROR: No data loaded from any run directory")
        sys.exit(1)

    # Labels
    if args.labels:
        labels = args.labels
    else:
        labels = [Path(rd).name for rd in run_dirs_resolved]

    # Pad labels if needed
    while len(labels) < len(runs_data):
        labels.append(f'run_{len(labels)}')

    # Collect all unique tags across runs
    all_tags = set()
    for df in runs_data:
        all_tags.update(df['tag'].unique())
    all_tags = sorted(all_tags)

    # Filter tags
    if args.tags:
        all_tags = [t for t in all_tags if any(sub in t for sub in args.tags)]
    if args.exclude:
        all_tags = [t for t in all_tags if not any(sub in t for sub in args.exclude)]

    if not all_tags:
        print("ERROR: No tags match the filter criteria")
        sys.exit(1)

    # Output directory
    if args.output:
        out_base = Path(args.output)
    elif multi_run:
        out_base = Path('plots_comparison')
    else:
        out_base = Path(run_dirs_resolved[0]) / 'plots'

    if not args.no_show:
        print(f"Output: {out_base}/")
        print(f"Tags:   {len(all_tags)} metrics")
        print(f"Smooth: {args.smooth}")
        print()

    # Group agent tags if --plot_agent_together
    # Detect pattern: "agentN/metric" where N is a digit
    agent_pattern = re.compile(r'^(agent\d+)/(.+)$')

    agent_grouped = {}   # base_metric -> list of full tags
    non_agent_tags = []

    if args.plot_agent_together:
        for tag in all_tags:
            m = agent_pattern.match(tag)
            if m:
                base_metric = m.group(2)
                agent_grouped.setdefault(base_metric, []).append(tag)
            else:
                non_agent_tags.append(tag)
    else:
        non_agent_tags = all_tags

    # Plot each non-agent tag normally
    count = 0
    for tag in non_agent_tags:
        rel_path = tag_to_path(tag)
        out_path = out_base / f'{rel_path}.{args.format}'

        if multi_run:
            plot_multi_run(runs_data, tag, str(out_path), labels,
                          smooth_weight=args.smooth, dpi=args.dpi, fmt=args.format)
        else:
            plot_single_run(runs_data[0], tag, str(out_path),
                           smooth_weight=args.smooth, dpi=args.dpi, fmt=args.format)
        count += 1
        if not args.no_show:
            print(f"  saved: {out_path}")

    # Plot grouped agent metrics
    for base_metric, agent_tags in sorted(agent_grouped.items()):
        rel_path = tag_to_path(f'agents_combined/{base_metric}')
        out_path = out_base / f'{rel_path}.{args.format}'

        if multi_run:
            # Collect per-run agent tags
            agent_tags_per_run = []
            for df in runs_data:
                run_tags_available = set(df['tag'].unique())
                agent_tags_per_run.append([t for t in agent_tags if t in run_tags_available])
            plot_agents_together_multi_run(runs_data, agent_tags_per_run, base_metric,
                                          str(out_path), labels,
                                          smooth_weight=args.smooth, dpi=args.dpi, fmt=args.format)
        else:
            plot_agents_together(runs_data[0], agent_tags, base_metric,
                                str(out_path),
                                smooth_weight=args.smooth, dpi=args.dpi, fmt=args.format)
        count += 1
        if not args.no_show:
            print(f"  saved: {out_path}  (agents combined: {', '.join(agent_tags)})")

    if not args.no_show:
        print(f"\nDone! {count} plots saved to {out_base}/")


if __name__ == '__main__':
    main()
