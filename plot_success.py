#!/usr/bin/env python3
"""
Plot success rates from test result files.

Handles three formats automatically:
  1. HAPPO test_results.txt (structured):
       10M       Success Rate: 0.0198 (1.98%)  [6/300 episodes]
  2. MAPPO success_rate.txt (verbose logs):
       Loaded checkpoint from: .../rl_model_10000000_steps/module.pt
       success rate: 0.19333334267139435
  3. Upper-level HAPPO test_results.txt (table format):
       100M        71.00%     2.601525   19.287062   0.010786

Usage:
    python plot_success.py <file1> [file2 ...]
    python plot_success.py results/mapush/.../test_results.txt
    python plot_success.py file1.txt file2.txt --labels "HAPPO" "MAPPO"

Options:
    --labels NAME [NAME ...]   Custom legend labels (one per file)
    --dpi INT                  DPI for saved image. Default: 150
    -o, --output PATH          Output file path override
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def parse_step_label(label: str) -> int:
    """Convert '10M', '200M', '1500K', '500K' etc to integer steps."""
    label = label.strip().upper()
    if label.endswith('M'):
        return int(float(label[:-1]) * 1_000_000)
    elif label.endswith('K'):
        return int(float(label[:-1]) * 1_000)
    else:
        return int(label)


def parse_happo_format(text: str):
    """Parse structured HAPPO test_results.txt format.

    Lines like:
        10M       Success Rate: 0.0198 (1.98%)  [6/300 episodes]
    """
    pattern = re.compile(
        r'^\s*(\d+[MKmk]?)\s+Success Rate:\s+([\d.]+)',
        re.MULTILINE
    )
    results = []
    for m in pattern.finditer(text):
        step = parse_step_label(m.group(1))
        rate = float(m.group(2))
        results.append((step, rate))
    return results


def parse_mappo_format(text: str):
    """Parse verbose MAPPO success_rate.txt format.

    Pairs of:
        Loaded checkpoint from: .../rl_model_10000000_steps/module.pt
        success rate: 0.19333334267139435
    """
    # Find all (checkpoint_path, success_rate) pairs
    # The success rate line appears right after the checkpoint load line
    # but there are duplicates — take only the first success rate after each checkpoint line
    checkpoint_pattern = re.compile(r'rl_model_(\d+)_steps')
    success_pattern = re.compile(r'^success rate:\s+([\d.]+)', re.MULTILINE)

    # Strategy: walk lines, track current checkpoint, grab first success rate after it
    lines = text.split('\n')
    results = []
    current_step = None
    seen_rate_for_current = False

    for line in lines:
        cp_match = checkpoint_pattern.search(line)
        if cp_match:
            current_step = int(cp_match.group(1))
            seen_rate_for_current = False
            continue

        if current_step is not None and not seen_rate_for_current:
            sr_match = success_pattern.match(line)
            if sr_match:
                rate = float(sr_match.group(1))
                results.append((current_step, rate))
                seen_rate_for_current = True

    return results


def parse_upper_happo_format(text: str):
    """Parse upper-level HAPPO test_results.txt table format.

    Lines like:
        100M        71.00%     2.601525   19.287062   0.010786
    """
    pattern = re.compile(
        r'^\s*(\d+[MKmk]?)\s+([\d.]+)%',
        re.MULTILINE
    )
    results = []
    for m in pattern.finditer(text):
        step = parse_step_label(m.group(1))
        rate = float(m.group(2)) / 100.0  # Convert percentage to fraction
        results.append((step, rate))
    return results


def detect_and_parse(filepath: str):
    """Auto-detect format and parse the file."""
    text = Path(filepath).read_text()

    # Detect format: HAPPO has "Success Rate:" lines, upper-level has "success%" table header,
    # MAPPO has "success rate:" + "rl_model_"
    if re.search(r'^\s*\d+[MKmk]?\s+Success Rate:', text, re.MULTILINE):
        results = parse_happo_format(text)
        fmt = 'happo'
    elif re.search(r'Checkpoint\s+success%', text):
        results = parse_upper_happo_format(text)
        fmt = 'upper_happo'
    elif 'rl_model_' in text and 'success rate:' in text:
        results = parse_mappo_format(text)
        fmt = 'mappo'
    else:
        print(f"ERROR: Could not detect format in {filepath}")
        sys.exit(1)

    if not results:
        print(f"ERROR: No data parsed from {filepath}")
        sys.exit(1)

    # Deduplicate: if same step appears multiple times, keep last occurrence
    # (handles multiple runs in the same file — last run is usually the final one)
    seen = {}
    for step, rate in results:
        seen[step] = rate
    results = sorted(seen.items(), key=lambda x: x[0])

    print(f"  Parsed {filepath} ({fmt} format): {len(results)} checkpoints")
    return results


def format_step(x, _):
    """Format step numbers as 10M, 100K, etc."""
    if x >= 1e6:
        return f'{x/1e6:.0f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}K'
    return f'{x:.0f}'


def plot_success_rates(all_data, labels, out_path, dpi=150):
    """Plot success rate curves."""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (data, label) in enumerate(zip(all_data, labels)):
        steps = [d[0] for d in data]
        rates = [d[1] * 100 for d in data]  # Convert to percentage
        color = colors[i % len(colors)]

        ax.plot(steps, rates, color=color, linewidth=2.0, label=label,
                marker='o', markersize=4, markeredgewidth=0)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate vs Training Steps', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_step))
    ax.set_ylim(-2, 105)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

    if len(all_data) > 1:
        ax.legend(fontsize=11, loc='best')

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot success rates from test result files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('files', nargs='+', help='Test result file(s)')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Legend labels (one per file)')
    parser.add_argument('--dpi', type=int, default=150, help='Image DPI. Default: 150')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output file path override')

    args = parser.parse_args()

    # Parse all files
    all_data = []
    for f in args.files:
        if not Path(f).exists():
            print(f"ERROR: File not found: {f}")
            sys.exit(1)
        data = detect_and_parse(f)
        all_data.append(data)

    # Labels
    if args.labels:
        labels = args.labels
    else:
        labels = [Path(f).parent.name for f in args.files]
    while len(labels) < len(all_data):
        labels.append(f'run_{len(labels)}')

    # Output path
    if args.output:
        out_path = args.output
    elif len(args.files) == 1:
        # Save next to the input file
        p = Path(args.files[0])
        out_path = str(p.parent / 'success_rate.png')
    else:
        out_path = 'success_rate_comparison.png'

    plot_success_rates(all_data, labels, out_path, dpi=args.dpi)


if __name__ == '__main__':
    main()
