#!/usr/bin/env python3
"""
TensorBoard Log Analyzer for HARL/MAPush Training Runs

This script analyzes TensorBoard logs to diagnose training issues.
It extracts key metrics, computes statistics, and identifies potential problems.

Usage:
    python analyze_tensorboard_logs.py <log_dir>
    python analyze_tensorboard_logs.py <log_dir> --output report.txt
    python analyze_tensorboard_logs.py <log_dir> --json output.json

Example:
    python scripts/analyze_tensorboard_logs.py results/mapush/go1push_mid/happo/critic6/seed-00001-2025-12-19-16-53-40/logs

Author: Claude Code Analysis Tool
Date: December 2025
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("ERROR: tensorboard not installed. Run: pip install tensorboard")
    sys.exit(1)


@dataclass
class MetricStats:
    """Statistics for a single metric."""
    name: str
    count: int
    first_value: float
    last_value: float
    min_value: float
    max_value: float
    mean_value: float
    std_value: float
    first_step: int
    last_step: int
    # Trend analysis
    change_absolute: float = 0.0
    change_percent: float = 0.0
    trend: str = "stable"  # increasing, decreasing, stable, volatile
    # Convergence analysis
    is_converged: bool = False
    convergence_step: Optional[int] = None
    final_variance: float = 0.0


@dataclass
class DiagnosticResult:
    """Result of a diagnostic check."""
    name: str
    status: str  # "OK", "WARNING", "CRITICAL"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisReport:
    """Complete analysis report."""
    log_dir: str
    total_steps: int
    available_metrics: List[str]
    metric_stats: Dict[str, MetricStats]
    diagnostics: List[DiagnosticResult]
    summary: str


class TensorBoardAnalyzer:
    """Analyzer for TensorBoard log files."""

    # Known metric patterns for MAPush/HARL
    CRITICAL_METRICS = [
        'average_step_reward',
        'average_episode_reward',
        'mapush/success_rate',
        'success_rate',
    ]

    ENTROPY_METRICS = [
        'dist_entropy',
        'entropy',
    ]

    LOSS_METRICS = [
        'value_loss',
        'policy_loss',
        'critic_loss',
        'actor_loss',
    ]

    GRADIENT_METRICS = [
        'grad_norm',
        'actor_grad_norm',
        'critic_grad_norm',
    ]

    REWARD_COMPONENT_METRICS = [
        'rewards/',
        'reward/',
    ]

    def __init__(self, log_dir: str, size_guidance: Optional[Dict] = None):
        """
        Initialize the analyzer.

        Args:
            log_dir: Path to TensorBoard log directory
            size_guidance: Optional size guidance for event accumulator
        """
        self.log_dir = log_dir
        self.size_guidance = size_guidance or {
            event_accumulator.SCALARS: 0,  # Load all scalars
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
            event_accumulator.AUDIO: 0,
            event_accumulator.COMPRESSED_HISTOGRAMS: 0,
            event_accumulator.TENSORS: 0,
        }
        self.ea = None
        self.metrics_data: Dict[str, List[Tuple[int, float]]] = {}
        self.metric_stats: Dict[str, MetricStats] = {}
        self.summary_json_path = os.path.join(log_dir, "summary.json")

    def load(self) -> bool:
        """Load TensorBoard event files."""
        if not os.path.exists(self.log_dir):
            print(f"ERROR: Log directory not found: {self.log_dir}")
            return False

        try:
            self.ea = event_accumulator.EventAccumulator(
                self.log_dir,
                size_guidance=self.size_guidance
            )
            self.ea.Reload()
            return True
        except Exception as e:
            print(f"ERROR: Failed to load event files: {e}")
            return False

    def get_available_metrics(self) -> List[str]:
        """Get list of available scalar metrics."""
        if self.ea is None:
            return []
        return self.ea.Tags().get('scalars', [])

    def extract_metric(self, metric_name: str) -> List[Tuple[int, float]]:
        """Extract data for a specific metric."""
        if self.ea is None:
            return []

        try:
            scalars = self.ea.Scalars(metric_name)
            return [(s.step, s.value) for s in scalars]
        except KeyError:
            return []

    def extract_all_metrics(self) -> Dict[str, List[Tuple[int, float]]]:
        """Extract data for all available metrics."""
        metrics = self.get_available_metrics()
        self.metrics_data = {}

        for metric in metrics:
            data = self.extract_metric(metric)
            if data:
                self.metrics_data[metric] = data

        # Also try to load from summary.json if it exists
        # This captures metrics from nested subdirectories (like agent0/dist_entropy)
        self._load_from_summary_json()

        return self.metrics_data

    def _load_from_summary_json(self) -> None:
        """Load additional metrics from summary.json if available."""
        if not os.path.exists(self.summary_json_path):
            return

        try:
            with open(self.summary_json_path, 'r') as f:
                summary_data = json.load(f)

            for full_path, values in summary_data.items():
                # Extract a clean metric name from the path
                # Path format: "./results/.../logs/agent0/dist_entropy/agent0/dist_entropy"
                parts = full_path.split('/')
                if len(parts) >= 2:
                    # Get the last meaningful part
                    metric_name = parts[-1]
                    # Add agent prefix if present
                    for i, part in enumerate(parts):
                        if part.startswith('agent') and i < len(parts) - 1:
                            metric_name = f"{part}/{parts[-1]}"
                            break
                        if part == 'critic' and i < len(parts) - 1:
                            metric_name = f"critic/{parts[-1]}"
                            break
                else:
                    metric_name = full_path.split('/')[-1]

                # Skip if we already have this metric from TensorBoard directly
                if metric_name in self.metrics_data:
                    continue

                # Parse the values: [[timestamp, step, value], ...]
                if isinstance(values, list) and len(values) > 0:
                    data = []
                    for entry in values:
                        if isinstance(entry, list) and len(entry) >= 3:
                            step = int(entry[1])
                            value = float(entry[2])
                            data.append((step, value))

                    if data:
                        self.metrics_data[metric_name] = data

        except Exception as e:
            print(f"Warning: Could not load summary.json: {e}")

    def compute_metric_stats(self, name: str, data: List[Tuple[int, float]]) -> MetricStats:
        """Compute statistics for a metric."""
        if not data:
            return MetricStats(
                name=name, count=0, first_value=0, last_value=0,
                min_value=0, max_value=0, mean_value=0, std_value=0,
                first_step=0, last_step=0
            )

        steps = [d[0] for d in data]
        values = [d[1] for d in data]
        values_arr = np.array(values)

        # Basic stats
        stats = MetricStats(
            name=name,
            count=len(data),
            first_value=values[0],
            last_value=values[-1],
            min_value=float(np.min(values_arr)),
            max_value=float(np.max(values_arr)),
            mean_value=float(np.mean(values_arr)),
            std_value=float(np.std(values_arr)),
            first_step=steps[0],
            last_step=steps[-1],
        )

        # Change analysis
        stats.change_absolute = stats.last_value - stats.first_value
        if abs(stats.first_value) > 1e-10:
            stats.change_percent = (stats.change_absolute / abs(stats.first_value)) * 100
        else:
            stats.change_percent = 0.0 if abs(stats.change_absolute) < 1e-10 else float('inf')

        # Trend analysis
        stats.trend = self._analyze_trend(values_arr)

        # Convergence analysis
        stats.is_converged, stats.convergence_step = self._analyze_convergence(steps, values_arr)

        # Final variance (last 10% of data)
        final_portion = values_arr[int(len(values_arr) * 0.9):]
        if len(final_portion) > 1:
            stats.final_variance = float(np.var(final_portion))

        return stats

    def _analyze_trend(self, values: np.ndarray) -> str:
        """Analyze the trend of a metric."""
        if len(values) < 10:
            return "insufficient_data"

        # Split into quarters and compare
        q1 = np.mean(values[:len(values)//4])
        q4 = np.mean(values[3*len(values)//4:])

        # Calculate coefficient of variation for volatility
        cv = np.std(values) / (abs(np.mean(values)) + 1e-10)

        if cv > 0.5:
            return "volatile"

        relative_change = (q4 - q1) / (abs(q1) + 1e-10)

        if relative_change > 0.1:
            return "increasing"
        elif relative_change < -0.1:
            return "decreasing"
        else:
            return "stable"

    def _analyze_convergence(self, steps: List[int], values: np.ndarray) -> Tuple[bool, Optional[int]]:
        """Check if metric has converged."""
        if len(values) < 20:
            return False, None

        # Check if last 20% has low variance compared to overall
        final_portion = values[int(len(values) * 0.8):]
        overall_range = np.max(values) - np.min(values)
        final_range = np.max(final_portion) - np.min(final_portion)

        if overall_range < 1e-10:
            return True, steps[0]

        # Converged if final range is less than 10% of overall range
        if final_range / overall_range < 0.1:
            # Find approximate convergence point
            window_size = max(5, len(values) // 20)
            for i in range(len(values) - window_size):
                window = values[i:i+window_size]
                if (np.max(window) - np.min(window)) / overall_range < 0.15:
                    return True, steps[i]
            return True, steps[int(len(steps) * 0.8)]

        return False, None

    def compute_all_stats(self) -> Dict[str, MetricStats]:
        """Compute statistics for all metrics."""
        if not self.metrics_data:
            self.extract_all_metrics()

        self.metric_stats = {}
        for name, data in self.metrics_data.items():
            self.metric_stats[name] = self.compute_metric_stats(name, data)

        return self.metric_stats

    def run_diagnostics(self) -> List[DiagnosticResult]:
        """Run diagnostic checks on the training run."""
        if not self.metric_stats:
            self.compute_all_stats()

        diagnostics = []

        # Check 1: Success rate
        diagnostics.append(self._check_success_rate())

        # Check 2: Reward trend
        diagnostics.append(self._check_reward_trend())

        # Check 3: Entropy behavior
        diagnostics.append(self._check_entropy())

        # Check 4: Value loss
        diagnostics.append(self._check_value_loss())

        # Check 5: Policy loss
        diagnostics.append(self._check_policy_loss())

        # Check 6: Gradient norms
        diagnostics.append(self._check_gradients())

        # Check 7: Reward components
        diagnostics.append(self._check_reward_components())

        # Filter out None results
        return [d for d in diagnostics if d is not None]

    def _find_metric(self, patterns: List[str]) -> Optional[MetricStats]:
        """Find a metric matching any of the given patterns."""
        for name, stats in self.metric_stats.items():
            for pattern in patterns:
                if pattern in name.lower():
                    return stats
        return None

    def _find_all_metrics(self, patterns: List[str]) -> List[MetricStats]:
        """Find all metrics matching any of the given patterns."""
        results = []
        for name, stats in self.metric_stats.items():
            for pattern in patterns:
                if pattern in name.lower():
                    results.append(stats)
                    break
        return results

    def _check_success_rate(self) -> Optional[DiagnosticResult]:
        """Check success rate metric."""
        stats = self._find_metric(['success_rate', 'success'])

        if stats is None:
            return DiagnosticResult(
                name="Success Rate",
                status="WARNING",
                message="No success rate metric found",
                details={}
            )

        if stats.last_value < 0.01:
            return DiagnosticResult(
                name="Success Rate",
                status="CRITICAL",
                message=f"Success rate is near zero ({stats.last_value:.4f}). Training has not learned the task.",
                details={
                    "first_value": stats.first_value,
                    "last_value": stats.last_value,
                    "max_value": stats.max_value,
                    "trend": stats.trend
                }
            )
        elif stats.last_value < 0.5:
            return DiagnosticResult(
                name="Success Rate",
                status="WARNING",
                message=f"Success rate is low ({stats.last_value:.4f}). Training may need more steps or hyperparameter tuning.",
                details={
                    "first_value": stats.first_value,
                    "last_value": stats.last_value,
                    "max_value": stats.max_value,
                    "trend": stats.trend
                }
            )
        else:
            return DiagnosticResult(
                name="Success Rate",
                status="OK",
                message=f"Success rate is good ({stats.last_value:.4f})",
                details={
                    "first_value": stats.first_value,
                    "last_value": stats.last_value,
                    "trend": stats.trend
                }
            )

    def _check_reward_trend(self) -> Optional[DiagnosticResult]:
        """Check reward trend."""
        stats = self._find_metric(['average_step_reward', 'average_episode_reward', 'reward'])

        if stats is None:
            return DiagnosticResult(
                name="Reward Trend",
                status="WARNING",
                message="No reward metric found",
                details={}
            )

        if stats.trend == "decreasing" and stats.change_percent < -20:
            return DiagnosticResult(
                name="Reward Trend",
                status="CRITICAL",
                message=f"Reward is decreasing significantly ({stats.change_percent:.1f}%). Training may be diverging.",
                details={
                    "first_value": stats.first_value,
                    "last_value": stats.last_value,
                    "change_percent": stats.change_percent,
                    "trend": stats.trend
                }
            )
        elif stats.trend == "stable" and abs(stats.change_percent) < 5:
            return DiagnosticResult(
                name="Reward Trend",
                status="WARNING",
                message=f"Reward is stagnant (change: {stats.change_percent:.1f}%). Policy may not be learning.",
                details={
                    "first_value": stats.first_value,
                    "last_value": stats.last_value,
                    "change_percent": stats.change_percent,
                    "trend": stats.trend
                }
            )
        elif stats.trend == "increasing":
            return DiagnosticResult(
                name="Reward Trend",
                status="OK",
                message=f"Reward is improving ({stats.change_percent:.1f}% change)",
                details={
                    "first_value": stats.first_value,
                    "last_value": stats.last_value,
                    "change_percent": stats.change_percent,
                    "trend": stats.trend
                }
            )
        else:
            return DiagnosticResult(
                name="Reward Trend",
                status="WARNING",
                message=f"Reward trend: {stats.trend} ({stats.change_percent:.1f}% change)",
                details={
                    "first_value": stats.first_value,
                    "last_value": stats.last_value,
                    "change_percent": stats.change_percent,
                    "trend": stats.trend
                }
            )

    def _check_entropy(self) -> Optional[DiagnosticResult]:
        """Check entropy behavior."""
        entropy_stats = self._find_all_metrics(['entropy', 'dist_entropy'])

        if not entropy_stats:
            return DiagnosticResult(
                name="Entropy",
                status="WARNING",
                message="No entropy metric found",
                details={}
            )

        # Use the first entropy metric found
        stats = entropy_stats[0]

        # Entropy should typically decrease (policy becomes more deterministic)
        if stats.trend == "increasing" and stats.change_percent > 20:
            return DiagnosticResult(
                name="Entropy",
                status="CRITICAL",
                message=f"Entropy is INCREASING ({stats.change_percent:.1f}%). Policy is becoming MORE random - this indicates no learning is happening!",
                details={
                    "first_value": stats.first_value,
                    "last_value": stats.last_value,
                    "change_percent": stats.change_percent,
                    "trend": stats.trend,
                    "explanation": "In successful training, entropy should decrease as policy converges to good actions. Increasing entropy means the policy finds no useful actions and drifts to maximum entropy (random)."
                }
            )
        elif stats.trend == "decreasing":
            return DiagnosticResult(
                name="Entropy",
                status="OK",
                message=f"Entropy is decreasing ({stats.change_percent:.1f}%). Policy is converging.",
                details={
                    "first_value": stats.first_value,
                    "last_value": stats.last_value,
                    "change_percent": stats.change_percent,
                    "trend": stats.trend
                }
            )
        else:
            return DiagnosticResult(
                name="Entropy",
                status="WARNING",
                message=f"Entropy trend: {stats.trend} ({stats.change_percent:.1f}% change)",
                details={
                    "first_value": stats.first_value,
                    "last_value": stats.last_value,
                    "change_percent": stats.change_percent,
                    "trend": stats.trend
                }
            )

    def _check_value_loss(self) -> Optional[DiagnosticResult]:
        """Check value/critic loss."""
        stats = self._find_metric(['value_loss', 'critic_loss'])

        if stats is None:
            return None

        if stats.trend == "increasing" and stats.change_percent > 50:
            return DiagnosticResult(
                name="Value Loss",
                status="CRITICAL",
                message=f"Value loss is increasing ({stats.change_percent:.1f}%). Critic is not fitting properly.",
                details={
                    "first_value": stats.first_value,
                    "last_value": stats.last_value,
                    "change_percent": stats.change_percent,
                    "trend": stats.trend
                }
            )
        elif stats.trend == "decreasing":
            return DiagnosticResult(
                name="Value Loss",
                status="OK",
                message=f"Value loss is decreasing ({stats.change_percent:.1f}%). Critic is learning.",
                details={
                    "first_value": stats.first_value,
                    "last_value": stats.last_value,
                    "trend": stats.trend
                }
            )
        else:
            return DiagnosticResult(
                name="Value Loss",
                status="WARNING",
                message=f"Value loss trend: {stats.trend}",
                details={
                    "first_value": stats.first_value,
                    "last_value": stats.last_value,
                    "trend": stats.trend
                }
            )

    def _check_policy_loss(self) -> Optional[DiagnosticResult]:
        """Check policy/actor loss."""
        stats = self._find_metric(['policy_loss', 'actor_loss'])

        if stats is None:
            return None

        # Policy loss magnitude check
        if abs(stats.mean_value) < 1e-5:
            return DiagnosticResult(
                name="Policy Loss",
                status="WARNING",
                message=f"Policy loss is very small ({stats.mean_value:.6f}). Gradients may be vanishing.",
                details={
                    "mean_value": stats.mean_value,
                    "std_value": stats.std_value,
                    "trend": stats.trend
                }
            )

        return DiagnosticResult(
            name="Policy Loss",
            status="OK",
            message=f"Policy loss: mean={stats.mean_value:.6f}, std={stats.std_value:.6f}",
            details={
                "mean_value": stats.mean_value,
                "std_value": stats.std_value,
                "trend": stats.trend
            }
        )

    def _check_gradients(self) -> Optional[DiagnosticResult]:
        """Check gradient norms."""
        grad_stats = self._find_all_metrics(['grad_norm'])

        if not grad_stats:
            return None

        issues = []
        for stats in grad_stats:
            if stats.max_value > 100:
                issues.append(f"{stats.name}: max={stats.max_value:.2f}")
            if stats.mean_value < 0.001:
                issues.append(f"{stats.name}: vanishing (mean={stats.mean_value:.6f})")

        if issues:
            return DiagnosticResult(
                name="Gradient Norms",
                status="WARNING",
                message=f"Gradient issues detected: {'; '.join(issues)}",
                details={"issues": issues}
            )

        return DiagnosticResult(
            name="Gradient Norms",
            status="OK",
            message="Gradient norms are within normal range",
            details={}
        )

    def _check_reward_components(self) -> Optional[DiagnosticResult]:
        """Check individual reward components."""
        reward_stats = self._find_all_metrics(['rewards/', 'reward/'])

        if not reward_stats:
            return None

        zero_rewards = []
        negative_trends = []

        for stats in reward_stats:
            # Check for near-zero rewards
            if abs(stats.mean_value) < 1e-5 and 'punishment' not in stats.name.lower():
                zero_rewards.append(stats.name.split('/')[-1])

            # Check for negative trends in positive rewards
            if stats.trend == "decreasing" and 'punishment' not in stats.name.lower() and 'penalty' not in stats.name.lower():
                if stats.change_percent < -20:
                    negative_trends.append(f"{stats.name.split('/')[-1]}: {stats.change_percent:.1f}%")

        issues = []
        if zero_rewards:
            issues.append(f"Near-zero rewards: {', '.join(zero_rewards)}")
        if negative_trends:
            issues.append(f"Decreasing rewards: {', '.join(negative_trends)}")

        if issues:
            return DiagnosticResult(
                name="Reward Components",
                status="WARNING",
                message="; ".join(issues),
                details={
                    "zero_rewards": zero_rewards,
                    "negative_trends": negative_trends,
                    "all_components": [s.name for s in reward_stats]
                }
            )

        return DiagnosticResult(
            name="Reward Components",
            status="OK",
            message=f"Analyzed {len(reward_stats)} reward components",
            details={"all_components": [s.name for s in reward_stats]}
        )

    def generate_report(self) -> AnalysisReport:
        """Generate complete analysis report."""
        if not self.metrics_data:
            self.extract_all_metrics()
        if not self.metric_stats:
            self.compute_all_stats()

        diagnostics = self.run_diagnostics()

        # Generate summary
        critical_count = sum(1 for d in diagnostics if d.status == "CRITICAL")
        warning_count = sum(1 for d in diagnostics if d.status == "WARNING")
        ok_count = sum(1 for d in diagnostics if d.status == "OK")

        if critical_count > 0:
            summary = f"CRITICAL: {critical_count} critical issues found. Training likely failed."
        elif warning_count > 2:
            summary = f"WARNING: {warning_count} warnings found. Training may have issues."
        else:
            summary = f"OK: Training appears healthy ({ok_count} checks passed)."

        # Get total steps
        total_steps = 0
        for stats in self.metric_stats.values():
            if stats.last_step > total_steps:
                total_steps = stats.last_step

        return AnalysisReport(
            log_dir=self.log_dir,
            total_steps=total_steps,
            available_metrics=list(self.metrics_data.keys()),
            metric_stats=self.metric_stats,
            diagnostics=diagnostics,
            summary=summary
        )

    def print_report(self, report: AnalysisReport, verbose: bool = False):
        """Print formatted report to console."""
        print("=" * 80)
        print("TENSORBOARD LOG ANALYSIS REPORT")
        print("=" * 80)
        print(f"Log Directory: {report.log_dir}")
        print(f"Total Steps: {report.total_steps:,}")
        print(f"Metrics Found: {len(report.available_metrics)}")
        print()

        # Summary
        print("-" * 80)
        print("SUMMARY")
        print("-" * 80)
        print(report.summary)
        print()

        # Diagnostics
        print("-" * 80)
        print("DIAGNOSTICS")
        print("-" * 80)
        for diag in report.diagnostics:
            status_color = {
                "CRITICAL": "🔴",
                "WARNING": "🟡",
                "OK": "🟢"
            }.get(diag.status, "⚪")
            print(f"{status_color} [{diag.status}] {diag.name}")
            print(f"   {diag.message}")
            if verbose and diag.details:
                for key, value in diag.details.items():
                    if key != "explanation":
                        print(f"      {key}: {value}")
                if "explanation" in diag.details:
                    print(f"   💡 {diag.details['explanation']}")
            print()

        # Key Metrics Summary
        print("-" * 80)
        print("KEY METRICS SUMMARY")
        print("-" * 80)

        key_patterns = ['success', 'reward', 'entropy', 'loss']
        printed_metrics = set()

        for pattern in key_patterns:
            for name, stats in report.metric_stats.items():
                if pattern in name.lower() and name not in printed_metrics:
                    printed_metrics.add(name)
                    short_name = name.split('/')[-1] if '/' in name else name
                    print(f"{short_name}:")
                    print(f"   Start: {stats.first_value:.6f} → End: {stats.last_value:.6f}")
                    print(f"   Change: {stats.change_percent:+.1f}% ({stats.trend})")
                    print()

        if verbose:
            print("-" * 80)
            print("ALL METRICS")
            print("-" * 80)
            for name in sorted(report.available_metrics):
                stats = report.metric_stats[name]
                print(f"{name}: {stats.first_value:.6f} → {stats.last_value:.6f} ({stats.trend})")

    def export_json(self, report: AnalysisReport, filepath: str):
        """Export report to JSON file."""
        # Convert dataclasses to dicts
        output = {
            "log_dir": report.log_dir,
            "total_steps": report.total_steps,
            "available_metrics": report.available_metrics,
            "summary": report.summary,
            "metric_stats": {
                name: asdict(stats) for name, stats in report.metric_stats.items()
            },
            "diagnostics": [asdict(d) for d in report.diagnostics]
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"Report exported to: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze TensorBoard logs for training diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s results/mapush/.../logs
  %(prog)s results/mapush/.../logs --verbose
  %(prog)s results/mapush/.../logs --json report.json
  %(prog)s results/mapush/.../logs --output report.txt
        """
    )
    parser.add_argument("log_dir", help="Path to TensorBoard log directory")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show detailed output")
    parser.add_argument("-j", "--json", metavar="FILE",
                        help="Export report to JSON file")
    parser.add_argument("-o", "--output", metavar="FILE",
                        help="Save text report to file")

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = TensorBoardAnalyzer(args.log_dir)

    # Load data
    print(f"Loading TensorBoard logs from: {args.log_dir}")
    if not analyzer.load():
        sys.exit(1)

    # Generate report
    print("Analyzing metrics...")
    report = analyzer.generate_report()

    # Output
    if args.output:
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            analyzer.print_report(report, verbose=args.verbose)

        with open(args.output, 'w') as out:
            out.write(f.getvalue())
        print(f"Report saved to: {args.output}")
    else:
        analyzer.print_report(report, verbose=args.verbose)

    if args.json:
        analyzer.export_json(report, args.json)

    # Exit code based on status
    critical_count = sum(1 for d in report.diagnostics if d.status == "CRITICAL")
    if critical_count > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
