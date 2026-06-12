import torch
import statistics
import os
from collections import deque
from RenforceRL import configclass
from typing import TYPE_CHECKING
from .logger_base import LoggerBase, LoggerBaseCfg

if TYPE_CHECKING:
    from RenforceRL.runners.on_policy.on_policy_runner import OnPolicyRunner


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"
    GRAY = "\033[90m"


class TqdmStyleLogger(LoggerBase):
    """A logger that displays information in a box with progress bar, similar to tqdm.
    
    Features:
    - Updates content in-place within a box frame
    - Shows progress bar for iterations
    - Optional performance curve for iteration time
    - Colorful output for better readability
    """
    
    def __init__(self, cfg, log_dir):
        super().__init__(cfg, log_dir)
        self.iteration_times = deque(maxlen=cfg.performance_history_size) if cfg.show_performance_curve else None
        self.use_colors = cfg.use_colors and os.getenv("TERM") is not None
        
    def log(
        self,
        runner: "OnPolicyRunner",
        locs: dict,
        width: int = None,
        pad: int = None,
        ep_string: str = ""
    ):
        width = self.cfg.width if width is None else width
        pad = self.cfg.pad if pad is None else pad
        
        # Update global counters
        self._update_time_counters(runner, locs)
        
        # Collect iteration time for performance curve
        if self.iteration_times is not None:
            iteration_time = locs["collection_time"] + locs["learn_time"]
            self.iteration_times.append(iteration_time)
        
        # Determine whether to show performance curve (affects where we close the box)
        show_curve = (
            self.cfg.show_performance_curve
            and self.iteration_times
            and len(self.iteration_times) > 1
        )

        # Build content
        content_lines = []
        content_lines.extend(self._build_header(locs, width))
        content_lines.extend(self._build_progress_bar(locs, width))
        content_lines.extend(self._build_episode_infos(runner, locs, pad, width))
        content_lines.extend(self._build_alg_update_infos(runner, locs, pad, width))
        content_lines.extend(self._build_sample_infos(runner, locs, pad, width))
        content_lines.extend(self._build_statistics(runner, locs, pad, width))
        content_lines.extend(self._build_footer(runner, locs, pad, width, add_bottom_border=not show_curve))
        
        # Optional performance curve (adds its own bottom border when shown)
        if show_curve:
            content_lines.extend(self._build_performance_curve(width))
        
        # Clear screen and write new content
        self._update_output(content_lines)
    
    def _update_output(self, content_lines):
        """Clear screen and write new content."""
        # Clear screen and move cursor to top-left
        print("\033[2J\033[H", end="", flush=True)
        
        # Write all content
        output = "\n".join(content_lines)
        print(output, flush=True)
    
    def _colorize(self, text, color):
        """Add color to text if colors are enabled."""
        return f"{color}{text}{Colors.RESET}" if self.use_colors else text
    
    def _strip_ansi(self, text):
        """Remove ANSI escape codes from text."""
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)
    
    def _get_display_width(self, text):
        """Get the display width of text, ignoring ANSI codes."""
        return len(self._strip_ansi(text))
    
    def _ensure_line_width(self, line, target_width):
        """Ensure a line matches the target width exactly."""
        current_width = self._get_display_width(line)
        if current_width == target_width:
            return line
        
        # Find the position of the closing border
        if line.endswith(" │"):
            # Insert padding before the closing border
            padding_needed = target_width - current_width
            if padding_needed > 0:
                return line.replace(" │", " " * padding_needed + " │")
            elif padding_needed < 0:
                # Remove excess characters before the border
                excess = -padding_needed
                border_pos = line.rfind(" │")
                if border_pos > excess:
                    return line[:border_pos - excess] + " │"
        
        return line
    
    def _build_header(self, locs, width):
        """Build header with iteration info."""
        title = f" Learning iteration {locs['it']}/{locs['tot_iter']} "
        border = "─" * (width - 2)
        return [
            self._colorize("┌" + border + "┐", Colors.CYAN),
            self._colorize("│", Colors.CYAN) + title.center(width - 2) + self._colorize("│", Colors.CYAN),
            self._colorize("├" + border + "┤", Colors.CYAN)
        ]
    
    def _build_progress_bar(self, locs, width):
        """Build progress bar showing iteration progress."""
        current = locs['it'] + 1
        total = locs['tot_iter']
        progress = current / total if total > 0 else 0
        
        # Format: "│ [<bar>] XX.X% │"
        percentage_text = f"{progress * 100:5.1f}%"
        percentage_colored = self._colorize(percentage_text, Colors.BOLD)
        percentage_width = self._get_display_width(percentage_colored)
        
        # Total fixed chars besides the bar:
        # 2 borders ("│" ... "│") + 2 brackets "[]" + 2 spaces + percentage
        # So line_width = 2 + 2 + 2 + percentage_width + bar_width
        # We want line_width == width → bar_width = width - (6 + percentage_width)
        bar_width = max(0, width - (6 + percentage_width))
        
        filled = int(bar_width * progress)
        filled_bar = self._colorize("█" * filled, Colors.GREEN)
        empty_bar = "░" * (bar_width - filled)
        
        bar = f"│ [{filled_bar}{empty_bar}] {percentage_colored} │"
        return [self._ensure_line_width(bar, width)]
    
    def _build_info_lines(self, info_string, width):
        """Build info lines from string, wrapping in box borders."""
        lines = []
        if info_string:
            for line in info_string.strip().split("\n"):
                if line.strip():
                    # Calculate actual display width and pad accordingly
                    display_width = self._get_display_width(line)
                    padding = width - 4 - display_width  # 4 = 2 borders + 2 spaces
                    if padding > 0:
                        padded_line = line + " " * padding
                    else:
                        padded_line = line
                    formatted_line = f"│ {padded_line} │"
                    
                    # Ensure exact width match
                    formatted_line = self._ensure_line_width(formatted_line, width)
                    lines.append(formatted_line)
        return lines
    
    def _build_episode_infos(self, runner, locs, pad, width):
        """Build episode information lines."""

        ep_string = self._log_episode_infos(runner, locs, pad)
        if not self.cfg.is_log_ep_info or not locs["ep_infos"]:
            return []
        return self._build_info_lines(ep_string, width)
    
    def _build_alg_update_infos(self, runner, locs, pad, width):
        """Build algorithm update information lines."""
        update_string = self._log_alg_update_infos(runner, locs, pad)
        if not self.cfg.is_log_update:
            return []
        return self._build_info_lines(update_string, width)
    
    def _build_sample_infos(self, runner, locs, pad, width):
        """Build sample information lines."""
        sample_string = self._log_sample_infos(runner, locs, pad)
        if not self.cfg.is_log_sample:
            return []
        return self._build_info_lines(sample_string, width)
    
    def _build_statistics(self, runner, locs, pad, width):
        """Build statistics lines."""
        if len(runner.rewbuffer) == 0:
            return []
        
        stats_string = self._log_statistics_string(runner, locs, pad)
        return self._build_info_lines(stats_string, width)
    
    def _build_footer(self, runner, locs, pad, width, add_bottom_border: bool = True):
        """Build footer with timing information."""
        sample_infos = locs["sample_infos"]
        fps = int(runner.cfg.num_steps_per_env * runner.env.num_envs / 
                 (sample_infos["collection_time"] + locs["learn_time"]))
        if self.writer:
            self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
            self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
            self.writer.add_scalar("Perf/collection_time", sample_infos["collection_time"], locs["it"])
        
        iteration_time = sample_infos["collection_time"] + locs["learn_time"]
        eta = (self.tot_time / (locs["it"] + 1) * 
               (locs["num_learning_iterations"] - locs["it"]))
        
        # Build footer lines with proper alignment
        lines = []
        label_width = pad
        
        def _format_line(label_text, value_text, label_color=Colors.BLUE, value_color=None):
            """Format a footer line with proper alignment."""
            label = self._colorize(label_text, label_color)
            value = self._colorize(value_text, value_color) if value_color else value_text
            
            # Calculate padding needed for label
            label_display_width = self._get_display_width(label)
            label_padding = label_width - label_display_width
            
            # Build the line
            line = f"│ {label:>{label_display_width + label_padding}} {value} │"
            
            # Ensure exact width match
            return self._ensure_line_width(line, width)
        
        lines.append(_format_line('Computation:', f'{fps:.0f} steps/s', Colors.BLUE, Colors.GREEN))
        lines.append(_format_line('Total timesteps:', str(self.tot_timesteps), Colors.BLUE))
        lines.append(_format_line('Iteration time:', f'{iteration_time:.2f}s', Colors.BLUE, Colors.YELLOW))
        lines.append(_format_line('Total time:', f'{self.tot_time:.2f}s', Colors.BLUE))
        lines.append(_format_line('ETA:', f'{eta:.1f}s', Colors.BLUE, Colors.MAGENTA))
        
        if add_bottom_border:
            border = "─" * (width - 2)
            lines.append(self._colorize("└" + border + "┘", Colors.CYAN))
        return lines
    
    def _build_performance_curve(self, width):
        """Build ASCII performance curve for iteration time."""
        if not self.iteration_times or len(self.iteration_times) < 2:
            return []
        
        border = "─" * (width - 2)
        title = self._colorize('Performance Curve (Iteration Time):', Colors.BOLD)
        title_display_width = self._get_display_width(title)
        title_padding = width - 4 - title_display_width
        title_line = f"│ {title}{' ' * title_padding} │"
        lines = [
            self._colorize("├" + border + "┤", Colors.CYAN),
            self._ensure_line_width(title_line, width)
        ]
        
        # Create ASCII chart
        chart_width = width - 6  # Account for borders and padding: │ [content] │ = 6 chars
        chart_height = 8
        
        if chart_width < 10 or chart_height < 3:
            return lines
        
        # Normalize iteration times for display
        times = list(self.iteration_times)
        min_time = min(times)
        max_time = max(times)
        time_range = max_time - min_time if max_time > min_time else 1
        
        # Create chart grid
        chart = [[' ' for _ in range(chart_width)] for _ in range(chart_height)]
        
        # Plot points
        for i, time_val in enumerate(times):
            x = int((i / (len(times) - 1)) * (chart_width - 1)) if len(times) > 1 else 0
            y = int(((time_val - min_time) / time_range) * (chart_height - 1))
            y = min(max(y, 0), chart_height - 1)
            chart[chart_height - 1 - y][x] = self._colorize('●', Colors.YELLOW) if self.use_colors else '●'
        
        # Draw chart
        for row in chart:
            chart_line = "│ " + "".join(row) + " │"
            lines.append(self._ensure_line_width(chart_line, width))
        
        # Add min/max labels
        min_label = self._colorize(f'{min_time:.2f}s', Colors.GREEN)
        max_label = self._colorize(f'{max_time:.2f}s', Colors.RED)
        label = f"Min: {min_label}, Max: {max_label}"
        label_display_width = self._get_display_width(label)
        label_padding = width - 4 - label_display_width
        label_line = f"│ {label}{' ' * label_padding} │"
        lines.append(self._ensure_line_width(label_line, width))

        # Close the performance curve box (and thus the whole logger box)
        border = "─" * (width - 2)
        lines.append(self._colorize("└" + border + "┘", Colors.CYAN))
        return lines


@configclass
class TqdmStyleLoggerCfg(LoggerBaseCfg):
    class_type: type[TqdmStyleLogger] = TqdmStyleLogger
    
    show_performance_curve: bool = False
    """Whether to show performance curve for iteration time."""
    
    performance_history_size: int = 50
    """Number of recent iteration times to keep for performance curve."""
    
    use_colors: bool = True
    """Whether to use colors in the output. Automatically disabled if TERM is not set."""