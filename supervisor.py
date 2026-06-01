#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLMClient 外部守护脚本

零侵入地包装任意批量 Python 脚本，通过解析 stdout 检测 API 集中失败，
执行熔断-冷却-重启循环。

用法:
    python -m LLMClient_v2.supervisor [options] -- python my_batch_script.py --arg1 val1
"""

from __future__ import annotations

import argparse
import asyncio
import enum
import logging
import os
import re
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

try:
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
    from rich.panel import Panel as RichPanel
    from rich.console import Console

    _rich_available = True
except ImportError:
    _rich_available = False

logger = logging.getLogger("LLMClient_v2.supervisor")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


class LineClass(enum.Enum):
    SUCCESS_CALL = "success_call"
    FAILURE_CALL_API = "failure_call_api"
    FAILURE_CALL_LOGIC = "failure_call_logic"
    BATCH_SUCCESS = "batch_success"
    BATCH_FAILURE = "batch_failure"
    WARNING_API = "warning_api"
    PROGRESS = "progress"
    OTHER = "other"


@dataclass
class SupervisorConfig:
    command: List[str]
    failure_window_seconds: int = 120
    failure_threshold_ratio: float = 0.8
    consecutive_failure_limit: int = 15
    cooldown_seconds: int = 300
    max_restarts: int = 10
    enable_panel: bool = True


@dataclass
class HealthSnapshot:
    state: str
    total_calls: int
    success_count: int
    failure_count: int
    logic_error_count: int
    consecutive_failures: int
    window_size: int
    current_failure_ratio: float
    restart_count: int
    max_restarts: int
    uptime_seconds: float
    last_events: List[str] = field(default_factory=list)
    batch_progress: Optional[str] = None
    cooldown_remaining: Optional[float] = None


# ---------------------------------------------------------------------------
# FailureDetector
# ---------------------------------------------------------------------------

# 最终结果行（值错误/未知错误必须在通用调用失败之前匹配）
_PATTERNS: List[Tuple[re.Pattern, LineClass]] = [
    (re.compile(r"\[call_\w+\]\s*✅\s*Vision调用成功"), LineClass.SUCCESS_CALL),
    (re.compile(r"\[call_\w+\]\s*✅\s*调用成功"), LineClass.SUCCESS_CALL),
    (re.compile(r"\[\d+/\d+\]\s*✅\s*成功"), LineClass.BATCH_SUCCESS),
    (re.compile(r"\[\d+/\d+\]\s*❌\s*失败"), LineClass.BATCH_FAILURE),
    (re.compile(r"\[call_\w+\]\s*❌\s*值错误"), LineClass.FAILURE_CALL_LOGIC),
    (re.compile(r"\[call_\w+\]\s*❌\s*未知错误"), LineClass.FAILURE_CALL_LOGIC),
    (re.compile(r"\[call_\w+\]\s*❌\s*(?:Vision)?调用失败"), LineClass.FAILURE_CALL_API),
    (re.compile(r"\[call_\w+\]\s*⚠️\s*API连接失败"), LineClass.WARNING_API),
    (re.compile(r"\[call_\w+\]\s*⚠️\s*API错误"), LineClass.WARNING_API),
    (re.compile(r"\[call_\w+\]\s*⚠️\s*请求超时"), LineClass.WARNING_API),
    (re.compile(r"\[call_\w+\]\s*⚠️\s*收到空响应"), LineClass.WARNING_API),
    (re.compile(r"🧮\s*待处理任务"), LineClass.PROGRESS),
    (re.compile(r"✅\s*批次\s*\d+/\d+\s*完成"), LineClass.PROGRESS),
]

_BATCH_PROGRESS_RE = re.compile(r"(批次\s*\d+/\d+)")
_MIN_WINDOW_SIZE = 5


class FailureDetector:
    def __init__(self, config: SupervisorConfig):
        self._config = config
        self._window: deque[Tuple[float, bool]] = deque()
        self._consecutive_failures: int = 0
        self._total_successes: int = 0
        self._total_failures: int = 0
        self._total_logic_errors: int = 0
        self._last_event_time: Optional[float] = None
        self._last_events: deque[str] = deque(maxlen=15)
        self._batch_progress: Optional[str] = None

    def classify_line(self, line: str) -> LineClass:
        for pattern, cls in _PATTERNS:
            if pattern.search(line):
                return cls
        return LineClass.OTHER

    def feed(self, line: str, timestamp: Optional[float] = None) -> LineClass:
        now = timestamp if timestamp is not None else time.monotonic()
        cls = self.classify_line(line)

        if cls == LineClass.OTHER:
            return cls

        ts_str = datetime.now().strftime("%H:%M:%S")
        short = line.strip()[:80]
        self._last_events.append(f"[{ts_str}] {short}")
        self._last_event_time = now

        if cls in (LineClass.SUCCESS_CALL, LineClass.BATCH_SUCCESS):
            self._window.append((now, False))
            self._consecutive_failures = 0
            self._total_successes += 1
        elif cls in (LineClass.FAILURE_CALL_API, LineClass.BATCH_FAILURE):
            self._window.append((now, True))
            self._consecutive_failures += 1
            self._total_failures += 1
        elif cls == LineClass.FAILURE_CALL_LOGIC:
            self._total_logic_errors += 1
        elif cls == LineClass.PROGRESS:
            m = _BATCH_PROGRESS_RE.search(line)
            if m:
                self._batch_progress = m.group(1)

        self._prune_window(now)
        return cls

    def _prune_window(self, now: float) -> None:
        cutoff = now - self._config.failure_window_seconds
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

    def should_trip(self) -> bool:
        if self._consecutive_failures >= self._config.consecutive_failure_limit:
            return True
        if len(self._window) >= _MIN_WINDOW_SIZE:
            failures = sum(1 for _, is_fail in self._window if is_fail)
            ratio = failures / len(self._window)
            if ratio >= self._config.failure_threshold_ratio:
                return True
        return False

    @property
    def failure_ratio(self) -> float:
        if not self._window:
            return 0.0
        failures = sum(1 for _, is_fail in self._window if is_fail)
        return failures / len(self._window)

    @property
    def window_failures(self) -> int:
        return sum(1 for _, is_fail in self._window if is_fail)

    def reset(self) -> None:
        self._window.clear()
        self._consecutive_failures = 0

    def snapshot(
        self,
        state: str,
        restart_count: int,
        max_restarts: int,
        uptime: float,
        cooldown_remaining: Optional[float] = None,
    ) -> HealthSnapshot:
        return HealthSnapshot(
            state=state,
            total_calls=self._total_successes + self._total_failures,
            success_count=self._total_successes,
            failure_count=self._total_failures,
            logic_error_count=self._total_logic_errors,
            consecutive_failures=self._consecutive_failures,
            window_size=len(self._window),
            current_failure_ratio=self.failure_ratio,
            restart_count=restart_count,
            max_restarts=max_restarts,
            uptime_seconds=uptime,
            last_events=list(self._last_events),
            batch_progress=self._batch_progress,
            cooldown_remaining=cooldown_remaining,
        )


# ---------------------------------------------------------------------------
# HealthPanel
# ---------------------------------------------------------------------------


def _fmt_duration(seconds: float) -> str:
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


class HealthPanel:
    def __init__(self, enable_rich: bool = True):
        self._use_rich = enable_rich and _rich_available
        self._live: Optional[Live] = None  # type: ignore[name-defined]
        self._console: Optional[Console] = None  # type: ignore[name-defined]
        self._last_fallback_time: float = 0

    def start(self) -> None:
        if self._use_rich:
            self._console = Console()
            self._live = Live(console=self._console, refresh_per_second=2)
            self._live.start()

    def stop(self) -> None:
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

    def update(self, snap: HealthSnapshot) -> None:
        if self._use_rich and self._live is not None:
            self._live.update(self._build_rich_display(snap))
        else:
            now = time.monotonic()
            if now - self._last_fallback_time >= 5:
                self._print_fallback(snap)
                self._last_fallback_time = now

    def log_event(self, message: str) -> None:
        if self._use_rich and self._live is not None:
            self._live.console.print(message)
        else:
            clean = re.sub(r"\[/?[a-zA-Z_ ]*\]", "", message)
            print(clean, flush=True)

    def _build_rich_display(self, snap: HealthSnapshot) -> RichPanel:  # type: ignore[name-defined]
        state_icons = {
            "Running": "[bold green]🟢 Running[/]",
            "Circuit-Break": "[bold red]🔴 Circuit-Break[/]",
            "Cooldown": "[bold yellow]🟡 Cooldown[/]",
            "Stopped": "[dim]⚪ Stopped[/]",
        }
        state_text = state_icons.get(snap.state, snap.state)

        if snap.state == "Cooldown" and snap.cooldown_remaining is not None:
            state_text += f" ({_fmt_duration(snap.cooldown_remaining)})"

        success_pct = (
            f"{snap.success_count / snap.total_calls * 100:.1f}%"
            if snap.total_calls > 0
            else "-"
        )
        window_fail = sum(
            1
            for _ in range(0)  # placeholder; use ratio directly
        )
        ratio_str = f"{snap.current_failure_ratio * 100:.1f}%"

        table = Table.grid(padding=(0, 2))
        table.add_column(width=28)
        table.add_column(width=28)

        table.add_row(
            f"State:     {state_text}",
            f"Uptime:    {_fmt_duration(snap.uptime_seconds)}",
        )
        table.add_row(
            f"Restarts:  {snap.restart_count}/{snap.max_restarts}",
            "",
        )
        table.add_row("", "")
        table.add_row(
            f"Calls:     {snap.total_calls} total",
            f"Success:   {snap.success_count} ({success_pct})",
        )
        table.add_row(
            f"API Fail:  {snap.failure_count}",
            f"Logic Err: {snap.logic_error_count}",
        )
        table.add_row(
            f"Window:    {snap.window_size} ({ratio_str})",
            f"Consec:    {snap.consecutive_failures}",
        )

        if snap.batch_progress:
            table.add_row(f"Batch:     {snap.batch_progress}", "")

        if snap.last_events:
            table.add_row("", "")
            table.add_row("[bold]Recent:[/]", "")
            for ev in snap.last_events[-6:]:
                table.add_row(f"  {ev}", "")

        return RichPanel(table, title="LLMClient Supervisor", border_style="cyan")

    def _print_fallback(self, snap: HealthSnapshot) -> None:
        cd = ""
        if snap.state == "Cooldown" and snap.cooldown_remaining is not None:
            cd = f" cd={_fmt_duration(snap.cooldown_remaining)}"
        ratio_str = f"{snap.current_failure_ratio * 100:.1f}%"
        line = (
            f"[SUPERVISOR] {snap.state}{cd} | "
            f"calls={snap.total_calls} ok={snap.success_count} "
            f"fail={snap.failure_count} ratio={ratio_str} "
            f"consec={snap.consecutive_failures} "
            f"restarts={snap.restart_count}/{snap.max_restarts}"
        )
        if snap.batch_progress:
            line += f" | {snap.batch_progress}"
        print(f"\r{line}  ", end="", flush=True)


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


class Supervisor:
    def __init__(self, config: SupervisorConfig):
        self._config = config
        self._detector = FailureDetector(config)
        self._panel = HealthPanel(enable_rich=config.enable_panel)
        self._restart_count = 0
        self._start_time: float = 0
        self._state: str = "Stopped"
        self._child: Optional[asyncio.subprocess.Process] = None
        self._cooldown_remaining: Optional[float] = None
        self._shutdown_requested = False
        self._circuit_break_event = asyncio.Event()

    async def run(self) -> int:
        self._start_time = time.monotonic()
        self._install_signal_handlers()
        self._panel.start()

        cmd_str = " ".join(self._config.command)
        self._panel.log_event(f"[bold]Supervisor 启动[/] | 守护命令: {cmd_str}")
        self._panel.log_event(
            f"  冷却={self._config.cooldown_seconds}s  "
            f"窗口={self._config.failure_window_seconds}s  "
            f"阈值={self._config.failure_threshold_ratio}  "
            f"连续上限={self._config.consecutive_failure_limit}  "
            f"最大重启={self._config.max_restarts}"
        )

        try:
            while self._restart_count <= self._config.max_restarts:
                if self._shutdown_requested:
                    self._state = "Stopped"
                    self._panel.log_event("用户中断，退出")
                    return 130

                self._state = "Running"
                self._detector.reset()
                self._circuit_break_event.clear()
                self._cooldown_remaining = None

                action = "首次启动" if self._restart_count == 0 else f"第 {self._restart_count} 次重启"
                self._panel.log_event(f"🚀 {action}: {cmd_str}")

                proc = await self._start_child()
                self._child = proc

                stdout_task = asyncio.create_task(self._read_stdout(proc))
                stderr_task = asyncio.create_task(self._read_stderr(proc))
                panel_task = asyncio.create_task(self._panel_refresh_loop())
                circuit_task = asyncio.create_task(self._circuit_break_event.wait())

                io_done = asyncio.gather(stdout_task, stderr_task)

                done, _ = await asyncio.wait(
                    [io_done, circuit_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                panel_task.cancel()

                if self._circuit_break_event.is_set():
                    trip_reason = self._trip_reason()
                    self._panel.log_event(f"🔴 熔断触发: {trip_reason}")
                    logger.warning("Circuit-break triggered: %s", trip_reason)
                    await self._kill_child()

                    stdout_task.cancel()
                    stderr_task.cancel()
                    try:
                        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
                    except Exception:
                        pass

                    self._restart_count += 1
                    if self._restart_count > self._config.max_restarts:
                        self._state = "Stopped"
                        self._panel.log_event("❌ 达到最大重启次数，停止守护")
                        return 1

                    self._state = "Cooldown"
                    self._panel.log_event(
                        f"⏳ 冷却 {self._config.cooldown_seconds} 秒..."
                    )
                    await self._cooldown()
                    continue

                await io_done
                await proc.wait()
                exit_code = proc.returncode or 0

                circuit_task.cancel()

                if exit_code == 0:
                    self._state = "Stopped"
                    self._panel.log_event("✅ 子进程正常退出 (exit 0)")
                    return 0

                self._panel.log_event(f"⚠️ 子进程异常退出 (exit {exit_code})")
                self._restart_count += 1
                if self._restart_count > self._config.max_restarts:
                    self._state = "Stopped"
                    self._panel.log_event("❌ 达到最大重启次数，停止守护")
                    return 1

                self._state = "Cooldown"
                self._panel.log_event(
                    f"⏳ 冷却 {self._config.cooldown_seconds} 秒后重启..."
                )
                await self._cooldown()

            self._state = "Stopped"
            return 1

        except asyncio.CancelledError:
            await self._kill_child()
            return 130
        finally:
            self._panel.stop()

    async def _start_child(self) -> asyncio.subprocess.Process:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if sys.platform == "win32":
            env["PYTHONUTF8"] = "1"

        proc = await asyncio.create_subprocess_exec(
            *self._config.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        logger.info("Child started PID=%s cmd=%s", proc.pid, self._config.command)
        return proc

    async def _read_stdout(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        while True:
            try:
                raw = await proc.stdout.readline()
            except (asyncio.CancelledError, ConnectionResetError):
                return
            if not raw:
                return
            line = raw.decode("utf-8", errors="replace").rstrip("\n\r")

            if not self._panel._use_rich:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()

            cls = self._detector.feed(line, time.monotonic())

            if self._detector.should_trip() and not self._circuit_break_event.is_set():
                self._state = "Circuit-Break"
                self._circuit_break_event.set()
                return

    async def _read_stderr(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stderr is not None
        while True:
            try:
                raw = await proc.stderr.readline()
            except (asyncio.CancelledError, ConnectionResetError):
                return
            if not raw:
                return
            line = raw.decode("utf-8", errors="replace").rstrip("\n\r")
            if line:
                sys.stderr.write(line + "\n")
                sys.stderr.flush()

    async def _panel_refresh_loop(self) -> None:
        try:
            while True:
                uptime = time.monotonic() - self._start_time
                snap = self._detector.snapshot(
                    state=self._state,
                    restart_count=self._restart_count,
                    max_restarts=self._config.max_restarts,
                    uptime=uptime,
                    cooldown_remaining=self._cooldown_remaining,
                )
                self._panel.update(snap)
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    async def _kill_child(self) -> None:
        if self._child is None or self._child.returncode is not None:
            return

        logger.info("Terminating child PID=%s", self._child.pid)

        if sys.platform == "win32":
            self._child.terminate()
            try:
                await asyncio.wait_for(self._child.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._child.kill()
                await self._child.wait()
        else:
            self._child.send_signal(signal.SIGINT)
            try:
                await asyncio.wait_for(self._child.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._child.send_signal(signal.SIGTERM)
                try:
                    await asyncio.wait_for(self._child.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    self._child.kill()
                    await self._child.wait()

    async def _cooldown(self) -> None:
        remaining = float(self._config.cooldown_seconds)
        panel_task = asyncio.create_task(self._panel_refresh_loop())

        try:
            while remaining > 0 and not self._shutdown_requested:
                self._cooldown_remaining = remaining
                await asyncio.sleep(1)
                remaining -= 1
        finally:
            self._cooldown_remaining = None
            panel_task.cancel()
            try:
                await panel_task
            except asyncio.CancelledError:
                pass

    def _trip_reason(self) -> str:
        parts = []
        if self._detector._consecutive_failures >= self._config.consecutive_failure_limit:
            parts.append(
                f"连续失败 {self._detector._consecutive_failures} 次 "
                f"(上限 {self._config.consecutive_failure_limit})"
            )
        ratio = self._detector.failure_ratio
        if len(self._detector._window) >= _MIN_WINDOW_SIZE and ratio >= self._config.failure_threshold_ratio:
            parts.append(
                f"窗口失败率 {ratio * 100:.1f}% "
                f"(阈值 {self._config.failure_threshold_ratio * 100:.0f}%, "
                f"窗口 {len(self._detector._window)} 条)"
            )
        return "; ".join(parts) if parts else "unknown"

    def _install_signal_handlers(self) -> None:
        if sys.platform == "win32":
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGBREAK, self._handle_signal)
        else:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda: self._handle_shutdown())

    def _handle_signal(self, signum, frame):
        self._handle_shutdown()

    def _handle_shutdown(self):
        self._shutdown_requested = True
        self._circuit_break_event.set()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_supervisor_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m LLMClient_v2.supervisor",
        description="LLMClient 外部守护脚本 — 熔断 / 冷却 / 重启",
        usage="%(prog)s [options] -- COMMAND [ARGS...]",
    )
    parser.add_argument(
        "--cooldown", type=int, default=300,
        help="熔断后冷却秒数 (默认: 300)",
    )
    parser.add_argument(
        "--max-restarts", type=int, default=10,
        help="最大重启次数 (默认: 10)",
    )
    parser.add_argument(
        "--failure-window", type=int, default=120,
        help="滑动窗口秒数 (默认: 120)",
    )
    parser.add_argument(
        "--failure-threshold", type=float, default=0.8,
        help="失败率阈值 0-1 (默认: 0.8)",
    )
    parser.add_argument(
        "--consecutive-limit", type=int, default=15,
        help="连续失败上限 (默认: 15)",
    )
    parser.add_argument(
        "--no-panel", action="store_true",
        help="禁用 rich 面板，使用纯文本输出",
    )
    parser.add_argument(
        "command_args", nargs=argparse.REMAINDER,
        help="被守护的命令 (在 '--' 之后)",
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = build_supervisor_parser()
    args = parser.parse_args()

    cmd = args.command_args
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("未指定命令。用法: supervisor [options] -- COMMAND [ARGS...]")

    config = SupervisorConfig(
        command=cmd,
        failure_window_seconds=args.failure_window,
        failure_threshold_ratio=args.failure_threshold,
        consecutive_failure_limit=args.consecutive_limit,
        cooldown_seconds=args.cooldown,
        max_restarts=args.max_restarts,
        enable_panel=not args.no_panel,
    )

    supervisor = Supervisor(config)

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    return asyncio.run(supervisor.run())


if __name__ == "__main__":
    raise SystemExit(main())
