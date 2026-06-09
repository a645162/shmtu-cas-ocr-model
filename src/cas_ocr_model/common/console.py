"""统一控制台输出: 基于 rich 的标签化彩色日志.

所有 v2 模块通过此模块输出, 确保风格统一, DDP 安全.

用法::

    # 1) 非训练场景 (导出 / 推理 / split 等) — 便捷函数
    from cas_ocr_model.common.console import tag_print
    tag_print("export", "saved -> ./model.onnx")
    tag_print("split", "n_total=1000 n_train=800 n_val=100 n_test=100")

    # 2) 需要更多控制 — 获取 Console 实例
    from cas_ocr_model.common.console import get_console
    console = get_console()
    console.tag_print("init", "rank=0 world_size=8")
    console.rule("Epoch 1/100", style="bold blue")
    console.print("[bold green]Done![/]")

    # 3) DDP 训练场景 — AcceleratorConsole 自动 rank 过滤
    from cas_ocr_model.common.console import AcceleratorConsole
    console = AcceleratorConsole(accelerator)
    console.tag_print("train", "loss=0.5432 acc=0.8123")
    console.rule("Epoch 1/100")
"""
from __future__ import annotations

import sys
from typing import Any, TextIO

from rich.console import Console as RichConsole
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ---------------------------------------------------------------------------
# Tag → rich style 映射
# ---------------------------------------------------------------------------
TAG_STYLES: dict[str, str] = {
    # 训练阶段
    "init": "bold cyan",
    "config": "bold blue",
    "data": "bold green",
    "model": "bold magenta",
    "model-stats": "bold magenta",
    "train": "bold yellow",
    "val": "bold bright_blue",
    "test": "bold bright_green",
    "eval": "bold green",
    "epoch-summary": "bold white",
    # 检查点 & 停止
    "ckpt": "bold bright_yellow",
    "resume": "bold cyan",
    "early-stop": "bold red",
    "nonfinite-backprop": "bold red",
    "nonfinite-stop": "bold bright_red",
    # 导出 & 发布
    "export": "bold cyan",
    "export-ncnn-python": "bold cyan",
    "release-export": "bold cyan",
    # 数据集
    "split": "bold blue",
    # 服务
    "api-server": "bold green",
    # 推理后端
    "onnx-backend": "bold cyan",
    "ncnn-backend": "bold cyan",
    # 性能测试
    "benchmark": "bold cyan",
    "single-gpu-bench": "bold cyan",
    # 通用状态
    "done": "bold bright_green",
    "saved": "bold green",
    "WARN": "bold yellow",
}

_THEME = Theme(
    {f"tag.{k}": v for k, v in TAG_STYLES.items()}
    | {
        "metric.good": "bold green",
        "metric.bad": "bold red",
        "metric.best": "bold bright_green",
        "highlight.best": "bold bright_green on dark_green",
    }
)

# 默认标签样式 (未在 TAG_STYLES 中注册的 tag)
_DEFAULT_TAG_STYLE = "bold white"


# ---------------------------------------------------------------------------
# Console — 基础控制台
# ---------------------------------------------------------------------------
class Console:
    """统一控制台输出, 支持标签化彩色日志.

    自动检测终端能力; 非交互式环境 (管道 / 重定向) 下降级为纯文本.
    """

    def __init__(
        self,
        file: TextIO | None = None,
        *,
        force_terminal: bool | None = None,
    ) -> None:
        self._console = RichConsole(
            file=file or sys.stdout,
            theme=_THEME,
            highlight=False,
            force_terminal=force_terminal,
        )

    # -- 核心方法 ----------------------------------------------------------

    def print(
        self,
        message: str = "",
        *,
        markup: bool = True,
        highlight: bool = False,
        **kwargs: Any,
    ) -> None:
        """通用输出, 支持可选的 rich markup."""
        self._console.print(message, markup=markup, highlight=highlight, **kwargs)

    def tag_print(self, tag: str, message: str = "") -> None:
        """标签化输出: ``[tag] message`` 格式, tag 自动着色.

        使用 Text 对象拼接, 无 markup 转义问题.

        Examples::

            console.tag_print("init", "rank=0 world_size=8")
            # 渲染: [init] rank=0 world_size=8   (init 为 bold cyan)
        """
        style = TAG_STYLES.get(tag, _DEFAULT_TAG_STYLE)
        tag_text = Text(f"[{tag}]", style=style)
        if message:
            self._console.print(Text.assemble(tag_text, Text(f" {message}")))
        else:
            self._console.print(tag_text)

    def rule(self, title: str = "", *, style: str = "bold blue", **kwargs: Any) -> None:
        """输出带标题的分隔线."""
        self._console.rule(title, style=style, **kwargs)

    # -- 便捷方法 ----------------------------------------------------------

    def success(self, message: str) -> None:
        """输出成功消息 (绿色 ✓)."""
        self._console.print(f"[bold bright_green]✓[/] {message}")

    def warning(self, message: str) -> None:
        """输出警告消息 (黄色 ⚠)."""
        self._console.print(f"[bold yellow]⚠[/] {message}")

    def error(self, message: str) -> None:
        """输出错误消息 (红色 ✗)."""
        self._console.print(f"[bold red]✗[/] {message}")

    # -- 底层访问 ----------------------------------------------------------

    @property
    def rich(self) -> RichConsole:
        """底层 rich Console, 用于 Table / Panel 等高级场景."""
        return self._console


# ---------------------------------------------------------------------------
# AcceleratorConsole — DDP 安全控制台
# ---------------------------------------------------------------------------
class AcceleratorConsole(Console):
    """DDP 安全的控制台输出, 仅 rank 0 打印.

    用法::

        console = AcceleratorConsole(accelerator)
        console.tag_print("init", "rank=0 world_size=8")
    """

    def __init__(self, accelerator: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._accelerator = accelerator

    def print(
        self,
        message: str = "",
        *,
        markup: bool = True,
        highlight: bool = False,
        **kwargs: Any,
    ) -> None:
        if not self._accelerator.is_main_process:
            return
        super().print(message, markup=markup, highlight=highlight, **kwargs)

    def tag_print(self, tag: str, message: str = "") -> None:
        if not self._accelerator.is_main_process:
            return
        super().tag_print(tag, message)

    def rule(self, title: str = "", *, style: str = "bold blue", **kwargs: Any) -> None:
        if not self._accelerator.is_main_process:
            return
        super().rule(title, style=style, **kwargs)

    def success(self, message: str) -> None:
        if not self._accelerator.is_main_process:
            return
        super().success(message)

    def warning(self, message: str) -> None:
        if not self._accelerator.is_main_process:
            return
        super().warning(message)

    def error(self, message: str) -> None:
        if not self._accelerator.is_main_process:
            return
        super().error(message)


# ---------------------------------------------------------------------------
# 全局单例 & 便捷函数
# ---------------------------------------------------------------------------
_default_console: Console | None = None


def get_console(
    file: TextIO | None = None,
    *,
    force_terminal: bool | None = None,
) -> Console:
    """获取全局 Console 实例 (单例)."""
    global _default_console
    if _default_console is None:
        _default_console = Console(file=file, force_terminal=force_terminal)
    return _default_console


def tag_print(tag: str, message: str = "") -> None:
    """全局便捷函数: 标签化彩色输出."""
    get_console().tag_print(tag, message)


# ---------------------------------------------------------------------------
# Benchmark 表格 — 通用 rich.Table 输出
# ---------------------------------------------------------------------------
def print_benchmark_table(
    *,
    title: str,
    backend: str,
    device: str,
    image_size: tuple[int, int] | list[int],
    python_version: str,
    torch_version: str,
    single_stats: dict[str, float],
    single_qps: float,
    batch_scan: dict[Any, dict[str, float]],
    batch_throughput: dict[Any, float],
    peak_memory_mb: float,
) -> None:
    """使用 rich.Table 输出 benchmark 报告, 替代手动 print 格式化.

    ``single_stats`` / ``batch_scan`` 中的 dict 需包含:
        mean_ms, p50_ms, p90_ms, p99_ms

    Examples::

        print_benchmark_table(
            title="Benchmark",
            backend="pytorch",
            device="cuda",
            image_size=(64, 192),
            python_version="3.12",
            torch_version="2.4",
            single_stats={"mean_ms": 2.5, "p50_ms": 2.3, "p90_ms": 3.1, "p99_ms": 4.2},
            single_qps=400.0,
            batch_scan={8: {"mean_ms": 1.2, "p50_ms": 1.1, "p90_ms": 1.5, "p99_ms": 2.0}},
            batch_throughput={8: 6400.0},
            peak_memory_mb=256.0,
        )
    """
    console = get_console()

    # 标题 & 环境信息
    console.print()
    console.print(f"[bold cyan]⏱  {title}[/]")
    console.print(
        f"  [dim]backend=[/]{backend}  [dim]device=[/]{device}  "
        f"[dim]image=[/]{tuple(image_size) if isinstance(image_size, list) else image_size}"
    )
    console.print(
        f"  [dim]python=[/]{python_version}  [dim]torch=[/]{torch_version}"
    )

    # 延迟表格
    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("bs", justify="right", style="cyan", width=6)
    table.add_column("mean (ms)", justify="right", width=10)
    table.add_column("p50 (ms)", justify="right", width=10)
    table.add_column("p90 (ms)", justify="right", width=10)
    table.add_column("p99 (ms)", justify="right", width=10)
    table.add_column("qps", justify="right", style="green", width=10)

    # bs=1 行 (高亮)
    s = single_stats
    table.add_row(
        "[bold]1[/]",
        f"[bold]{s['mean_ms']:.2f}[/]",
        f"[bold]{s['p50_ms']:.2f}[/]",
        f"[bold]{s['p90_ms']:.2f}[/]",
        f"[bold]{s['p99_ms']:.2f}[/]",
        f"[bold green]{single_qps:.1f}[/]",
    )

    # batch scan 行
    for bs_key, stats in batch_scan.items():
        qps = batch_throughput.get(bs_key, 0.0)
        table.add_row(
            str(bs_key),
            f"{stats['mean_ms']:.2f}",
            f"{stats['p50_ms']:.2f}",
            f"{stats['p90_ms']:.2f}",
            f"{stats['p99_ms']:.2f}",
            f"{qps:.1f}",
        )

    console.print(table)
    console.print(f"  [dim]peak memory=[/][bold]{peak_memory_mb:.1f} MB[/]")
    console.print()
