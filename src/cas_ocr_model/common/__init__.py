"""跨训练 / 推理共享的领域公共模块."""

from .console import AcceleratorConsole, Console, get_console, tag_print

__all__ = [
    "AcceleratorConsole",
    "Console",
    "get_console",
    "tag_print",
]

