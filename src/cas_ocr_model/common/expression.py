"""验证码表达式解析与标签归一化."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


CANONICAL_OPERATOR_LABELS: tuple[str, ...] = ("+", "-", "*")
OPERATOR_ALIASES: dict[str, str] = {
    "+": "+",
    "＋": "+",
    "加": "+",
    "加上": "+",
    "-": "-",
    "－": "-",
    "减": "-",
    "减去": "-",
    "*": "*",
    "×": "*",
    "x": "*",
    "X": "*",
    "乘": "*",
    "乘以": "*",
}
EQUAL_ALIASES: tuple[str, ...] = ("=", "＝", "等", "等于")

_SORTED_OPERATOR_ALIASES = sorted(OPERATOR_ALIASES, key=len, reverse=True)
_SORTED_EQUAL_ALIASES = sorted(EQUAL_ALIASES, key=len, reverse=True)


@dataclass(frozen=True)
class ParsedExpression:
    digit_left: str
    operator: str
    digit_right: str
    equal_token: Optional[str] = None
    answer: Optional[int] = None


def _strip_spaces(text: str) -> str:
    return "".join(str(text).split())


def _match_prefix(text: str, candidates: list[str]) -> Optional[str]:
    for token in candidates:
        if text.startswith(token):
            return token
    return None


def parse_captcha_expression(expr: str) -> Optional[ParsedExpression]:
    """解析 `1+2=` / `1加2等于3` 这类验证码表达式."""
    text = _strip_spaces(expr)
    if len(text) < 3 or not text[0].isdigit():
        return None

    digit_left = text[0]
    rest = text[1:]

    op_token = _match_prefix(rest, _SORTED_OPERATOR_ALIASES)
    if op_token is None:
        return None
    operator = OPERATOR_ALIASES[op_token]
    rest = rest[len(op_token):]

    if not rest or not rest[0].isdigit():
        return None
    digit_right = rest[0]
    rest = rest[1:]

    equal_token: Optional[str] = None
    if rest:
        eq_token = _match_prefix(rest, _SORTED_EQUAL_ALIASES)
        if eq_token is not None:
            equal_token = eq_token
            rest = rest[len(eq_token):]

    answer: Optional[int] = None
    if rest:
        if not rest.isdigit():
            return None
        answer = int(rest)

    return ParsedExpression(
        digit_left=digit_left,
        operator=operator,
        digit_right=digit_right,
        equal_token=equal_token,
        answer=answer,
    )

