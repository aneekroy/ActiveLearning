"""Utilities to build prompts for classification and multichoice."""

from typing import List, Tuple


def build_classification_prompt(demos: List[Tuple[str, str]], test_input: str) -> str:
    lines = []
    for idx, (text, label) in enumerate(demos, 1):
        lines.append(f"Example {idx}:")
        lines.append(f"Input: {text}")
        lines.append(f"Output: {label}")
        lines.append("")
    lines.append("Test:")
    lines.append(f"Input: {test_input}")
    lines.append("Output:")
    return "\n".join(lines)


def build_multichoice_prompt(demos: List[Tuple[str, List[str], str]], question: str, choices: List[str]) -> str:
    lines = []
    for idx, (q, ch, ans) in enumerate(demos, 1):
        lines.append(f"Example {idx}:")
        lines.append(f"Q: {q}")
        lines.append("Choices:")
        for letter, option in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", ch):
            lines.append(f"{letter}) {option}")
        lines.append(f"Answer: {ans}")
        lines.append("")
    lines.append("Test:")
    lines.append(f"Q: {question}")
    lines.append("Choices:")
    for letter, option in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", choices):
        lines.append(f"{letter}) {option}")
    lines.append("Answer:")
    return "\n".join(lines)
