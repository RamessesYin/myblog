#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 done/type1.md 模板，为每条链接生成 todo/task1.md, task2.md, ...
「参考论文」小节内仅包含对应的一条链接（保留其余章节与原文一致）。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REF_HEADING = "# 参考论文"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_template(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if REF_HEADING not in text:
        raise ValueError(f"模板中未找到「{REF_HEADING}」：{path}")
    return text


def inject_single_reference(template: str, reference_line: str) -> str:
    """在「# 参考论文」标题后插入一条内容，直到下一个「\\n# 」开头的章节为止。"""
    ref = reference_line.strip()
    if not ref:
        raise ValueError("链接/参考行不能为空")

    pattern = re.compile(
        rf"({re.escape(REF_HEADING)}\n)\n*",
        re.MULTILINE,
    )
    new_text, n = pattern.subn(rf"\1{ref}\n\n", template, count=1)
    if n != 1:
        raise ValueError("无法在模板中唯一替换「参考论文」小节起始位置")
    return new_text


def collect_urls(args: argparse.Namespace) -> list[str]:
    urls: list[str] = []
    if args.file:
        p = Path(args.file)
        lines = p.read_text(encoding="utf-8").splitlines()
        urls.extend(line.strip() for line in lines if line.strip() and not line.strip().startswith("#"))
    urls.extend(u.strip() for u in args.urls if u.strip())
    if not urls:
        print("错误：未提供任何链接（命令行参数或 --file）", file=sys.stderr)
        sys.exit(1)
    return urls


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description="按 type1 模板为每条链接生成 todo/taskN.md")
    parser.add_argument(
        "urls",
        nargs="*",
        help="论文或博客链接（可多个）",
    )
    parser.add_argument(
        "-f",
        "--file",
        metavar="PATH",
        help="从文件读取链接，每行一条；以 # 开头的行视为注释忽略",
    )
    parser.add_argument(
        "-t",
        "--template",
        type=Path,
        default=root / "done" / "type1.md",
        help="模板路径（默认：仓库内 done/type1.md）",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=root / "todo",
        help="输出目录（默认：仓库内 todo/）",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        metavar="N",
        help="task 编号起始值（默认 1，即 task1.md）",
    )
    args = parser.parse_args()

    template_path = args.template.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    template = load_template(template_path)
    urls = collect_urls(args)

    if args.start < 1:
        print("错误：--start 须 >= 1", file=sys.stderr)
        sys.exit(1)

    for i, url in enumerate(urls):
        n = args.start + i
        body = inject_single_reference(template, url)
        out_path = out_dir / f"task{n}.md"
        out_path.write_text(body, encoding="utf-8", newline="\n")
        print(out_path)

    print(f"已生成 {len(urls)} 个文件。", file=sys.stderr)


if __name__ == "__main__":
    main()
