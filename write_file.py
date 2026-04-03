#!/usr/bin/env python3
"""
write_file.py — 通过 stdin 写入大型文件，彻底绕过 MAX_ARG_STRLEN 128KB 限制。

用法：
    python3 write_file.py /path/to/output.md << 'EOF'
    # 大量 Markdown 内容...
    EOF

    cat content.md   | python3 write_file.py /path/to/output.md
    python3 write_file.py /path/to/output.md < content.md
    python3 write_file.py /path/to/output.md --append   # 追加模式
    python3 write_file.py /path/to/output.md --backup   # 写前备份
    python3 write_file.py /path/to/output.md --dry-run  # 仅统计，不写文件
"""

import sys
import os
import argparse
import shutil
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 stdin 读取内容并写入文件（绕过 shell 参数长度限制）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "file_path",
        metavar="FILE",
        help="目标文件路径（绝对路径或相对路径）",
    )
    parser.add_argument(
        "--append", "-a",
        action="store_true",
        default=False,
        help="追加模式：在文件末尾追加内容（默认：覆盖）",
    )
    parser.add_argument(
        "--backup", "-b",
        action="store_true",
        default=False,
        help="写入前备份原文件为 <file>.bak.<timestamp>",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="仅读取 stdin 并显示统计信息，不实际写文件（用于测试）",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="文件编码（默认：utf-8）",
    )
    return parser.parse_args()


def make_backup(path: Path) -> Path:
    """创建时间戳备份，返回备份路径。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_suffix(f".bak.{ts}{path.suffix}")
    shutil.copy2(path, backup_path)
    return backup_path


def read_stdin(encoding: str) -> str:
    """以指定编码从 stdin 读取全部内容。
    用 sys.stdin.buffer 先读字节再 decode，比直接 sys.stdin.read() 更稳健
    （避免平台换行符自动转换问题，正确处理 BOM 等）。
    """
    raw = sys.stdin.buffer.read()
    return raw.decode(encoding)


def write_file(file_path: Path, content: str, append: bool, encoding: str) -> int:
    """写入文件，返回写入字节数。newline='' 禁止自动换行符转换，原样保存。"""
    mode = "a" if append else "w"
    with open(file_path, mode, encoding=encoding, newline="") as f:
        f.write(content)
    return len(content.encode(encoding))


def format_size(num_bytes: int) -> str:
    """人类可读的文件大小。"""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def main() -> int:
    args = parse_args()

    # ── 1. 解析目标路径 ────────────────────────────────────────────────────
    target = Path(args.file_path).expanduser().resolve()

    # ── 2. 从 stdin 读取内容 ────────────────────────────────────────────────
    if sys.stdin.isatty():
        print(
            "[write_file] 等待 stdin 输入（Ctrl+D 结束）…",
            file=sys.stderr,
        )

    try:
        content = read_stdin(args.encoding)
    except UnicodeDecodeError as e:
        print(f"[write_file] ✗ stdin 解码失败（{args.encoding}）: {e}", file=sys.stderr)
        return 2

    # ── 3. 统计信息 ────────────────────────────────────────────────────────
    num_bytes = len(content.encode(args.encoding))
    num_lines = content.count("\n")
    num_chars = len(content)

    if args.dry_run:
        print(f"[write_file] --dry-run 模式，不写入文件")
        print(f"  目标路径  : {target}")
        print(f"  内容大小  : {format_size(num_bytes)}（{num_bytes:,} bytes）")
        print(f"  行数      : {num_lines:,} 行")
        print(f"  字符数    : {num_chars:,} 字符")
        print(f"  写入模式  : {'追加' if args.append else '覆盖'}")
        return 0

    # ── 4. 创建目录（如不存在）────────────────────────────────────────────
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        print(f"[write_file] ✗ 无法创建目录 {target.parent}: {e}", file=sys.stderr)
        return 3

    # ── 5. 备份原文件（如需要） ────────────────────────────────────────────
    if args.backup and target.exists():
        try:
            backup_path = make_backup(target)
            print(f"[write_file] 已备份原文件 → {backup_path}", file=sys.stderr)
        except Exception as e:
            print(f"[write_file] ✗ 备份失败: {e}", file=sys.stderr)
            return 4

    # ── 6. 写入文件 ────────────────────────────────────────────────────────
    try:
        written = write_file(target, content, args.append, args.encoding)
    except PermissionError as e:
        print(f"[write_file] ✗ 权限错误，无法写入 {target}: {e}", file=sys.stderr)
        return 3
    except OSError as e:
        print(f"[write_file] ✗ 写入失败: {e}", file=sys.stderr)
        return 5

    # ── 7. 成功输出 ────────────────────────────────────────────────────────
    mode_label = "追加" if args.append else "写入"
    print(
        f"[write_file] ✓ {mode_label}成功 → {target}\n"
        f"  大小: {format_size(written)}（{written:,} bytes）| "
        f"行数: {num_lines:,} | 字符数: {num_chars:,}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
