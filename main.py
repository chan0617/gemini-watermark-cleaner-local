"""Entry point.

    python main.py            one batch pass over input/
    python main.py --watch    keep polling input/ and process new files as they arrive
"""
from __future__ import annotations

import argparse
import time

from src.pipeline import RunSummary, run_once


def print_summary(summary: RunSummary) -> None:
    processed = summary.success + summary.failed
    print("=" * 40)
    print(f"총 {processed}장")
    print(f"성공 {summary.success}장")
    print(f"실패 {summary.failed}장")
    if summary.skipped:
        print(f"(건너뜀 {summary.skipped}장 - 이미 처리됨)")
    print("=" * 40)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini watermark cleaner (local, offline)")
    parser.add_argument("--watch", action="store_true", help="input/ 폴더를 계속 감시하며 새 이미지를 자동 처리합니다.")
    parser.add_argument("--interval", type=float, default=2.0, help="watch 모드 폴링 주기(초, 기본 2초).")
    args = parser.parse_args()

    if args.watch:
        print("Watch mode 시작 — input/ 폴더를 감시합니다. 종료하려면 Ctrl+C.")
        try:
            while True:
                summary = run_once()
                if summary.total:
                    print_summary(summary)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nWatch mode 종료.")
    else:
        summary = run_once()
        print_summary(summary)


if __name__ == "__main__":
    main()
