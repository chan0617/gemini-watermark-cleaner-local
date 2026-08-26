"""Entry point.

    python main.py            one batch pass over input/
    python main.py --watch    keep polling input/ and process new files as they arrive
    python main.py --web      open a local browser page to upload/delete files and run processing
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
    parser.add_argument("--web", action="store_true", help="브라우저에서 업로드/삭제할 수 있는 로컬 웹페이지를 엽니다 (127.0.0.1만 사용).")
    parser.add_argument("--port", type=int, default=8765, help="웹 UI 포트 (기본 8765).")
    args = parser.parse_args()

    if args.web:
        import threading
        import webbrowser

        from webui import app

        url = f"http://127.0.0.1:{args.port}"
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        print(f"웹 UI 시작: {url} (종료하려면 Ctrl+C)")
        app.run(host="127.0.0.1", port=args.port, threaded=True)
    elif args.watch:
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
