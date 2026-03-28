#!/usr/bin/env python3
import argparse
import re
import sys
from typing import Optional


def fetch_html(url: str, timeout: int = 15) -> str:
    try:
        import requests
    except ModuleNotFoundError:
        print("缺少 requests，請先執行: pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(2)

    headers = {
        "User-Agent": "web_reader_agent/1.0 (+https://local.tool)",
        "Accept": "text/html,application/xhtml+xml",
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text


def extract_text(html: str) -> tuple[Optional[str], str]:
    try:
        from bs4 import BeautifulSoup
    except ModuleNotFoundError:
        print("缺少 beautifulsoup4，請先執行: pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(2)

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else None
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()
    return title, text


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="web_reader_agent",
        description="讀取網頁並輸出純文字內容",
    )
    parser.add_argument("url", help="目標網址，例如 https://example.com")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="最多輸出字元數（預設：2000）",
    )
    args = parser.parse_args()

    try:
        html = fetch_html(args.url)
        title, text = extract_text(html)
    except Exception as exc:
        if exc.__class__.__name__ == "HTTPError" or exc.__class__.__name__ == "RequestException":
            print(f"讀取失敗: {exc}", file=sys.stderr)
            return 1
        raise

    if title:
        print(f"# {title}\n")

    print(text[: args.max_chars])
    if len(text) > args.max_chars:
        print("\n... (內容已截斷)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
