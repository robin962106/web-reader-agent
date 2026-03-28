from __future__ import annotations
        
import json
import os
import re
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, HttpUrl
from dotenv import load_dotenv

APP_NAME = "OpenClaw-like Web Reader Agent"
APP_VERSION = "1.0.0"

# 自動載入專案根目錄的 .env 設定
load_dotenv()

KEYWORD_RULES = {
    "使用者登入與帳號管理": ["login", "sign in", "sign up", "account", "會員", "註冊", "登入"],
    "商品展示與購物車": ["cart", "checkout", "shop", "pricing", "buy", "商品", "購物車", "結帳"],
    "搜尋功能": ["search", "find", "query", "搜尋"],
    "內容發布或部落格": ["blog", "article", "news", "post", "文章", "最新消息"],
    "客服與聯絡方式": ["contact", "support", "help", "客服", "聯絡我們", "faq"],
    "文件與教學資源": ["docs", "documentation", "guide", "api", "教學", "文件"],
    "預約或行程管理": ["book", "booking", "schedule", "appointment", "預約", "行程"],
}


class SummarizeRequest(BaseModel):
    url: HttpUrl = Field(..., description="要分析的網頁網址")
    max_chars: int = Field(6000, ge=500, le=50000, description="抓取文字上限")


class SummarizeResponse(BaseModel):
    agent: str
    url: HttpUrl
    domain: str
    title: str | None
    summary: str
    summary_source: str
    key_features: list[str]
    evidence: list[str]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="使用者輸入，建議貼網址")
    max_chars: int = Field(6000, ge=500, le=50000)


class ChatResponse(BaseModel):
    user: str
    reply: str


class KeywordSearchRequest(BaseModel):
    keyword: str = Field(..., min_length=2, max_length=200, description="搜尋關鍵字")
    max_results: int = Field(5, ge=1, le=10)


class SearchItem(BaseModel):
    title: str
    snippet: str
    source_url: str


class KeywordSearchResponse(BaseModel):
    keyword: str
    summary: str
    summary_source: str
    crawled_count: int
    results: list[SearchItem]


app = FastAPI(title=APP_NAME, version=APP_VERSION)


def _fetch_html(url: str, timeout: int = 20) -> str:
    headers = {
        "User-Agent": "web_reader_agent/2.0 (+local-fastapi)",
        "Accept": "text/html,application/xhtml+xml",
    }
    resp = requests.get(url, timeout=timeout, headers=headers)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")
    if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
        raise ValueError(f"不支援的內容類型: {ctype}")
    return resp.text


def _fetch_html_relaxed(url: str, timeout: int = 15) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    }
    resp = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
    resp.raise_for_status()
    return resp.text


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def _extract_content(html: str, max_chars: int) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "footer"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else None

    meta_desc = None
    meta_desc_tag = soup.find("meta", attrs={"name": "description"})
    if meta_desc_tag and meta_desc_tag.get("content"):
        meta_desc = meta_desc_tag["content"].strip()

    headings = [h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2", "h3"])[:10]]

    blocks = []
    for tag in soup.find_all(["p", "li"])[:120]:
        chunk = tag.get_text(" ", strip=True)
        if len(chunk) > 25:
            blocks.append(chunk)

    raw_text = "\n".join(blocks)
    clean = _clean_text(raw_text)[:max_chars]

    return {
        "title": title,
        "meta_desc": meta_desc,
        "headings": headings,
        "text": clean,
    }


def _infer_features(title: str | None, meta_desc: str | None, headings: list[str], text: str) -> tuple[list[str], list[str]]:
    corpus = "\n".join([title or "", meta_desc or "", *headings, text]).lower()

    detected = []
    evidence = []
    for feature, keywords in KEYWORD_RULES.items():
        matched = [kw for kw in keywords if kw.lower() in corpus]
        if matched:
            detected.append(feature)
            evidence.append(f"{feature}: {', '.join(matched[:3])}")

    if not detected:
        detected = ["資訊展示與內容導覽"]
        evidence = ["未偵測到明確交易或互動關鍵字，以內容展示型網站判定"]

    return detected, evidence


def _make_summary(title: str | None, meta_desc: str | None, headings: list[str], features: list[str], text: str) -> str:
    parts = []
    if title:
        parts.append(f"此網站主題為「{title}」。")
    if meta_desc:
        parts.append(f"根據頁面描述，重點為：{meta_desc}")
    if headings:
        parts.append(f"主要內容區塊包含：{'; '.join(headings[:4])}。")

    parts.append(f"推測核心功能有：{', '.join(features)}。")

    snippet = text[:280].replace("\n", " ")
    if snippet:
        parts.append(f"內容摘要片段：{snippet}")

    return " ".join(parts)


def _build_llm_prompt(content: dict[str, Any], features: list[str]) -> str:
    return (
        "請一律使用繁體中文（台灣用語）。"
        "你是網站分析 AI Agent。請根據輸入內容，輸出精簡中文摘要，"
        "重點放在此網站提供的功能、目標使用者與可能使用情境。"
        "控制在 120~220 字。\n\n"
        f"標題: {content.get('title')}\n"
        f"描述: {content.get('meta_desc')}\n"
        f"推測功能: {', '.join(features)}\n"
        f"段落內容:\n{content.get('text', '')[:3000]}"
    )


def _extract_json_payload(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        return json.loads(snippet)
    raise ValueError("LLM 未回傳可解析 JSON")


def _build_direct_url_prompt(url: str) -> str:
    return (
        "請使用繁體中文（台灣用語）分析這個網址，並嚴格輸出 JSON，不要輸出其他文字。\n"
        "JSON 格式："
        '{"title":"", "summary":"", "key_features":[""], "evidence":[""]}\n'
        "要求：summary 120~220 字，key_features 3~6 項，evidence 至少 2 項。\n"
        f"目標網址: {url}"
    )


def _summarize_with_openai(prompt: str, timeout: int = 30) -> str:
    # 在這裡填入 OpenAI API Key:
    # export OPENAI_API_KEY="你的金鑰"
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 未設定")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/responses")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    system_msg = "你是精準的網站功能分析助手。所有回覆必須使用繁體中文，不可使用簡體中文。"

    # 預設走 Responses API（新版）
    if url.endswith("/responses"):
        payload = {
            "model": model,
            "instructions": system_msg,
            "input": prompt,
            "temperature": 0.2,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code >= 400:
            raise ValueError(f"OpenAI Responses API 錯誤 {resp.status_code}: {resp.text[:400]}")
        data = resp.json()
        text = data.get("output_text")
        if text:
            return text.strip()
        # 兼容部分回傳格式
        output = data.get("output", [])
        chunks: list[str] = []
        for item in output:
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    chunks.append(content["text"])
        if chunks:
            return "\n".join(chunks).strip()
        raise ValueError("OpenAI Responses API 未回傳可用文字")

    # 舊版 Chat Completions（相容用）
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise ValueError(f"OpenAI Chat API 錯誤 {resp.status_code}: {resp.text[:400]}")
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _summarize_url_direct_with_openai(url: str, timeout: int = 30) -> dict[str, Any]:
    prompt = _build_direct_url_prompt(url)
    raw = _summarize_with_openai(prompt, timeout=timeout)
    payload = _extract_json_payload(raw)
    return {
        "title": payload.get("title"),
        "summary": str(payload.get("summary", "")).strip(),
        "key_features": payload.get("key_features") or [],
        "evidence": payload.get("evidence") or [],
    }


def _summarize_with_google(prompt: str, timeout: int = 30) -> str:
    # 在這裡填入 Google AI Studio API Key:
    # export GOOGLE_API_KEY="你的金鑰"
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY 未設定")

    model = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": f"請全部使用繁體中文回答。\n\n{prompt}"}]}],
        "generationConfig": {"temperature": 0.2},
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def _summarize_url_direct_with_google(url: str, timeout: int = 30) -> dict[str, Any]:
    prompt = _build_direct_url_prompt(url)
    raw = _summarize_with_google(prompt, timeout=timeout)
    payload = _extract_json_payload(raw)
    return {
        "title": payload.get("title"),
        "summary": str(payload.get("summary", "")).strip(),
        "key_features": payload.get("key_features") or [],
        "evidence": payload.get("evidence") or [],
    }


def _summarize_with_provider(content: dict[str, Any], features: list[str]) -> tuple[str, str]:
    # 可切換: mock / openai / google
    provider = os.getenv("LLM_PROVIDER", "mock").strip().lower()
    prompt = _build_llm_prompt(content, features)

    if provider == "openai":
        return _summarize_with_openai(prompt), "openai"
    if provider == "google":
        return _summarize_with_google(prompt), "google_ai_studio"
    return (
        _make_summary(content["title"], content["meta_desc"], content["headings"], features, content["text"]),
        "heuristic",
    )


def _get_summary_mode() -> str:
    # extract / direct_url / hybrid
    return os.getenv("SUMMARY_MODE", "extract").strip().lower()


def _allow_llm_fallback() -> bool:
    # 預設關閉，避免「看起來有輸出但其實是規則摘要」造成誤判
    return os.getenv("ALLOW_LLM_FALLBACK", "false").strip().lower() in {"1", "true", "yes", "on"}


def _build_reply_from_url(url_text: str, max_chars: int) -> str:
    url = url_text.strip()
    domain = urlparse(url).netloc
    if not domain:
        return "請輸入有效網址，例如：https://example.com"

    try:
        html = _fetch_html(url)
        content = _extract_content(html, max_chars)
    except requests.RequestException as exc:
        return f"讀取網址失敗：{exc}"
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"分析失敗：{exc}"

    features, _evidence = _infer_features(
        content["title"], content["meta_desc"], content["headings"], content["text"]
    )
    try:
        summary, _source = _summarize_with_provider(content, features)
    except Exception:
        summary = _make_summary(
            content["title"], content["meta_desc"], content["headings"], features, content["text"]
        )
    return summary


def _resolve_result_url(raw_href: str) -> str:
    if not raw_href:
        return ""
    if raw_href.startswith("http://") or raw_href.startswith("https://"):
        return raw_href
    if raw_href.startswith("/l/?"):
        query = parse_qs(urlparse(raw_href).query)
        uddg = query.get("uddg", [""])[0]
        return unquote(uddg) if uddg else raw_href
    return raw_href


def _search_duckduckgo_html(keyword: str, max_results: int = 5) -> list[dict[str, str]]:
    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "web_reader_agent/2.0 (+local-fastapi)",
        "Accept": "text/html,application/xhtml+xml",
    }
    resp = requests.get(url, params={"q": keyword}, headers=headers, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = soup.select(".result")
    items: list[dict[str, str]] = []
    for row in rows:
        a = row.select_one("a.result__a")
        if not a:
            continue
        title = a.get_text(" ", strip=True)
        href = _resolve_result_url(a.get("href", "").strip())
        snippet_node = row.select_one(".result__snippet")
        snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""

        if not title or not href:
            continue
        items.append({"title": title, "snippet": snippet, "source_url": href})
        if len(items) >= max_results:
            break
    return items


def _search_duckduckgo_lite(keyword: str, max_results: int = 5) -> list[dict[str, str]]:
    url = "https://lite.duckduckgo.com/lite/"
    headers = {
        "User-Agent": "web_reader_agent/2.0 (+local-fastapi)",
        "Accept": "text/html,application/xhtml+xml",
    }
    resp = requests.post(url, data={"q": keyword}, headers=headers, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    items: list[dict[str, str]] = []
    for a in soup.select("a[href]"):
        href = _resolve_result_url((a.get("href") or "").strip())
        title = a.get_text(" ", strip=True)
        if not href or not title:
            continue
        if href.startswith("https://duckduckgo.com/"):
            continue
        items.append({"title": title, "snippet": "", "source_url": href})
        if len(items) >= max_results:
            break
    return items


def _search_duckduckgo(keyword: str, max_results: int = 5) -> list[dict[str, str]]:
    # 優先 HTML 版本，失敗改用 Lite 版本
    try:
        items = _search_duckduckgo_html(keyword, max_results)
        if items:
            return items
    except Exception:
        pass
    return _search_duckduckgo_lite(keyword, max_results)


def _search_google_cse(keyword: str, max_results: int = 5) -> list[dict[str, str]]:
    api_key = os.getenv("GOOGLE_CSE_API_KEY", "").strip()
    cx = os.getenv("GOOGLE_CSE_CX", "").strip()
    if not api_key or not cx:
        raise ValueError("Google 搜尋未設定：請設定 GOOGLE_CSE_API_KEY 與 GOOGLE_CSE_CX")

    endpoint = "https://www.googleapis.com/customsearch/v1"
    resp = requests.get(
        endpoint,
        params={"key": api_key, "cx": cx, "q": keyword, "num": min(max_results, 10)},
        timeout=20,
    )
    if resp.status_code >= 400:
        raise ValueError(f"Google CSE 錯誤 {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    items = []
    for row in data.get("items", []):
        title = (row.get("title") or "").strip()
        source_url = (row.get("link") or "").strip()
        snippet = (row.get("snippet") or "").strip()
        if title and source_url:
            items.append({"title": title, "snippet": snippet, "source_url": source_url})
        if len(items) >= max_results:
            break
    return items


def _search_web(keyword: str, max_results: int = 5) -> tuple[list[dict[str, str]], str]:
    provider = os.getenv("SEARCH_PROVIDER", "duckduckgo").strip().lower()
    if provider == "google_cse":
        try:
            return _search_google_cse(keyword, max_results), "google_cse"
        except Exception:
            # 若 Google CSE 設定缺失或請求失敗，降級到 DuckDuckGo
            return _search_duckduckgo(keyword, max_results), "duckduckgo_fallback"
    if provider == "duckduckgo":
        return _search_duckduckgo(keyword, max_results), "duckduckgo"
    raise ValueError("不支援的 SEARCH_PROVIDER，請使用 duckduckgo 或 google_cse")


def _make_search_summary(keyword: str, items: list[dict[str, str]]) -> str:
    if not items:
        return f"找不到與「{keyword}」相關的搜尋結果。"
    lines = [f"關鍵字「{keyword}」共整理 {len(items)} 筆結果。"]
    top = items[:3]
    for idx, item in enumerate(top, start=1):
        snippet = item["snippet"][:90] if item["snippet"] else "無摘要片段"
        lines.append(f"{idx}. {item['title']}：{snippet}")
    return " ".join(lines)


def _crawl_search_results(items: list[dict[str, str]], max_sites: int = 3, max_chars: int = 2500) -> list[dict[str, str]]:
    crawled: list[dict[str, str]] = []
    for item in items[:max_sites]:
        source_url = item["source_url"]
        try:
            html = _fetch_html_relaxed(source_url, timeout=12)
            content = _extract_content(html, max_chars=max_chars)
            text = content.get("text", "").strip()
            if not text:
                continue
            crawled.append(
                {
                    "source_url": source_url,
                    "title": content.get("title") or item.get("title") or source_url,
                    "excerpt": text[:900],
                }
            )
        except Exception:
            continue
    return crawled


def _summarize_search_from_crawled(keyword: str, crawled: list[dict[str, str]]) -> str:
    if not crawled:
        return f"已搜尋「{keyword}」，但目前無法成功擷取網站內文。"
    lines = [f"關鍵字「{keyword}」已整理 {len(crawled)} 個網站內容。"]
    for idx, row in enumerate(crawled[:3], start=1):
        excerpt = row["excerpt"].replace("\n", " ")[:85]
        lines.append(f"{idx}. {row['title']}：{excerpt}")
    lines.append("可從下方來源網址進一步查證。")
    return " ".join(lines)


def _summarize_search_with_provider(keyword: str, crawled: list[dict[str, str]], fallback_items: list[dict[str, str]]) -> tuple[str, str]:
    provider = os.getenv("LLM_PROVIDER", "mock").strip().lower()
    if provider not in {"openai", "google"}:
        if crawled:
            return _summarize_search_from_crawled(keyword, crawled), "heuristic_search_crawl"
        return _make_search_summary(keyword, fallback_items), "heuristic_search_snippet"

    source_lines = []
    if crawled:
        for idx, item in enumerate(crawled, start=1):
            source_lines.append(
                f"[{idx}] 標題: {item['title']}\n網址: {item['source_url']}\n內文節錄: {item['excerpt']}"
            )
    else:
        for idx, item in enumerate(fallback_items, start=1):
            source_lines.append(
                f"[{idx}] 標題: {item['title']}\n網址: {item['source_url']}\n片段: {item['snippet']}"
            )
    prompt = (
        "請使用繁體中文摘要以下多網站資料，控制在 160~260 字。"
        "必須根據提供內容，不可捏造。結尾請提醒使用者可點來源網址驗證。\n\n"
        f"關鍵字: {keyword}\n\n" + "\n\n".join(source_lines)
    )

    if provider == "openai":
        return _summarize_with_openai(prompt), "openai_search_crawl"
    return _summarize_with_google(prompt), "google_ai_studio_search_crawl"


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": APP_NAME,
        "version": APP_VERSION,
        "llm_provider": os.getenv("LLM_PROVIDER", "mock").strip().lower(),
        "search_provider": os.getenv("SEARCH_PROVIDER", "duckduckgo").strip().lower(),
        "google_cse_key_set": "yes" if bool(os.getenv("GOOGLE_CSE_API_KEY", "").strip()) else "no",
        "google_cse_cx_set": "yes" if bool(os.getenv("GOOGLE_CSE_CX", "").strip()) else "no",
        "summary_mode": _get_summary_mode(),
        "allow_llm_fallback": "yes" if _allow_llm_fallback() else "no",
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Web Reader Agent 目錄式摘要</title>
  <style>
    :root {
      --bg:#f4f7fc;
      --panel:#ffffff;
      --line:#d7e0ee;
      --text:#1a2740;
      --muted:#61708b;
      --accent:#0057c2;
      --soft:#edf3ff;
    }
    body {
      margin:0;
      font-family:"Noto Sans TC","PingFang TC","Microsoft JhengHei",sans-serif;
      background:linear-gradient(160deg,#f9fbff 0%,#eef4ff 100%);
      color:var(--text);
    }
    .wrap { max-width:1100px; margin:24px auto; padding:0 14px; }
    .toolbar {
      background:var(--panel);
      border:1px solid var(--line);
      border-radius:14px;
      padding:14px;
      margin-bottom:12px;
    }
    h1 { margin:0 0 8px; font-size:22px; }
    .desc { margin:0 0 12px; color:var(--muted); }
    .row { display:flex; gap:8px; }
    input {
      flex:1;
      border:1px solid var(--line);
      border-radius:10px;
      padding:12px;
      font-size:15px;
      background:#fff;
    }
    button {
      border:none;
      border-radius:10px;
      padding:12px 16px;
      background:var(--accent);
      color:#fff;
      font-weight:700;
      cursor:pointer;
    }
    .layout {
      display:grid;
      grid-template-columns:260px 1fr;
      gap:12px;
    }
    .card {
      background:var(--panel);
      border:1px solid var(--line);
      border-radius:14px;
      padding:14px;
      min-height:440px;
    }
    .toc-title {
      font-size:12px;
      color:var(--muted);
      letter-spacing:0.08em;
      margin-bottom:8px;
    }
    .toc { margin:0; padding-left:18px; line-height:1.9; font-size:14px; }
    .toc a {
      color:var(--text);
      text-decoration:none;
      border-bottom:1px dotted #a8b7d3;
    }
    .section {
      margin-bottom:16px;
      padding:10px 12px;
      border:1px solid var(--line);
      border-radius:10px;
      background:#fff;
    }
    .section h2 { margin:0 0 8px; font-size:16px; }
    .list { margin:0; padding-left:18px; line-height:1.7; }
    .empty { color:var(--muted); white-space:pre-wrap; line-height:1.8; }
    @media (max-width: 900px) {
      .layout { grid-template-columns:1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="toolbar">
      <h1>Web Reader Agent 目錄式摘要頁</h1>
      <p class="desc">輸入網址後，系統會產生網站分析內容，並用章節目錄方式呈現。</p>
      <div class="row">
        <input id="urlInput" placeholder="請輸入網址，例如 https://example.com" />
        <button id="sendBtn">網址摘要</button>
      </div>
      <div class="row" style="margin-top:8px;">
        <input id="keywordInput" placeholder="或輸入關鍵字，例如：台灣 半導體 供應鏈" />
        <button id="searchBtn">關鍵字搜尋</button>
      </div>
    </div>

    <div class="layout">
      <aside class="card">
        <div class="toc-title">TABLE OF CONTENTS</div>
        <ol class="toc" id="toc">
          <li><a href="#sec-overview">1. 網站資訊</a></li>
          <li><a href="#sec-summary">2. 摘要</a></li>
          <li><a href="#sec-search">3. 搜尋結果與來源</a></li>
        </ol>
      </aside>

      <main class="card">
        <section class="section" id="sec-overview">
          <h2>1. 網站資訊</h2>
          <div class="empty">尚未分析，請先輸入網址。</div>
        </section>
        <section class="section" id="sec-summary">
          <h2>2. 摘要</h2>
          <div class="empty">等待分析結果。</div>
        </section>
        <section class="section" id="sec-search">
          <h2>3. 搜尋結果與來源</h2>
          <div class="empty">尚未搜尋，請輸入關鍵字。</div>
        </section>
      </main>
    </div>
  </div>

  <script>
    const input = document.getElementById("urlInput");
    const btn = document.getElementById("sendBtn");
    const keywordInput = document.getElementById("keywordInput");
    const searchBtn = document.getElementById("searchBtn");
    const secOverview = document.querySelector("#sec-overview .empty");
    const secSummary = document.querySelector("#sec-summary .empty");
    const secSearch = document.querySelector("#sec-search .empty");

    function resetSection(id, title, fallbackText) {
      const section = document.getElementById(id);
      section.innerHTML = `<h2>${title}</h2><div class="empty">${fallbackText}</div>`;
      return section.querySelector(".empty");
    }

    async function send() {
      const url = input.value.trim();
      if (!url) return;

      let o = resetSection("sec-overview", "1. 網站資訊", "分析中...");
      let s = resetSection("sec-summary", "2. 摘要", "分析中...");

      try {
        const resp = await fetch("/v1/openclaw/agent/summarize", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({url})
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || "分析失敗");

        o.textContent = [
          `網址: ${data.url || "-"}`,
          `網域: ${data.domain || "-"}`,
          `標題: ${data.title || "-"}`,
          `摘要來源: ${data.summary_source || "-"}`
        ].join("\\n");
        s.textContent = data.summary || "無摘要";
      } catch (err) {
        o.textContent = "分析失敗";
        s.textContent = String(err && err.message ? err.message : "連線失敗，請稍後再試。");
      }
    }

    async function searchKeyword() {
      const keyword = keywordInput.value.trim();
      if (!keyword) return;
      let rs = resetSection("sec-search", "3. 搜尋結果與來源", "搜尋中...");
      searchBtn.disabled = true;
      searchBtn.textContent = "搜尋中...";
      try {
        const resp = await fetch("/v1/openclaw/search", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({keyword, max_results: 5})
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || "搜尋失敗");

        const block = [];
        block.push(`關鍵字: ${data.keyword}`);
        block.push(`摘要來源: ${data.summary_source}`);
        block.push(`實際擷取網站數: ${data.crawled_count ?? 0}`);
        block.push(`摘要: ${data.summary}`);
        block.push("");
        block.push("來源清單:");
        (data.results || []).forEach((item, idx) => {
          block.push(`${idx + 1}. ${item.title}`);
          block.push(`   - URL: ${item.source_url}`);
          if (item.snippet) block.push(`   - 片段: ${item.snippet}`);
        });
        rs.textContent = block.join("\\n");
      } catch (err) {
        rs.textContent = String(err && err.message ? err.message : "搜尋失敗");
      } finally {
        searchBtn.disabled = false;
        searchBtn.textContent = "關鍵字搜尋";
      }
    }

    btn.addEventListener("click", send);
    searchBtn.addEventListener("click", searchKeyword);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") send();
    });
    keywordInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") searchKeyword();
    });
  </script>
</body>
</html>"""


@app.post("/v1/openclaw/agent/summarize", response_model=SummarizeResponse)
@app.post("/v1/agent/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest) -> SummarizeResponse:
    url = str(req.url)
    domain = urlparse(url).netloc
    provider = os.getenv("LLM_PROVIDER", "mock").strip().lower()
    mode = _get_summary_mode()

    # direct_url: 直接把 URL 交給 LLM 分析
    if mode in {"direct_url", "hybrid"} and provider in {"openai", "google"}:
        # OpenAI API 不保證可直接讀取外部網頁內容；避免產生無依據輸出
        if provider == "openai" and mode == "direct_url":
            raise HTTPException(
                status_code=400,
                detail="openai + direct_url 容易產生未經驗證摘要，請改用 SUMMARY_MODE=extract 或 hybrid",
            )
        try:
            if provider == "openai":
                direct = _summarize_url_direct_with_openai(url)
                source = "openai_direct_url"
            else:
                direct = _summarize_url_direct_with_google(url)
                source = "google_ai_studio_direct_url"

            title = direct.get("title")
            summary = direct.get("summary") or "未取得摘要"
            key_features = [str(x) for x in direct.get("key_features", []) if str(x).strip()]
            evidence = [str(x) for x in direct.get("evidence", []) if str(x).strip()]
            if not key_features:
                key_features = ["資訊展示與內容導覽"]
            if not evidence:
                evidence = ["由 LLM 直接根據網址推斷，未提供明確證據明細"]

            return SummarizeResponse(
                agent="OpenClaw-Sim-Agent",
                url=req.url,
                domain=domain,
                title=title,
                summary=summary,
                summary_source=source,
                key_features=key_features,
                evidence=evidence,
            )
        except Exception as exc:
            if mode == "direct_url":
                raise HTTPException(status_code=502, detail=f"direct_url 模式失敗: {exc}") from exc
            # hybrid 失敗則繼續走 extract

    try:
        html = _fetch_html(url)
        content = _extract_content(html, req.max_chars)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"網址讀取失敗: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"分析失敗: {exc}") from exc

    features, evidence = _infer_features(
        content["title"], content["meta_desc"], content["headings"], content["text"]
    )
    try:
        summary, summary_source = _summarize_with_provider(content, features)
    except Exception as exc:
        if provider != "mock" and not _allow_llm_fallback():
            raise HTTPException(status_code=502, detail=f"LLM 呼叫失敗: {exc}") from exc
        summary = _make_summary(
            content["title"], content["meta_desc"], content["headings"], features, content["text"]
        )
        summary_source = "heuristic_fallback"
        evidence.append(f"LLM 呼叫失敗，改用規則摘要: {exc}")

    return SummarizeResponse(
        agent="OpenClaw-Sim-Agent",
        url=req.url,
        domain=domain,
        title=content["title"],
        summary=summary,
        summary_source=summary_source,
        key_features=features,
        evidence=evidence,
    )


@app.post("/v1/openclaw/search", response_model=KeywordSearchResponse)
@app.post("/v1/search", response_model=KeywordSearchResponse)
def keyword_search(req: KeywordSearchRequest) -> KeywordSearchResponse:
    keyword = req.keyword.strip()
    try:
        raw_items, search_provider = _search_web(keyword, req.max_results)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"搜尋引擎請求失敗: {exc}") from exc

    results = [SearchItem(**item) for item in raw_items]
    if not results:
        return KeywordSearchResponse(
            keyword=keyword,
            summary=f"找不到與「{keyword}」相關的搜尋結果。",
            summary_source=f"heuristic_{search_provider}",
            crawled_count=0,
            results=[],
        )

    crawled = _crawl_search_results(raw_items, max_sites=min(3, len(raw_items)))
    try:
        summary, summary_source = _summarize_search_with_provider(keyword, crawled, raw_items)
    except Exception as exc:
        if os.getenv("LLM_PROVIDER", "mock").strip().lower() != "mock" and not _allow_llm_fallback():
            raise HTTPException(status_code=502, detail=f"搜尋摘要 LLM 呼叫失敗: {exc}") from exc
        summary = _summarize_search_from_crawled(keyword, crawled) if crawled else _make_search_summary(keyword, raw_items)
        summary_source = "heuristic_fallback"

    return KeywordSearchResponse(
        keyword=keyword,
        summary=summary,
        summary_source=f"{summary_source}_{search_provider}",
        crawled_count=len(crawled),
        results=results,
    )


@app.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    user_msg = req.message.strip()
    reply = _build_reply_from_url(user_msg, req.max_chars)
    return ChatResponse(user=user_msg, reply=reply)
