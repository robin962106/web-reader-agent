# web_reader_agent

使用 FastAPI 製作的「模擬 OpenClaw AI Agent」服務。使用者提交網址後，API 會自動抓取頁面內容並回傳網站功能摘要。

## 安裝

```bash
cd /home/shin/web_reader/web_reader_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 啟動 API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

啟動後直接開瀏覽器：

- `http://127.0.0.1:8000`

你會看到簡單聊天頁，僅顯示「使用者」與「回覆」。

## 串接 LLM（OpenAI / Google AI Studio）

預設使用 `mock`（規則摘要，不呼叫外部 LLM）。

建議在專案目錄建立 `.env`：

```bash
cd /home/shin/web_reader/web_reader_agent
cat > .env <<'EOF'
# mock / openai / google
LLM_PROVIDER=openai

# OpenAI
OPENAI_API_KEY=你的OPENAI_API_KEY
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1/responses

# Google AI Studio (改用 Google 時再打開)
# LLM_PROVIDER=google
# GOOGLE_API_KEY=你的GOOGLE_API_KEY
# GOOGLE_MODEL=gemini-1.5-flash

# 搜尋引擎（關鍵字搜尋）
SEARCH_PROVIDER=duckduckgo
GOOGLE_CSE_API_KEY=你的_GOOGLE_CUSTOM_SEARCH_API_KEY
GOOGLE_CSE_CX=你的_GOOGLE_CUSTOM_SEARCH_ENGINE_ID
EOF
```

程式啟動時會自動讀取 `.env`，不需要手動 `export`。

## 摘要模式（重點）

可透過 `SUMMARY_MODE` 控制摘要流程：

- `extract`：先抓 HTML 內容再給 LLM（預設，最穩定）
- `direct_url`：直接把網址交給 LLM 分析（你要的模式）
- `hybrid`：先 `direct_url`，失敗後改走 `extract`

範例：

```env
SUMMARY_MODE=direct_url
```

建議穩定設定：

```env
SUMMARY_MODE=extract
ALLOW_LLM_FALLBACK=false
```

說明：
- `ALLOW_LLM_FALLBACK=false` 時，LLM 失敗會直接報錯，不會偷偷改用規則摘要。
- `openai + direct_url` 會被阻擋，避免輸出未經網頁內容驗證的摘要。

你也可以繼續用 `export` 方式：

```bash
# 1) OpenAI
export LLM_PROVIDER=openai
export OPENAI_API_KEY="你的OPENAI_API_KEY"
export OPENAI_MODEL="gpt-4o-mini"   # 可改

# 2) Google AI Studio (Gemini)
# export LLM_PROVIDER=google
# export GOOGLE_API_KEY="你的GOOGLE_API_KEY"
# export GOOGLE_MODEL="gemini-1.5-flash"   # 可改
```

程式內已註解 API Key 位置，檔案：
- `app.py` 的 `_summarize_with_openai()`
- `app.py` 的 `_summarize_with_google()`

## API 端點

- `GET /`（聊天網頁）
- `GET /health`
- `POST /v1/chat`（聊天 API，輸入網址字串）
- `POST /v1/openclaw/search`
- `POST /v1/search`（別名）
- `POST /v1/openclaw/agent/summarize`
- `POST /v1/agent/summarize` (別名)

## Request 範例

```bash
curl -X POST http://127.0.0.1:8000/v1/openclaw/agent/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "max_chars": 6000
  }'
```

```bash
curl -X POST http://127.0.0.1:8000/v1/openclaw/search \
  -H "Content-Type: application/json" \
  -d '{
    "keyword": "台灣 半導體 供應鏈",
    "max_results": 5
  }'
```

說明：關鍵字搜尋會先向搜尋引擎取回多筆結果，再嘗試抓取前幾個網站內文，最後彙整摘要並附上來源網址。

預設使用 DuckDuckGo。若你要用 Google 搜尋，請先確認 `GET /health`：
- `search_provider` 應為 `google_cse`
- `google_cse_key_set` 應為 `yes`
- `google_cse_cx_set` 應為 `yes`

## Response 範例

```json
{
  "agent": "OpenClaw-Sim-Agent",
  "url": "https://example.com/",
  "domain": "example.com",
  "title": "Example Domain",
  "summary": "...",
  "summary_source": "openai",
  "key_features": ["資訊展示與內容導覽"],
  "evidence": ["未偵測到明確交易或互動關鍵字，以內容展示型網站判定"]
}
```
