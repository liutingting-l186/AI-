import threading
from typing import Any, Dict, Optional, Tuple

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import combine_rag


DEFAULT_LLM_BACKEND = "auto"
DEFAULT_API_MODEL = "qwen3.7-plus"
DEFAULT_API_BASE_URL = "https://llm-enqowi9yb0ihot6o.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
DEFAULT_DOTENV = ".env"
DEFAULT_API_KEY_NAMES = "DASHSCOPE_API_KEY,OPENAI_API_KEY,API_KEY"
DEFAULT_QA_THRESHOLD = 0.85
DEFAULT_HISTORY_ROUNDS = 3
DEFAULT_LOG_DIR = "./qa_memory"


class QueryRequest(BaseModel):
    question: str
    region: str = ""
    user_id: str = "default"
    use_graph: bool = True
    history_rounds: int = DEFAULT_HISTORY_ROUNDS
    qa_threshold: float = DEFAULT_QA_THRESHOLD
    llm_backend: str = DEFAULT_LLM_BACKEND
    api_model: str = DEFAULT_API_MODEL
    api_base_url: str = DEFAULT_API_BASE_URL
    dotenv: str = DEFAULT_DOTENV
    api_key_names: str = DEFAULT_API_KEY_NAMES
    log_dir: str = DEFAULT_LOG_DIR


class QueryResponse(BaseModel):
    answer: str
    mode: str
    answer_source: str
    confidence: float
    matched_question: str
    category: str
    elapsed_time: float
    region: str
    db_path: str
    qa_path: str


class _EngineHolder:
    def __init__(self, engine: combine_rag.HybridGraphRAGQuery):
        self.engine = engine
        self.lock = threading.Lock()


_engines: Dict[Tuple[str, str, str, str], _EngineHolder] = {}
_engines_lock = threading.Lock()


def _resolve_paths(region: str) -> Dict[str, str]:
    region_key = combine_rag._normalize_region(region)
    region_paths = combine_rag._resolve_region_paths(region_key)
    db_path = region_paths.get("db_path") or "./knowledge_new.db"
    qa_path = region_paths.get("qa_path") or "./qa_dataset.jsonl"
    qa_cache = region_paths.get("qa_cache") or "./qa_cache/qa_cache.npz"
    if region_paths.get("db_path") and not combine_rag.os.path.exists(db_path):
        db_path = "./knowledge_new.db"
    if region_paths.get("qa_path") and not combine_rag.os.path.exists(qa_path):
        qa_path = "./qa_dataset.jsonl"
    return {"region_key": region_key, "db_path": db_path, "qa_path": qa_path, "qa_cache": qa_cache}


def _get_engine(req: QueryRequest) -> Tuple[_EngineHolder, Dict[str, str]]:
    paths = _resolve_paths(req.region)
    region_key = paths["region_key"]
    engine_key = (region_key, req.llm_backend, req.api_model, req.api_base_url)

    with _engines_lock:
        holder = _engines.get(engine_key)
        if holder is None:
            api_key_names = [s.strip() for s in (req.api_key_names or "").split(",") if s.strip()]
            engine = combine_rag.HybridGraphRAGQuery(
                db_path=paths["db_path"],
                qa_path=paths["qa_path"],
                qa_threshold=req.qa_threshold,
                qa_cache_path=paths["qa_cache"],
                llm_backend=req.llm_backend,
                api_model_name=req.api_model,
                api_base_url=req.api_base_url,
                dotenv_path=req.dotenv,
                api_key_env_names=api_key_names,
                history_rounds=req.history_rounds,
                history_log_dir=req.log_dir,
                history_region=region_key,
                history_user_id=req.user_id,
            )
            holder = _EngineHolder(engine)
            _engines[engine_key] = holder
    return holder, paths


app = FastAPI(title="GraphRAG API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI 知识问答 - 真实知识库</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 20px; }
    .container { width: 100%; max-width: 600px; background: #fff; border-radius: 16px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); overflow: hidden; display: flex; flex-direction: column; height: 80vh; }
    .header { background: linear-gradient(135deg, #4A90D9 0%, #357ABD 100%); color: #fff; padding: 20px 24px; display: flex; align-items: center; gap: 12px; }
    .header-icon { width: 40px; height: 40px; background: rgba(255,255,255,0.2); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px; }
    .header-title { font-size: 18px; font-weight: 600; }
    .header-subtitle { font-size: 13px; opacity: 0.8; }
    .chat-area { flex: 1; overflow-y: auto; padding: 20px; background: #f8f9fa; }
    .message { display: flex; margin-bottom: 20px; gap: 12px; }
    .message.user { flex-direction: row-reverse; }
    .avatar { width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0; font-size: 14px; font-weight: 600; }
    .message.bot .avatar { background: linear-gradient(135deg, #4A90D9 0%, #357ABD 100%); color: #fff; }
    .message.user .avatar { background: #07C160; color: #fff; }
    .bubble { max-width: 75%; padding: 14px 18px; border-radius: 16px; font-size: 15px; line-height: 1.6; word-break: break-word; }
    .message.bot .bubble { background: #fff; color: #333; border-top-left-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
    .message.user .bubble { background: #4A90D9; color: #fff; border-top-right-radius: 4px; }
    .meta-info { font-size: 12px; color: #999; margin-top: 6px; display: flex; gap: 12px; }
    .loading { display: flex; gap: 6px; padding: 14px 18px; }
    .dot { width: 8px; height: 8px; background: #4A90D9; border-radius: 50%; animation: pulse 1.4s infinite ease-in-out; }
    .dot:nth-child(1) { animation-delay: -0.32s; }
    .dot:nth-child(2) { animation-delay: -0.16s; }
    .dot:nth-child(3) { animation-delay: 0s; }
    @keyframes pulse { 0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; } 40% { transform: scale(1); opacity: 1; } }
    .input-area { padding: 16px 20px; background: #fff; border-top: 1px solid #e5e5e5; display: flex; gap: 12px; }
    .input-box { flex: 1; padding: 12px 16px; border: 1px solid #ddd; border-radius: 24px; font-size: 15px; outline: none; transition: border-color 0.3s; }
    .input-box:focus { border-color: #4A90D9; }
    .send-btn { padding: 12px 24px; background: linear-gradient(135deg, #4A90D9 0%, #357ABD 100%); color: #fff; border: none; border-radius: 24px; font-size: 15px; cursor: pointer; transition: transform 0.2s; }
    .send-btn:hover { transform: translateY(-1px); }
    .send-btn:active { transform: translateY(0); }
    .send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .welcome { text-align: center; padding: 40px 20px; color: #666; }
    .welcome-icon { font-size: 48px; margin-bottom: 16px; }
    .toast { position: fixed; top: 20px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.8); color: #fff; padding: 12px 24px; border-radius: 8px; font-size: 14px; opacity: 0; transition: opacity 0.3s; pointer-events: none; z-index: 999; }
    .toast.show { opacity: 1; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="header-icon">🤖</div>
      <div>
        <div class="header-title">AI 知识问答</div>
        <div class="header-subtitle">真实知识库 · GraphRAG</div>
      </div>
    </div>
    <div class="chat-area" id="chatArea">
      <div class="welcome">
        <div class="welcome-icon">👋</div>
        <p>你好！我是 AI 知识助手，请向我提问。</p>
        <p style="font-size: 13px; color: #999; margin-top: 8px;">已连接真实知识库</p>
      </div>
    </div>
    <div class="input-area">
      <input type="text" class="input-box" id="inputBox" placeholder="请输入你的问题..." autocomplete="off">
      <button class="send-btn" id="sendBtn">发送</button>
    </div>
  </div>
  <div class="toast" id="toast"></div>
  <script>
    const API_BASE = '/';
    const chatArea = document.getElementById('chatArea');
    const inputBox = document.getElementById('inputBox');
    const sendBtn = document.getElementById('sendBtn');
    const toast = document.getElementById('toast');
    function showToast(msg) { toast.textContent = msg; toast.classList.add('show'); setTimeout(() => toast.classList.remove('show'), 2500); }
    function addMessage(role, content, meta) {
      const message = document.createElement('div');
      message.className = 'message ' + role;
      const avatar = document.createElement('div');
      avatar.className = 'avatar';
      avatar.textContent = role === 'user' ? '我' : 'AI';
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.textContent = content;
      const contentWrapper = document.createElement('div');
      contentWrapper.appendChild(bubble);
      if (meta) {
        const metaInfo = document.createElement('div');
        metaInfo.className = 'meta-info';
        if (meta.confidence !== undefined) metaInfo.innerHTML += '<span>置信度: ' + meta.confidence + '</span>';
        if (meta.mode) metaInfo.innerHTML += '<span>模式: ' + meta.mode + '</span>';
        if (meta.elapsed_time !== undefined) metaInfo.innerHTML += '<span>耗时: ' + meta.elapsed_time + 's</span>';
        contentWrapper.appendChild(metaInfo);
      }
      message.appendChild(avatar);
      message.appendChild(contentWrapper);
      chatArea.appendChild(message);
      chatArea.scrollTop = chatArea.scrollHeight;
    }
    function addLoading() {
      const message = document.createElement('div');
      message.className = 'message bot';
      message.id = 'loadingMsg';
      const avatar = document.createElement('div');
      avatar.className = 'avatar';
      avatar.textContent = 'AI';
      const bubble = document.createElement('div');
      bubble.className = 'bubble loading';
      bubble.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
      message.appendChild(avatar);
      message.appendChild(bubble);
      chatArea.appendChild(message);
      chatArea.scrollTop = chatArea.scrollHeight;
    }
    function removeLoading() { const loading = document.getElementById('loadingMsg'); if (loading) loading.remove(); }
    async function sendQuestion() {
      const question = inputBox.value.trim();
      if (!question) return;
      inputBox.value = '';
      sendBtn.disabled = true;
      addMessage('user', question);
      addLoading();
      try {
        const res = await fetch(API_BASE + 'query', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, user_id: 'web_user', use_graph: true }) });
        if (!res.ok) throw new Error('请求失败: ' + res.status);
        const data = await res.json();
        removeLoading();
        addMessage('bot', data.answer, { confidence: data.confidence, mode: data.mode, elapsed_time: data.elapsed_time });
      } catch (err) { removeLoading(); showToast(err.message); } finally { sendBtn.disabled = false; }
    }
    sendBtn.addEventListener('click', sendQuestion);
    inputBox.addEventListener('keydown', function(e) { if (e.key === 'Enter') sendQuestion(); });
  </script>
</body>
</html>"""


@app.post("/query", response_model=QueryResponse)
def query_api(req: QueryRequest) -> QueryResponse:
    holder, paths = _get_engine(req)
    with holder.lock:
        holder.engine.history_rounds = int(req.history_rounds)
        holder.engine.qa_threshold = float(req.qa_threshold)
        holder.engine.history_user_id = req.user_id
        result = holder.engine.query(req.question, use_graph=bool(req.use_graph))

    combine_rag.append_qa_memory(
        log_dir=req.log_dir,
        region=paths["region_key"],
        user_id=req.user_id,
        question=req.question,
        result=result,
        db_path=paths["db_path"],
        qa_path=paths["qa_path"],
        qa_threshold=float(req.qa_threshold),
    )

    return QueryResponse(
        answer=result.get("answer", ""),
        mode=result.get("mode", ""),
        answer_source=result.get("answer_source", ""),
        confidence=float(result.get("confidence", 0.0) or 0.0),
        matched_question=result.get("matched_question", "") or "",
        category=result.get("category", "") or "",
        elapsed_time=float(result.get("elapsed_time", 0.0) or 0.0),
        region=paths["region_key"],
        db_path=paths["db_path"],
        qa_path=paths["qa_path"],
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
