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
    return """
    <html>
      <head><meta charset="utf-8"><title>GraphRAG API</title></head>
      <body style="font-family: Arial, sans-serif; line-height: 1.5">
        <h2>GraphRAG API 已启动</h2>
        <ul>
          <li><a href="/docs">Swagger 文档 /docs</a></li>
          <li><a href="/openapi.json">OpenAPI /openapi.json</a></li>
          <li><a href="/health">健康检查 /health</a></li>
        </ul>
      </body>
    </html>
    """


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
