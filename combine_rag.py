"""
Hybrid GraphRAG 查询引擎（CPU 版）
功能：标准 QA 优先匹配 + 社区报告检索 + 文本检索 + LLM 生成答案
策略：高置信度匹配直接返回标准答案，否则走 GraphRAG 生成
"""
import os
import sqlite3
import numpy as np
import json
import time
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional

# ✅ 强制 CPU 模式（必须在导入模型前设置）
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def first_turn_filter(query: str) -> Optional[Dict[str, Any]]:
    """第一轮轻量过滤：命中则直接返回，不进入后续流程。返回 Dict 表示拦截成功，返回 None 表示放行。"""
    query_lower = (query or "").strip().lower()
    if not query_lower:
        return {
            "answer": "请详细说明你要咨询的问题，我会更准确地帮你解答~",
            "meta": {"intercepted": True, "type": "empty"},
        }

    import re
    query_clean = re.sub(r"[\s\W_]+", "", query_lower)

    greetings_exact = {"你好", "您好", "嗨", "hi", "hello", "hey", "早上好", "下午好", "晚上好", "在吗", "在么"}
    if query_clean in greetings_exact or re.fullmatch(r"(你好|您好|嗨|早上好|下午好|晚上好|在吗|在么)[啊呀哇吗嘛哦哈]*", query_clean):
        return {
            "answer": "你好！我是你的专业助手，请直接说明你要咨询的问题，我会尽力帮你~",
            "meta": {"intercepted": True, "type": "greeting"},
        }

    closings_exact = {"谢谢", "感谢", "拜拜", "再见", "ok", "好的", "明白了"}
    if query_clean in closings_exact or re.fullmatch(r"(谢谢|感谢|拜拜|再见|ok|好的|明白了)[啊呀哇吗嘛哦哈]*", query_clean):
        return {
            "answer": "不客气！如有其他问题随时问我~",
            "meta": {"intercepted": True, "type": "closing"},
        }

    abuse_patterns = [
        r"(傻逼|煞笔|sb|s[bB]|草你|操你|操尼|艹你|艹尼|你妈|你妈的|他妈|他妈的|妈的|滚|废物|垃圾|智障|脑残|去死)",
        r"(狗东西|贱人|贱货|婊子|傻子|蠢货|弱智|畜生|混蛋|王八蛋)",
        r"(删库|删数据|删除数据库|drop database|drop table|rm -rf|删除文件|破坏系统|攻击系统|黑客)",
    ]
    if any(re.search(p, query_lower) for p in abuse_patterns):
        return {
            "answer": "我理解你可能很着急，但请尽量使用礼貌用语。请直接描述你遇到的业务问题（例如：证照、许可、车辆审验、危化品运输等），我会尽力帮你解决。",
            "meta": {"intercepted": True, "type": "abuse"},
        }

    self_harm_patterns = [
        r"(自杀|轻生|想死|不想活|活不下去|结束生命|割腕|跳楼|上吊)",
    ]
    if any(re.search(p, query_lower) for p in self_harm_patterns):
        return {
            "answer": "听起来你正在经历非常困难的时刻。如果你有伤害自己的想法，请优先确保安全：立即联系身边可信的人，或拨打当地紧急电话（如 120/110），也可以尽快前往就近医院或心理健康机构寻求帮助。",
            "meta": {"intercepted": True, "type": "self_harm"},
        }

    explicit_patterns = [
        r"(色情|黄网|成人视频|A片|a片|做爱|性交|约炮|嫖娼|卖淫|成人视频|裸聊)",
    ]
    if any(re.search(p, query_lower) for p in explicit_patterns):
        return {
            "answer": "抱歉，我无法协助处理这类内容。我可以继续帮你解答车辆运输服务领域的业务问题，请描述你的具体需求。",
            "meta": {"intercepted": True, "type": "sensitive"},
        }

    irrelevant_patterns = [
        r"今天.*天气",
        r"讲个.*笑话",
        r"你.*名字",
        r"谁.*开发",
        r"唱歌",
        r"跳舞",
        r"聊天",
        r"陪我",
    ]
    if any(re.search(p, query_lower) for p in irrelevant_patterns):
        return {
            "answer": "抱歉，我主要专注在车辆运输服务领域的问题咨询。请描述你的具体问题，我会尽力提供专业帮助~",
            "meta": {"intercepted": True, "type": "out_of_scope"},
        }

    if len(query_lower.strip()) < 3 and not any(c.isalpha() or c.isdigit() for c in query_lower):
        return {
            "answer": "请详细说明你要咨询的问题，我会更准确地帮你解答~",
            "meta": {"intercepted": True, "type": "too_short"},
        }

    return None


def _load_dotenv(dotenv_path: str = ".env") -> Dict[str, str]:
    if not dotenv_path:
        return {}
    if not os.path.exists(dotenv_path):
        return {}
    values: Dict[str, str] = {}
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k:
                values[k] = v
    return values


def _get_api_key(dotenv_path: str = ".env", key_names: Optional[List[str]] = None) -> str:
    if key_names is None:
        key_names = ["DASHSCOPE_API_KEY", "OPENAI_API_KEY", "API_KEY"]
    for name in key_names:
        val = os.environ.get(name)
        if isinstance(val, str) and val.strip():
            return val.strip()
    env_map = _load_dotenv(dotenv_path)
    for name in key_names:
        val = env_map.get(name)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


# 本地 Qwen3-1.7B CPU 推理封装
class Qwen3CPUGenerator:
    def __init__(self, model_path: str = "./models/Qwen3-1.7B"):
        print("⏳ 加载 Qwen3-1.7B 模型（CPU 模式）...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print(f"Qwen3-1.7B 加载完成 (设备：{next(self.model.parameters()).device})")
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer([text], return_tensors="pt").to("cpu")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.85,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()
            return response
        except Exception as e:
            print(f"LLM 推理错误：{str(e)[:100]}")
            return ""


class DashScopeAPIGenerator:
    def __init__(
        self,
        api_key: str,
        model_name: str = "qwen3.7-plus",
        base_url: str = "https://llm-enqowi9yb0ihot6o.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        system_prompt: str = "You are a university teacher reviewing a student's thesis. Keep LaTeX formulas as-is.",
    ):
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.api_key = api_key

        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError(f"缺少依赖 openai，请先安装：pip install openai。原始错误：{e}")

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        try:
            completion = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"API 调用错误：{str(e)[:120]}")
            return ""


class HybridGraphRAGQuery:
    def __init__(self, 
                db_path: str = "./knowledge_new.db", 
                model_path: str = "./models/Qwen3-1.7B",
                qa_path: str = "./qa_dataset.jsonl",
                qa_threshold: float = 0.85,
                qa_cache_path: str = "./qa_cache/qa_cache.npz",
                llm_backend: str = "auto",
                api_model_name: str = "qwen3.7-plus",
                api_base_url: str = "https://llm-enqowi9yb0ihot6o.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
                api_key_env_names: Optional[List[str]] = None,
                dotenv_path: str = ".env",
                api_system_prompt: str = "You are a university teacher reviewing a student's thesis. Keep LaTeX formulas as-is.",
                history_rounds: int = 3,
                history_log_dir: str = "./qa_memory",
                history_region: str = "",
                history_user_id: str = "default"):
        
        self.db_path = db_path
        self.model_path = model_path
        self.qa_path = qa_path
        self.qa_threshold = qa_threshold
        self.qa_cache_path = qa_cache_path
        self.llm_backend = (llm_backend or "local").strip().lower()
        self.api_model_name = api_model_name
        self.api_base_url = api_base_url
        self.api_key_env_names = api_key_env_names
        self.dotenv_path = dotenv_path
        self.api_system_prompt = api_system_prompt
        self.history_rounds = int(history_rounds)
        self.history_log_dir = history_log_dir
        self.history_region = history_region
        self.history_user_id = history_user_id
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        
        # 标记哪些表可用
        self.tables_available = {}
        for table in ["chunks", "entities", "communities", "community_reports"]:
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
            self.tables_available[table] = bool(cursor.fetchone())
            if not self.tables_available[table]:
                if table in ["entities", "communities", "community_reports"]:
                    print(f"警告：表 '{table}' 不存在，GraphRAG 功能将降级")
                else:
                    print(f"错误：必需表 '{table}' 不存在，请先运行 document_processor.py")
        
        # 加载 Embedding 模型
        print("加载 Embedding 模型 (bge-large-zh-v1.5)...")
        self.embedding_model = SentenceTransformer("./models/bge-large-zh-v1.5", device="cpu")
        print(f"Embedding 模型加载完成 (设备：{self.embedding_model.device})")
        
        self.llm = None
        
        # 加载标准 QA 库
        self.qa_pairs = []
        self.qa_embeddings = None
        self.has_standard_qa = False
        self._init_standard_qa(qa_path)
        
        # 预加载社区报告（内部已做表检查）
        self.community_reports = []
        if self.tables_available.get("community_reports"):
            self._preload_community_reports()
        else:
            print("跳过社区报告预加载（表不存在）")
        
        print("HybridGraphRAGQuery 初始化完成")

    def _ensure_llm_loaded(self):
        if self.llm is not None:
            return
        backend = self.llm_backend
        if backend not in {"local", "api", "auto"}:
            backend = "auto"

        if backend in {"api", "auto"}:
            api_key = _get_api_key(self.dotenv_path, self.api_key_env_names)
            if api_key:
                try:
                    print(f"  🤖 调用模型 {self.api_model_name}...")
                    self.llm = DashScopeAPIGenerator(
                        api_key=api_key,
                        model_name=self.api_model_name,
                        base_url=self.api_base_url,
                        system_prompt=self.api_system_prompt,
                    )
                    return
                except Exception as e:
                    if backend == "api":
                        raise
                    print(f"API 模型初始化失败，将回退本地模型：{str(e)[:120]}")
            elif backend == "api":
                raise RuntimeError("未在环境变量或 .env 中找到 API Key（建议使用 DASHSCOPE_API_KEY）")

        self.llm = Qwen3CPUGenerator(self.model_path)

    def _try_load_qa_cache(self, qa_path: str, questions: List[str]) -> Optional[np.ndarray]:
        cache_path = (self.qa_cache_path or "").strip()
        if not cache_path:
            return None
        if not os.path.exists(cache_path):
            return None
        try:
            stat = os.stat(qa_path)
            qa_size = stat.st_size
            qa_mtime_ns = stat.st_mtime_ns
        except Exception:
            return None

        try:
            data = np.load(cache_path, allow_pickle=True)
            cached_questions = data.get("questions")
            cached_embeddings = data.get("embeddings")
            meta_raw = data.get("meta")
        except Exception:
            return None

        if cached_questions is None or cached_embeddings is None or meta_raw is None:
            return None
        try:
            if isinstance(meta_raw, np.ndarray):
                meta_json = str(meta_raw.tolist())
            else:
                meta_json = str(meta_raw)
            meta = json.loads(meta_json)
        except Exception:
            return None

        if meta.get("source_size") != qa_size or meta.get("source_mtime_ns") != qa_mtime_ns:
            return None

        try:
            cached_questions_list = cached_questions.tolist()
        except Exception:
            return None

        if len(cached_questions_list) != len(questions):
            return None
        if any(a != b for a, b in zip(cached_questions_list, questions)):
            return None

        emb = np.asarray(cached_embeddings, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[0] != len(questions):
            return None
        return emb
    
    def _init_standard_qa(self, qa_path: str):
        """加载标准问答库并预计算向量"""
        if not os.path.exists(qa_path):
            print(f"未找到标准问答库：{qa_path}，将仅使用 GraphRAG 模式")
            return
        
        try:
            print(f"加载标准问答库：{qa_path} ...")
            with open(qa_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.qa_pairs.append(json.loads(line))
            
            if not self.qa_pairs:
                print("标准问答库为空")
                return
            
            questions = [item['question'] for item in self.qa_pairs]
            cached = self._try_load_qa_cache(qa_path, questions)
            if cached is not None:
                self.qa_embeddings = cached
            else:
                print(f"正在向量化 {len(questions)} 个标准问题...")
                self.qa_embeddings = self.embedding_model.encode(
                    questions,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                # 保存缓存
                cache_path = (self.qa_cache_path or "").strip()
                if cache_path:
                    try:
                        stat = os.stat(qa_path)
                        meta = {
                            "source_size": stat.st_size,
                            "source_mtime_ns": stat.st_mtime_ns
                        }
                        # 确保缓存目录存在
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        np.savez_compressed(
                            cache_path,
                            questions=questions,
                            embeddings=self.qa_embeddings,
                            meta=json.dumps(meta)
                        )
                        print(f"标准问答库向量缓存已保存到：{cache_path}")
                    except Exception as e:
                        print(f"保存缓存失败：{e}")
            self.has_standard_qa = True
            print(f"标准问答库加载完成 (共 {len(self.qa_pairs)} 条)")
            
        except Exception as e:
            print(f"加载标准问答库失败：{e}，将仅使用 GraphRAG 模式")
    
    def _preload_community_reports(self):
        """预加载社区报告及其 Embedding"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT community_id, report_text, embedding FROM community_reports")
        rows = cursor.fetchall()
        
        self.community_reports = []
        for cid, text, emb_bytes in rows:
            if emb_bytes:
                emb = np.frombuffer(emb_bytes, dtype=np.float32)
                self.community_reports.append({
                    "community_id": cid,
                    "report_text": text,
                    "embedding": emb
                })
        
        print(f"预加载 {len(self.community_reports)} 份社区报告")
    
    def _embed_query(self, query: str) -> np.ndarray:
        """生成查询向量"""
        return self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
    
    def _match_standard_qa(self, query_vec: np.ndarray) -> Dict[str, Any]:
        """
        尝试匹配标准 QA 库
        返回：如果匹配成功返回答案 dict，否则返回 None
        """
        if not self.has_standard_qa or self.qa_embeddings is None:
            return None
        
        # 计算余弦相似度 (因为已归一化，点积=余弦相似度)
        similarities = np.dot(self.qa_embeddings, query_vec)
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        
        if best_sim >= self.qa_threshold:
            return {
                "answer": self.qa_pairs[best_idx]['answer'],
                "confidence": float(best_sim),
                "matched_question": self.qa_pairs[best_idx]['question'],
                "category": self.qa_pairs[best_idx].get('category', '未知')
            }
        return None
    
    def _retrieve_community_reports(self, query_vec: np.ndarray, top_k: int = 3) -> List[Dict]:
        """基于向量相似度检索社区报告"""
        if not self.community_reports:
            return []
        
        similarities = []
        for report in self.community_reports:
            sim = np.dot(query_vec, report["embedding"])
            similarities.append((sim, report))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in similarities[:top_k]]
    
    def _retrieve_text_chunks(self, query_vec: np.ndarray, top_k: int = 5) -> List[Dict]:
        """检索原始文本 chunks"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT chunk_id, doc_name, page_num, section, clause, chunk_text, embedding FROM chunks")
        
        similarities = []
        for cid, doc_name, page_num, section, clause, text, emb_bytes in cursor.fetchall():
            if emb_bytes:
                emb = np.frombuffer(emb_bytes, dtype=np.float32)
                sim = np.dot(query_vec, emb)
                similarities.append(
                    (
                        sim,
                        {
                            "chunk_id": cid,
                            "doc_name": doc_name,
                            "page_num": page_num,
                            "section": section,
                            "clause": clause,
                            "text": text,
                        },
                    )
                )
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in similarities[:top_k]]

    def _format_chunk_source(self, chunk: Dict[str, Any]) -> str:
        doc_name = (chunk.get("doc_name") or "").strip()
        page_num = chunk.get("page_num")
        section = (chunk.get("section") or "").strip()
        clause = (chunk.get("clause") or "").strip()

        parts = []
        if doc_name:
            parts.append(doc_name)
        if page_num is not None and str(page_num).strip():
            parts.append(f"P{page_num}")
        if section:
            parts.append(f"章节:{section}")
        if clause:
            parts.append(f"条款:{clause}")
        if not parts:
            return ""
        return "来源: " + " | ".join(parts)

    def _build_context(self, community_reports: List[Dict[str, Any]], text_chunks: List[Dict[str, Any]], max_chars: int = 12000) -> str:
        pieces: List[str] = []
        used = 0

        def _try_add(s: str) -> bool:
            nonlocal used
            if not s:
                return True
            if used + len(s) > max_chars:
                return False
            pieces.append(s)
            used += len(s)
            return True

        if community_reports:
            if not _try_add("[社区知识摘要]\n"):
                return "".join(pieces).strip()
            for i, rep in enumerate(community_reports, 1):
                text = (rep.get("report_text") or "").strip()
                if not text:
                    continue
                block = f"{i}. {text}\n\n"
                if not _try_add(block):
                    break

        if not _try_add("[相关法规条文]\n"):
            return "".join(pieces).strip()

        for i, chunk in enumerate(text_chunks, 1):
            src = self._format_chunk_source(chunk)
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            if src:
                header = f"{i}. [{src}]\n"
            else:
                header = f"{i}.\n"
            body = text + "\n\n"
            if not _try_add(header):
                break
            if not _try_add(body):
                break

        return "".join(pieces).strip()
    
    def query(self, question: str, use_graph: bool = True) -> Dict[str, Any]:
        """
        混合查询主入口
        
        逻辑：
        1. 优先匹配标准 QA 库 (高速、确定性)
        2. 若匹配度低，走 GraphRAG 流程 (泛化、生成式)
        """
        start_time = time.time()
        original_question = (question or "").strip()

        intercepted = first_turn_filter(original_question)
        if intercepted is not None:
            elapsed = time.time() - start_time
            meta = intercepted.get("meta") if isinstance(intercepted, dict) else {}
            return {
                "answer": intercepted.get("answer", "") if isinstance(intercepted, dict) else "",
                "mode": "filtered",
                "answer_source": "filter",
                "confidence": 1.0,
                "matched_question": "",
                "category": (meta.get("type", "") if isinstance(meta, dict) else ""),
                "elapsed_time": elapsed,
                "community_reports": [],
                "text_chunks": [],
            }

        recent_turns: List[Dict[str, Any]] = []
        if self.history_rounds > 0 and self.history_log_dir:
            recent_turns = _load_recent_turns(self.history_log_dir, self.history_region, self.history_user_id, self.history_rounds)

        query_vec = self._embed_query(original_question)
        
        # === 第一步：尝试匹配标准 QA ===
        standard_match = self._match_standard_qa(query_vec)
        
        if standard_match:
            # 命中标准库
            elapsed = time.time() - start_time
            print(f"⚡ 命中标准问答库 (相似度：{standard_match['confidence']:.4f})")
            return {
                "answer": standard_match['answer'],
                "mode": "standard_qa",
                "answer_source": "standard_qa",
                "confidence": standard_match['confidence'],
                "matched_question": standard_match['matched_question'],
                "category": standard_match['category'],
                "elapsed_time": elapsed,
                "community_reports": [],
                "text_chunks": []
            }
        
        def _has_coreference(q: str) -> bool:
            q = (q or "").strip()
            markers = [
                "上一轮",
                "上一次",
                "刚才",
                "刚刚",
                "上面",
                "前面",
                "那个",
                "这个",
                "那条",
                "那种",
                "你说的",
                "你刚才",
                "我刚才",
                "继续",
                "同上",
                "如上",
                "上述",
                "前述",
            ]
            return any(m in q for m in markers)

        def _history_text(turns: List[Dict[str, Any]], assistant_max_chars: int = 200) -> str:
            pieces: List[str] = []
            for t in turns:
                q = (t.get("q") or "").strip()
                a = (t.get("a") or "").strip()
                if q:
                    pieces.append(f"用户: {q}")
                if a:
                    a = " ".join(a.split())
                    if assistant_max_chars > 0 and len(a) > assistant_max_chars:
                        a = a[:assistant_max_chars].rstrip() + "..."
                    pieces.append(f"助手: {a}")
            return "\n".join(pieces).strip()

        intent_question = original_question
        if _has_coreference(original_question) and recent_turns:
            dialog = _history_text(recent_turns, assistant_max_chars=200)
            if len(dialog) > 1800:
                self._ensure_llm_loaded()
                compress_prompt = f"""请把下面的最近对话压缩成不超过800字的要点摘要，用于帮助理解用户意图。

要求：
1. 优先保留用户的需求、约束、地区/对象、关键名词和歧义点
2. 可以保留必要的助手结论，但要简短
3. 只输出摘要内容，不要输出解释

最近对话：
{dialog}
"""
                summary = self.llm.generate(compress_prompt, max_tokens=400, temperature=0.2).strip()
                if summary:
                    dialog = summary

            self._ensure_llm_loaded()
            intent_prompt = f"""你是车辆运输服务领域的专业助手。下面给出最近对话与用户当前问题。

任务：
1) 如果用户当前问题在历史中已经被明确回答过，请直接组织语言给出答案（无需检索）。
2) 否则，请输出一个“可用于检索的用户意图问题”（将指代补全成完整问句）。

输出格式（严格 JSON）：
{{"answer_if_known": "...", "intent_question": "..."}}

最近对话：
{dialog}

当前问题：
{original_question}
"""
            raw = self.llm.generate(intent_prompt, max_tokens=500, temperature=0.2).strip()
            answer_if_known = ""
            parsed_intent = ""
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    answer_if_known = (obj.get("answer_if_known") or "").strip()
                    parsed_intent = (obj.get("intent_question") or "").strip()
            except Exception:
                pass

            if answer_if_known:
                elapsed = time.time() - start_time
                return {
                    "answer": answer_if_known,
                    "mode": "multi_turn",
                    "answer_source": "llm_history",
                    "confidence": 0.0,
                    "matched_question": "",
                    "category": "",
                    "elapsed_time": elapsed,
                    "community_reports": [],
                    "text_chunks": [],
                }

            if parsed_intent:
                intent_question = parsed_intent

            query_vec = self._embed_query(intent_question)
            standard_match = self._match_standard_qa(query_vec)
            if standard_match:
                elapsed = time.time() - start_time
                print(f"⚡ 命中标准问答库 (相似度：{standard_match['confidence']:.4f})")
                return {
                    "answer": standard_match['answer'],
                    "mode": "standard_qa",
                    "answer_source": "standard_qa",
                    "confidence": standard_match['confidence'],
                    "matched_question": standard_match['matched_question'],
                    "category": standard_match['category'],
                    "elapsed_time": elapsed,
                    "community_reports": [],
                    "text_chunks": []
                }

        # === 第二步：未命中，走 GraphRAG ===
        print(f"🔄 未命中标准库，启动 GraphRAG 生成 (阈值：{self.qa_threshold})")
        
        community_reports = []
        text_chunks = []
        
        if use_graph and self.community_reports:
            rag_vec = self._embed_query(intent_question)
            community_reports = self._retrieve_community_reports(rag_vec, top_k=3)
            text_chunks = self._retrieve_text_chunks(rag_vec, top_k=5)
            mode = "graph_rag"
        else:
            rag_vec = self._embed_query(intent_question)
            text_chunks = self._retrieve_text_chunks(rag_vec, top_k=8)
            mode = "traditional_rag"
        
        context = self._build_context(community_reports, text_chunks, max_chars=12000)
        
        prompt = f"""你是一名交通法规专家，请基于以下资料回答用户问题。

资料：
{context}

用户问题：{original_question}
（已解析的检索意图：{intent_question}）

要求：
1. 仅基于上述资料回答，不编造信息
2. 如资料不足，请说明"根据现有资料无法确定"
3. 回答简洁专业，直接给出结论；不要给出思考过程。
4. 直接输出答案，不要输出思考。
5. 若用户语义不明或不完整，引导用户给出更具体的问题。
6. 在答案中尽量标注引用来源（资料中提供的来源信息）。
回答："""
        
        # 生成答案
        self._ensure_llm_loaded()
        answer = self.llm.generate(prompt, max_tokens=1000, temperature=0.2)
        elapsed = time.time() - start_time
        
        return {
            "answer": answer.strip(),
            "mode": mode,
            "answer_source": "llm_generated",
            "confidence": 0.0,  # 生成式无法提供精确置信度
            "matched_question": "",
            "category": "",
            "community_reports": community_reports,
            "text_chunks": text_chunks,
            "elapsed_time": elapsed
        }
    
    def close(self):
        self.conn.close()
        if self.llm is not None:
            if hasattr(self.llm, "model"):
                del self.llm.model
            if hasattr(self.llm, "tokenizer"):
                del self.llm.tokenizer
        if hasattr(self, 'embedding_model'):
            del self.embedding_model
        # 安全清理 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("QueryEngine 资源已释放")


def _safe_user_id(user_id: str) -> str:
    import re
    user_id = (user_id or "").strip()
    if not user_id:
        return "default"
    user_id = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", user_id)
    user_id = user_id.strip().strip(".")
    return user_id or "default"


def _normalize_region(region: str) -> str:
    region = (region or "").strip()
    if not region:
        return "default"
    if "71" in region:
        return "71"
    if "74" in region:
        return "74"
    if "陕西" in region:
        return "71"
    if "宁夏" in region:
        return "74"
    return _safe_user_id(region)


def _resolve_region_paths(region_key: str) -> Dict[str, str]:
    if region_key == "71":
        return {
            "db_path": "./knowledge_71.db",
            "qa_path": "./qa_dataset_71.jsonl",
            "qa_cache": "./qa_cache/qa_cache_71.npz",
        }
    if region_key == "74":
        return {
            "db_path": "./knowledge_74.db",
            "qa_path": "./qa_dataset_74.jsonl",
            "qa_cache": "./qa_cache/qa_cache_74.npz",
        }
    return {
        "db_path": "",
        "qa_path": "",
        "qa_cache": "",
    }

def _iter_history_files(log_dir: str, region: str, user_id: str) -> List[str]:
    root = os.path.join(log_dir, _normalize_region(region), _safe_user_id(user_id))
    if not os.path.exists(root):
        return []
    files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".jsonl")]
    files.sort()
    return files

def _parse_ts(ts: Any):
    if isinstance(ts, str) and ts.strip():
        try:
            from datetime import datetime
            return datetime.fromisoformat(ts.strip())
        except Exception:
            pass
    return None


def _load_recent_turns(log_dir: str, region: str, user_id: str, rounds: int) -> List[Dict[str, Any]]:
    files = _iter_history_files(log_dir, region, user_id)
    items: List[Dict[str, Any]] = []
    for path in reversed(files):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if (obj.get("answer_source") or "") == "filter" or (obj.get("mode") or "") == "filtered":
                        continue
                    q = obj.get("question")
                    a = obj.get("answer")
                    if not isinstance(q, str) or not q.strip():
                        continue
                    intercepted = first_turn_filter(q)
                    if intercepted is not None:
                        continue
                    items.append(
                        {
                            "q": q.strip(),
                            "a": (a or "").strip(),
                            "ts": obj.get("ts", ""),
                            "mode": obj.get("mode", ""),
                            "answer_source": obj.get("answer_source", ""),
                            "confidence": obj.get("confidence", 0.0),
                            "matched_question": obj.get("matched_question", ""),
                            "category": obj.get("category", ""),
                        }
                    )
        except Exception:
            continue
        if len(items) >= rounds * 6:
            break

    if not items:
        return []

    def _sort_key(x: Dict[str, Any]):
        dt = _parse_ts(x.get("ts"))
        return dt.timestamp() if dt else 0.0

    items.sort(key=_sort_key)
    return items[-max(0, int(rounds)):]


def _build_history_block(
    turns: List[Dict[str, Any]],
    mode: str,
    assistant_max_chars: int,
    max_chars: int,
) -> str:
    mode = (mode or "user_only").strip().lower()
    include_assistant = mode in {"user_assistant_light", "user_and_assistant", "both"}

    pieces: List[str] = []
    used = 0

    for t in turns:
        q = (t.get("q") or "").strip()
        a = (t.get("a") or "").strip()
        if q:
            part_q = f"用户: {q}\n"
            if used + len(part_q) > max_chars:
                break
            pieces.append(part_q)
            used += len(part_q)
        if include_assistant and a:
            a = " ".join(a.split())
            if assistant_max_chars > 0 and len(a) > assistant_max_chars:
                a = a[:assistant_max_chars].rstrip() + "..."
            part_a = f"助手(参考): {a}\n"
            if used + len(part_a) > max_chars:
                break
            pieces.append(part_a)
            used += len(part_a)

    return "".join(pieces).strip()


def _is_followup_question(q: str) -> bool:
    q = (q or "").strip()
    if not q:
        return False
    followup_markers = [
        "继续",
        "再说",
        "再讲",
        "补充",
        "还有",
        "那",
        "那么",
        "然后",
        "怎么办",
        "怎么处理",
        "需要哪些材料",
        "需要什么材料",
        "要哪些材料",
        "要准备什么",
        "具体怎么",
        "详细",
    ]
    return any(m in q for m in followup_markers)


def _is_meta_history_question(q: str) -> bool:
    q = (q or "").strip()
    if not q:
        return False
    meta_markers = [
        "我刚才问的什么",
        "我刚刚问的什么",
        "我刚才问了什么",
        "我刚刚问了什么",
        "我刚才说了什么",
        "我刚刚说了什么",
        "我上一个问题",
        "上一个问题",
        "上一条",
        "上一轮",
        "刚才问的是啥",
        "刚才问的啥",
        "你刚才说的什么",
        "你刚刚说的什么",
        "你刚才回答的什么",
        "你刚刚回答的什么",
        "回顾一下",
        "总结一下我们刚才",
    ]
    return any(m in q for m in meta_markers)


def _append_qa_memory(
    log_dir: str,
    region: str,
    user_id: str,
    question: str,
    result: Dict[str, Any],
    db_path: str,
    qa_path: str,
    qa_threshold: float,
):
    from datetime import datetime
    log_root = os.path.join(log_dir, _normalize_region(region), _safe_user_id(user_id))
    os.makedirs(log_root, exist_ok=True)
    day = datetime.now().strftime("%Y%m%d")
    log_path = os.path.join(log_root, f"{day}.jsonl")
    payload = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "region": _normalize_region(region),
        "user_id": _safe_user_id(user_id),
        "question": question,
        "answer": result.get("answer", ""),
        "mode": result.get("mode", ""),
        "answer_source": result.get("answer_source", ""),
        "confidence": result.get("confidence", 0.0),
        "matched_question": result.get("matched_question", ""),
        "category": result.get("category", ""),
        "elapsed_time": result.get("elapsed_time", 0.0),
        "db_path": db_path,
        "qa_path": qa_path,
        "qa_threshold": qa_threshold,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return log_path


def append_qa_memory(
    log_dir: str,
    region: str,
    user_id: str,
    question: str,
    result: Dict[str, Any],
    db_path: str,
    qa_path: str,
    qa_threshold: float,
):
    return _append_qa_memory(
        log_dir=log_dir,
        region=region,
        user_id=user_id,
        question=question,
        result=result,
        db_path=db_path,
        qa_path=qa_path,
        qa_threshold=qa_threshold,
    )


# ============ 独立测试入口 ============
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*", default=[])
    parser.add_argument("--region", default="")
    parser.add_argument("--db-path", default="./knowledge_new.db")
    parser.add_argument("--qa-path", default="./qa_dataset.jsonl")
    parser.add_argument("--qa-threshold", type=float, default=0.85)
    parser.add_argument("--qa-cache", default="./qa_cache/qa_cache.npz")
    parser.add_argument("--llm-backend", choices=["auto", "local", "api"], default="auto")
    parser.add_argument("--api-model", default="qwen3.7-plus")
    parser.add_argument("--api-base-url", default="https://llm-enqowi9yb0ihot6o.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--api-key-names", default="DASHSCOPE_API_KEY,OPENAI_API_KEY,API_KEY")
    parser.add_argument("--history-rounds", type=int, default=3)
    parser.add_argument("--user-id", default="default")
    parser.add_argument("--log-dir", default="./qa_memory")
    args = parser.parse_args()

    test_questions = [
        "请问办理危险货物车辆审验需要准备哪些资料？",
        "吴忠市交通运输行业未来发展规划是什么？",
        "你好",
    ]

    if args.question:
        test_questions = [" ".join(args.question).strip()]
    
    engine = None
    try:
        api_key_names = [s.strip() for s in (args.api_key_names or "").split(",") if s.strip()]

        region_key = _normalize_region(args.region)
        region_paths = _resolve_region_paths(region_key)
        db_path = args.db_path
        qa_path = args.qa_path
        qa_cache = args.qa_cache

        if region_paths.get("db_path") and os.path.exists(region_paths["db_path"]):
            db_path = region_paths["db_path"]
        if region_paths.get("qa_path") and os.path.exists(region_paths["qa_path"]):
            qa_path = region_paths["qa_path"]
        if region_paths.get("qa_cache"):
            qa_cache = region_paths["qa_cache"]

        engine = HybridGraphRAGQuery(
            db_path=db_path,
            qa_path=qa_path,
            qa_threshold=args.qa_threshold,
            qa_cache_path=qa_cache,
            llm_backend=args.llm_backend,
            api_model_name=args.api_model,
            api_base_url=args.api_base_url,
            dotenv_path=args.dotenv,
            api_key_env_names=api_key_names,
            history_rounds=args.history_rounds,
            history_log_dir=args.log_dir,
            history_region=region_key,
            history_user_id=args.user_id,
        )
        
        for question in test_questions:
            print(f"\n{'='*60}")
            print(f"❓ 用户问题：{question}")
            print(f"{'='*60}")
            
            result = engine.query(question, use_graph=True)
            
            print(f"\n💡 答案 ({result['mode']} 模式，耗时：{result['elapsed_time']:.2f}秒):")
            print("-" * 60)
            print(result["answer"])
            print("-" * 60)

            _append_qa_memory(
                log_dir=args.log_dir,
                region=region_key,
                user_id=args.user_id,
                question=question,
                result=result,
                db_path=db_path,
                qa_path=qa_path,
                qa_threshold=args.qa_threshold,
            )
            
            # 显示来源信息
            if result['answer_source'] == 'standard_qa':
                print(f"✅ 来源：标准问答库 (匹配度：{result['confidence']:.2%})")
                print(f"   匹配问题：{result['matched_question']}")
                print(f"   类别：{result['category']}")
            else:
                print(f"🤖 来源：AI 生成 (GraphRAG)")
                if result["community_reports"]:
                    print(f"   参考社区报告：{len(result['community_reports'])} 份")
            
    except Exception as e:
        print(f"\n查询失败：{str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if engine:
            engine.close()
