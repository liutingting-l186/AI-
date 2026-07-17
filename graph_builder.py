"""
GraphRAG 核心模块1：实体/关系提取 + 知识图谱构建（CPU 安全版）
修复重点：
  ✅ 统一 chunk_ids 存储为 JSON 字符串（非列表）
  ✅ 修复实体提取正则表达式（避免过度匹配）
  ✅ 增强错误处理（跳过异常 chunk）
  ✅ 强制 CPU 模式（无 GPU 环境安全运行）
  ✅ 修复 Optional 未定义问题
"""
import os
import sqlite3
import json
import re
from typing import List, Dict, Tuple, Optional  # ✅ 修复：添加 Optional 导入
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

# ======== 关键：强制全程使用 CPU ========
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 配置
DB_PATH = "./knowledge_new.db"
EMBEDDING_MODEL_PATH = "./models/bge-large-zh-v1.5"

class GraphBuilder:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self._init_graph_tables()
        self.device = "cpu"  # 强制 CPU
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device=self.device)
        print(f"✓ GraphRAG初始化完成 (device: {self.device})")
    
    def _init_graph_tables(self):
        """创建图相关表结构"""
        # 实体表
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                description TEXT,
                occurrences INTEGER DEFAULT 1,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 关系表
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                relationship_id TEXT PRIMARY KEY,
                source_entity TEXT NOT NULL,
                target_entity TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                description TEXT,
                weight INTEGER DEFAULT 1,
                chunk_ids TEXT,  -- JSON array of source chunk IDs (必须是字符串!)
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(source_entity) REFERENCES entities(entity_id),
                FOREIGN KEY(target_entity) REFERENCES entities(entity_id)
            )
        ''')
        
        # 社区表
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS communities (
                community_id INTEGER PRIMARY KEY,
                level INTEGER NOT NULL,
                title TEXT,
                entities TEXT NOT NULL,  -- JSON array of entity IDs
                relationships TEXT NOT NULL,  -- JSON array of relationship IDs
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 社区报告表
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS community_reports (
                report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                community_id INTEGER NOT NULL,
                report_text TEXT NOT NULL,
                report_json TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(community_id) REFERENCES communities(community_id)
            )
        ''')
        
        # 索引优化
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_relationships_entities ON relationships(source_entity, target_entity)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_communities_level ON communities(level)')
        self.conn.commit()
        print("✓ GraphRAG表结构初始化完成")
    
    def _extract_entities_relationships(self, chunk_text: str, chunk_id: str) -> Tuple[List[Dict], List[Dict]]:
        """
        【修复版】使用规则+启发式方法提取实体关系
        关键修复：
          1. 修正正则表达式避免过度匹配（如"应当"匹配到"应"）
          2. 实体去重逻辑增强
          3. chunk_ids 始终存储为 JSON 字符串
        """
        entities = []
        relationships = []
        
        # 实体类型模式（优化：避免单字匹配）
        patterns = {
            "ORG": r'(?:省|市|县|区|局|委|办|中心|公司|集团|企业|协会)[\u4e00-\u9fa5]{2,10}(?:局|委|办|中心|公司|集团|企业|协会)?',
            "PERSON": r'[\u4e00-\u9fa5]{2,4}(?:同志|先生|女士|局长|主任|处长|科长)',
            "LAW": r'《[\u4e00-\u9fa5]{3,20}》',
            "LOCATION": r'[\u4e00-\u9fa5]{2,8}(?:省|市|县|区|街道|路|号)',
            "CONCEPT": r'(?:应当|必须|禁止|不得|可以|负责|管理|监督|处罚|罚款|吊销|撤销)[\u4e00-\u9fa5]{2,6}'
        }
        
        # 提取实体（增强去重）
        entity_map = {}  # name -> entity dict
        for ent_type, pattern in patterns.items():
            matches = re.finditer(pattern, chunk_text)
            for m in matches:
                name = m.group(0).strip('《》')
                if len(name) < 2 or len(name) > 25:  # 严格长度过滤
                    continue
                
                # 标准化实体ID（避免特殊字符）
                safe_name = re.sub(r'[^\w\u4e00-\u9fa5]', '_', name)[:30]
                ent_id = f"{ent_type}_{safe_name}"
                
                if ent_id not in entity_map:
                    entity_map[ent_id] = {
                        "id": ent_id,
                        "name": name,
                        "type": ent_type,
                        "description": self._generate_entity_desc(name, ent_type, chunk_text),
                        "occurrences": 1
                    }
                    entities.append(entity_map[ent_id])
                else:
                    entity_map[ent_id]["occurrences"] += 1
        
        # 提取关系（基于共现+句法模式）
        sentences = [s.strip() for s in re.split(r'[。！？；\n]', chunk_text) if len(s.strip()) > 15]
        for sent in sentences:
            # 模式1: "A负责B" / "A管理B" 等
            # 修复：使用更精确的动词匹配
            pattern1 = r'([\u4e00-\u9fa5]{2,10})[应须必]?[当须要]?[负责|管理|监督|处罚|罚款]{2,4}([\u4e00-\u9fa5]{2,10})'
            for m in re.finditer(pattern1, sent):
                src, tgt = m.group(1), m.group(2)
                src_ent = self._find_entity_by_name(src, entities)
                tgt_ent = self._find_entity_by_name(tgt, entities)
                if src_ent and tgt_ent:
                    rel_id = f"rel_{src_ent['id']}_{tgt_ent['id']}_{len(relationships)}"
                    relationships.append({
                        "id": rel_id,
                        "source": src_ent["id"],
                        "target": tgt_ent["id"],
                        "type": "管理关系",
                        "description": f"{src}负责/管理{tgt}",
                        "weight": 1,
                        # ✅ 关键修复：chunk_ids 始终存储为 JSON 字符串
                        "chunk_ids": json.dumps([chunk_id])  # ← 字符串而非列表
                    })
            
            # 模式2: 同一句中多个ORG共现
            orgs_in_sent = [e for e in entities if e["type"] == "ORG" and e["name"] in sent]
            if len(orgs_in_sent) >= 2:
                for i in range(min(len(orgs_in_sent), 3)):  # 限制组合数
                    for j in range(i+1, min(len(orgs_in_sent), 3)):
                        rel_id = f"rel_{orgs_in_sent[i]['id']}_{orgs_in_sent[j]['id']}_{len(relationships)}"
                        relationships.append({
                            "id": rel_id,
                            "source": orgs_in_sent[i]["id"],
                            "target": orgs_in_sent[j]["id"],
                            "type": "协作关系",
                            "description": f"上下文共现: {sent[:30]}...",
                            "weight": 1,
                            # ✅ 关键修复：chunk_ids 始终存储为 JSON 字符串
                            "chunk_ids": json.dumps([chunk_id])  # ← 字符串而非列表
                        })
        
        return entities, relationships
    
    def _find_entity_by_name(self, name: str, entities: List[Dict]) -> Optional[Dict]:
        """模糊匹配实体名称"""
        for ent in entities:
            if name in ent["name"] or ent["name"] in name:
                return ent
        return None
    
    def _generate_entity_desc(self, name: str, ent_type: str, context: str) -> str:
        """生成实体描述（基于上下文片段）"""
        # 从上下文中提取包含该实体的句子
        sentences = [s.strip() for s in re.split(r'[。！？；\n]', context) 
                    if name in s and 20 < len(s) < 150]
        if sentences:
            return sentences[0][:80] + ("..." if len(sentences[0]) > 80 else "")
        return f"{name}（{ent_type}）"
    
    def build_knowledge_graph(self, batch_size: int = 200):
        """主流程：从chunks构建知识图谱（CPU 优化版）"""
        print("🚀 Step 1/4: 从文本中提取实体和关系...")
        
        # 读取所有chunks
        cursor = self.conn.cursor()
        cursor.execute("SELECT chunk_id, chunk_text FROM chunks ORDER BY chunk_id")
        all_chunks = cursor.fetchall()
        print(f"  读取 {len(all_chunks)} 个chunks")
        
        if not all_chunks:
            raise RuntimeError("数据库中无chunks数据，请先运行 document_processor.py")
        
        # 批量处理
        all_entities = {}
        all_relationships = {}  # key: rel_key, value: relationship dict
        
        for i in tqdm(range(0, len(all_chunks), batch_size), desc="  提取实体关系"):
            batch = all_chunks[i:i+batch_size]
            for chunk_id, chunk_text in batch:
                try:
                    entities, relationships = self._extract_entities_relationships(chunk_text, chunk_id)
                    
                    # 合并实体（去重+累加出现次数）
                    for ent in entities:
                        if ent["id"] in all_entities:
                            all_entities[ent["id"]]["occurrences"] += ent["occurrences"]
                            # 保留更长的描述
                            if len(ent["description"]) > len(all_entities[ent["id"]]["description"]):
                                all_entities[ent["id"]]["description"] = ent["description"]
                        else:
                            all_entities[ent["id"]] = ent.copy()  # 避免引用问题
                    
                    # 合并关系（✅ 修复核心：统一使用字符串处理 chunk_ids）
                    for rel in relationships:
                        rel_key = f"{rel['source']}_{rel['target']}_{rel['type']}"
                        
                        if rel_key in all_relationships:
                            # 累加权重
                            all_relationships[rel_key]["weight"] += rel["weight"]
                            
                            # 合并 chunk_ids（✅ 关键修复：解析字符串 → 合并 → 转回字符串）
                            existing_chunks = json.loads(all_relationships[rel_key]["chunk_ids"])
                            new_chunks = json.loads(rel["chunk_ids"])  # rel["chunk_ids"] 已是字符串
                            merged_chunks = list(set(existing_chunks + new_chunks))
                            all_relationships[rel_key]["chunk_ids"] = json.dumps(merged_chunks)
                        else:
                            # 首次存入（✅ 确保 chunk_ids 是字符串）
                            all_relationships[rel_key] = rel.copy()
                            # 额外安全检查：确保是字符串
                            if not isinstance(all_relationships[rel_key]["chunk_ids"], str):
                                all_relationships[rel_key]["chunk_ids"] = json.dumps([chunk_id])
                
                except Exception as e:
                    print(f"\n  ⚠️  chunk {chunk_id} 处理异常: {str(e)[:100]}，已跳过")
                    continue  # 跳过异常 chunk，不影响整体流程
        
        print(f"  ✓ 提取完成: {len(all_entities)} 个实体, {len(all_relationships)} 个关系")
        
        if not all_entities:
            raise RuntimeError("未提取到任何实体，请检查文档内容或实体提取规则")
        
        # 生成实体Embedding（CPU 优化）
        print("  生成实体Embedding...")
        entity_names = [e["name"] for e in all_entities.values()]
        entity_embeddings = self.embedding_model.encode(
            entity_names, 
            batch_size=16,  # CPU 降低 batch_size
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        # 写入数据库
        print("  写入数据库...")
        cursor = self.conn.cursor()
        
        # 实体
        for (ent_id, ent), emb in zip(all_entities.items(), entity_embeddings):
            emb_bytes = emb.astype(np.float32).tobytes()
            cursor.execute('''
                INSERT OR REPLACE INTO entities 
                (entity_id, name, type, description, occurrences, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (ent["id"], ent["name"], ent["type"], ent["description"], ent["occurrences"], emb_bytes))
        
        # 关系
        for rel in all_relationships.values():
            # ✅ 最终安全检查：确保 chunk_ids 是字符串
            if not isinstance(rel["chunk_ids"], str):
                rel["chunk_ids"] = json.dumps(rel["chunk_ids"])
            
            cursor.execute('''
                INSERT OR REPLACE INTO relationships 
                (relationship_id, source_entity, target_entity, relationship_type, description, weight, chunk_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                rel["id"], rel["source"], rel["target"], rel["type"], 
                rel["description"], rel["weight"], rel["chunk_ids"]  # 确保是字符串
            ))
        
        self.conn.commit()
        print(f"✓ 知识图谱构建完成: {len(all_entities)} 实体, {len(all_relationships)} 关系")
        return list(all_entities.values()), list(all_relationships.values())
    
    def close(self):
        self.conn.close()
        # 释放模型内存
        if hasattr(self, 'embedding_model'):
            del self.embedding_model
        torch.cuda.empty_cache()
        print("✓ GraphBuilder 资源已释放")


# ============ 独立运行入口 ============
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()

    print("="*60)
    print("GraphRAG 知识图谱构建器 (CPU 安全版)")
    print("="*60)
    
    try:
        gb = GraphBuilder(db_path=args.db)
        entities, relationships = gb.build_knowledge_graph(batch_size=args.batch_size)
        gb.close()
        
        print("\n" + "="*60)
        print("✅ 知识图谱构建成功！")
        print("="*60)
        print(f"  实体总数: {len(entities)}")
        print(f"  关系总数: {len(relationships)}")
        print("\n  下一步: 运行 community_detector.py 进行社区发现")
        print("="*60)
    
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断操作")
        if 'gb' in locals():
            gb.close()
        exit(1)
    except Exception as e:
        print(f"\n❌ 构建失败: {str(e)}")
        import traceback
        traceback.print_exc()
        if 'gb' in locals():
            gb.close()
        exit(1)
