"""
GraphRAG 核心模块2：社区检测（Leiden算法）+ 层级化社区构建（修复版）
修复重点：
  ✅ 添加 __main__ 入口自动执行
  ✅ 增强调试输出（每步显示数据量）
  ✅ 修复层级构建逻辑（避免因实体类型缺失导致无输出）
  ✅ 添加结果验证查询
"""
import sqlite3
import networkx as nx
import leidenalg as la
import igraph as ig
import json
from typing import List, Dict

class CommunityDetector:
    def __init__(self, db_path: str = "./knowledge.db"):
        self.conn = sqlite3.connect(db_path)
        self._verify_data()
    
    def _verify_data(self):
        """验证数据库中存在实体和关系"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM entities")
        entity_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM relationships")
        rel_count = cursor.fetchone()[0]
        
        if entity_count == 0 or rel_count == 0:
            raise RuntimeError(
                f"❌ 数据库中缺少图数据！\n"
                f"   实体数: {entity_count} (需 >0)\n"
                f"   关系数: {rel_count} (需 >0)\n"
                f"   请先运行 graph_builder.py 构建知识图谱"
            )
        print(f"✓ 数据验证通过: {entity_count} 实体, {rel_count} 关系")
    
    def _build_igraph(self) -> ig.Graph:
        """从SQLite构建igraph（Leiden算法需要）"""
        cursor = self.conn.cursor()
        
        # 读取实体（节点）
        cursor.execute("SELECT entity_id FROM entities")
        nodes = [row[0] for row in cursor.fetchall()]
        if not nodes:
            raise RuntimeError("❌ 未找到任何实体节点")
        
        node_index = {node: i for i, node in enumerate(nodes)}
        print(f"  → 实体节点数: {len(nodes)}")
        
        # 读取关系（边）
        cursor.execute("SELECT source_entity, target_entity, weight FROM relationships")
        edges = []
        weights = []
        for src, tgt, w in cursor.fetchall():
            if src in node_index and tgt in node_index:
                edges.append((node_index[src], node_index[tgt]))
                weights.append(max(1, w))  # 确保权重≥1
        
        if not edges:
            raise RuntimeError("❌ 未找到有效关系边（检查实体ID映射）")
        
        print(f"  → 有效关系边: {len(edges)} (原始: {cursor.rowcount})")
        
        # 构建igraph
        g = ig.Graph(directed=False)
        g.add_vertices(len(nodes))
        g.add_edges(edges)
        g.es['weight'] = weights
        g.vs['name'] = nodes
        
        # 检查图连通性（非必需，但有助于调试）
        components = g.clusters()
        print(f"  → 图连通分量: {len(components)} 个 (最大分量: {max(len(c) for c in components)} 节点)")
        
        return g
    
    def detect_communities(self, resolution: float = 1.0) -> Dict[int, List[str]]:
        """
        Leiden社区检测（单层）
        resolution: >1 更多小社区, <1 更少大社区（中文文本建议0.8-1.2）
        """
        print(f"\n🔍 执行Leiden社区检测 (resolution={resolution})...")
        g = self._build_igraph()
        
        # Leiden算法
        partition = la.find_partition(
            g,
            la.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            weights=g.es['weight']
        )
        
        # 转换为 {community_id: [entity_ids]}
        communities = {}
        for node_idx, comm_id in enumerate(partition.membership):
            entity_id = g.vs[node_idx]['name']
            communities.setdefault(comm_id, []).append(entity_id)
        
        # 过滤过小的社区（<2个实体）
        communities = {cid: ents for cid, ents in communities.items() if len(ents) >= 2}
        
        print(f"✓ 社区检测完成: {len(communities)} 个有效社区 (过滤掉{len(partition.membership)-len(communities)}个单实体社区)")
        return communities
    
    def build_hierarchical_communities(self, max_levels: int = 2):
        """
        构建层级化社区（从细粒度到粗粒度）
        Level 0: 叶子社区（最细粒度）
        Level 1: 按实体类型聚合（简化版）
        """
        print("\n" + "="*60)
        print("🚀 Step 2/4: 社区检测与层级构建")
        print("="*60)
        
        # Level 0: 基础社区（必须生成）
        level0_communities = self.detect_communities(resolution=1.0)
        if not level0_communities:
            raise RuntimeError("❌ Leiden算法未检测到任何有效社区（尝试调整resolution参数）")
        
        # 写入数据库
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM communities")  # 清空旧数据
        
        community_id_counter = 0
        total_entities_in_communities = 0
        
        # Level 0 写入
        print(f"\n📦 写入 Level 0 社区 ({len(level0_communities)} 个)...")
        for comm_id, entities in level0_communities.items():
            # 获取该社区内的所有关系
            placeholders = ','.join('?' * len(entities))
            cursor.execute(f'''
                SELECT relationship_id FROM relationships 
                WHERE source_entity IN ({placeholders}) OR target_entity IN ({placeholders})
            ''', entities * 2)
            rels = [row[0] for row in cursor.fetchall()]
            
            cursor.execute('''
                INSERT INTO communities (community_id, level, entities, relationships)
                VALUES (?, ?, ?, ?)
            ''', (
                community_id_counter,
                0,
                json.dumps(entities, ensure_ascii=False),
                json.dumps(rels, ensure_ascii=False)
            ))
            total_entities_in_communities += len(entities)
            community_id_counter += 1
        
        print(f"  ✓ Level 0: {len(level0_communities)} 个社区 ({total_entities_in_communities} 个实体)")
        
        # Level 1: 按实体类型聚合（仅当有足够数据时）
        if max_levels > 1 and community_id_counter > 3:  # 至少4个Level 0社区才聚合
            print(f"\n📦 构建 Level 1 社区（按实体类型聚合）...")
            
            # 获取实体类型映射
            cursor.execute("SELECT entity_id, type FROM entities")
            entity_types = dict(cursor.fetchall())
            
            # 按类型分组社区
            type_groups = {}
            for comm_id in range(community_id_counter):
                cursor.execute("SELECT entities FROM communities WHERE community_id=? AND level=0", (comm_id,))
                entities = json.loads(cursor.fetchone()[0])
                
                # 统计社区主导类型
                type_counts = {}
                for ent in entities:
                    ent_type = entity_types.get(ent, "OTHER")
                    type_counts[ent_type] = type_counts.get(ent_type, 0) + 1
                
                if type_counts:
                    dominant_type = max(type_counts, key=type_counts.get)
                    type_groups.setdefault(dominant_type, []).append(comm_id)
            
            # 创建Level 1社区
            level1_count = 0
            for group_type, comm_ids in type_groups.items():
                if len(comm_ids) < 2:  # 至少2个Level 0社区才聚合
                    continue
                
                # 聚合实体和关系
                all_ents = set()
                all_rels = set()
                for cid in comm_ids:
                    cursor.execute("SELECT entities, relationships FROM communities WHERE community_id=? AND level=0", (cid,))
                    ents_str, rels_str = cursor.fetchone()
                    all_ents.update(json.loads(ents_str))
                    all_rels.update(json.loads(rels_str))
                
                if len(all_ents) < 3:  # 避免过小社区
                    continue
                
                cursor.execute('''
                    INSERT INTO communities (community_id, level, title, entities, relationships)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    community_id_counter,
                    1,
                    f"{group_type}相关聚合社区",
                    json.dumps(list(all_ents), ensure_ascii=False),
                    json.dumps(list(all_rels), ensure_ascii=False)
                ))
                community_id_counter += 1
                level1_count += 1
            
            print(f"  ✓ Level 1: {level1_count} 个聚合社区")
        else:
            print(f"\nℹ️  跳过 Level 1 构建（Level 0 社区数不足或 max_levels=1）")
        
        self.conn.commit()
        
        # 验证写入结果
        cursor.execute("SELECT COUNT(*), COUNT(DISTINCT level) FROM communities")
        total, levels = cursor.fetchone()
        cursor.execute("SELECT level, COUNT(*) FROM communities GROUP BY level ORDER BY level")
        level_stats = cursor.fetchall()
        
        print("\n" + "="*60)
        print("✅ 社区构建完成！")
        print("="*60)
        print(f"  总社区数: {total}")
        for level, count in level_stats:
            print(f"    Level {level}: {count} 个社区")
        print(f"\n💡 提示: 社区数据已写入数据库表 'communities'")
        print("="*60)
        
        return total
    
    def preview_communities(self, n: int = 5):
        """预览社区内容"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT community_id, level, entities 
            FROM communities 
            ORDER BY level, community_id 
            LIMIT ?
        """, (n,))
        
        print("\n📊 社区预览 (前5个):")
        for row in cursor.fetchall():
            cid, level, ents_str = row
            entities = json.loads(ents_str)
            sample_ents = ", ".join([e.split('_')[1] if '_' in e else e for e in entities[:3]])
            print(f"  [Level {level}] 社区#{cid}: {sample_ents} ... (+{len(entities)-3}更多)" if len(entities)>3 else f"  [Level {level}] 社区#{cid}: {sample_ents}")
    
    def close(self):
        self.conn.close()
        print("✓ CommunityDetector 资源已释放")


# ============ 独立运行入口（关键修复：添加自动执行）============
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="./knowledge_new.db")
    parser.add_argument("--max-levels", type=int, default=2)
    parser.add_argument("--preview", type=int, default=5)
    args = parser.parse_args()

    print("="*60)
    print("GraphRAG 社区检测器 (Leiden算法)")
    print("="*60)
    
    try:
        detector = CommunityDetector(args.db)
        total_communities = detector.build_hierarchical_communities(max_levels=args.max_levels)
        detector.preview_communities(args.preview)
        detector.close()
        
        print("\n" + "="*60)
        print("🎉 社区检测成功完成！")
        print("="*60)
        print("\n➡️  下一步: 运行 report_generator.py 生成社区报告")
        print("="*60)
    
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断操作")
        if 'detector' in locals():
            detector.close()
        exit(1)
    except Exception as e:
        print(f"\n❌ 社区检测失败: {str(e)}")
        import traceback
        traceback.print_exc()
        if 'detector' in locals():
            detector.close()
        exit(1)
