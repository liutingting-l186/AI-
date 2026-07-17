"""
GraphRAG 社区网络图可视化
功能：
  • 绘制完整知识图谱（实体+关系）
  • 按 Level 0 社区自动着色
  • 节点大小 = 出现次数，边宽 = 关系权重
  • 交互式(Plotly) + 静态(Matplotlib) 双模式输出
"""
import os
import sys
import sqlite3
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path

# 尝试导入 Plotly（交互式可视化，非必需）
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ 未安装 plotly，将仅生成静态PNG图 (pip install plotly)")

class CommunityGraphVisualizer:
    def __init__(self, db_path: str = "./knowledge.db", output_dir: str = "./visualizations"):
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"数据库不存在: {self.db_path}")
        
        # 读取数据
        self.entities = self._load_entities()
        self.relationships = self._load_relationships()
        self.communities = self._load_communities()
        
        if not self.entities:
            raise RuntimeError("数据库中无实体数据，请先运行 graph_builder.py")
        if not self.communities:
            raise RuntimeError("数据库中无社区数据，请先运行 community_detector.py")
        
        print(f"✓ 加载完成: {len(self.entities)} 实体, {len(self.relationships)} 关系, {len(self.communities)} 社区")
    
    def _load_entities(self) -> dict:
        """加载实体 {entity_id: {name, type, occurrences, ...}}"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT entity_id, name, type, occurrences FROM entities ORDER BY occurrences DESC")
        entities = {
            row[0]: {
                "name": row[1],
                "type": row[2],
                "occurrences": row[3],
                "community": None  # 后续填充
            }
            for row in cursor.fetchall()
        }
        conn.close()
        return entities
    
    def _load_relationships(self) -> list:
        """加载关系列表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT source_entity, target_entity, relationship_type, weight FROM relationships")
        relationships = [
            {
                "source": row[0],
                "target": row[1],
                "type": row[2],
                "weight": max(1, row[3])  # 确保最小权重为1
            }
            for row in cursor.fetchall()
            if row[0] in self.entities and row[1] in self.entities  # 过滤无效关系
        ]
        conn.close()
        return relationships
    
    def _load_communities(self) -> dict:
        """加载 Level 0 社区 {community_id: [entity_ids]}"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT community_id, entities FROM communities WHERE level=0")
        communities = {}
        for cid, ents_str in cursor.fetchall():
            try:
                entities = json.loads(ents_str)
                communities[cid] = entities
                # 为实体标记所属社区
                for ent_id in entities:
                    if ent_id in self.entities:
                        self.entities[ent_id]["community"] = cid
            except:
                continue
        conn.close()
        return communities
    
    def build_networkx_graph(self) -> nx.Graph:
        """构建 NetworkX 图"""
        G = nx.Graph()
        
        # 添加节点（仅包含有社区归属的实体）
        for ent_id, ent in self.entities.items():
            if ent["community"] is not None:  # 只添加已分配社区的实体
                G.add_node(
                    ent_id,
                    name=ent["name"],
                    type=ent["type"],
                    community=ent["community"],
                    size=ent["occurrences"]  # 用于可视化大小
                )
        
        # 添加边
        for rel in self.relationships:
            src, tgt = rel["source"], rel["target"]
            if src in G.nodes and tgt in G.nodes:
                G.add_edge(
                    src, tgt,
                    type=rel["type"],
                    weight=rel["weight"]
                )
        
        print(f"✓ 构建图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        return G
    
    def generate_color_map(self, num_communities: int) -> dict:
        """生成社区颜色映射"""
        # 使用 Tableau 20 配色方案（美观且区分度高）
        tableau_colors = [
            '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
            '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF',
            '#AEC7E8', '#FFBB78', '#98DF8A', '#FF9896', '#C5B0D5',
            '#C49C94', '#F7B6D2', '#C7C7C7', '#DBDB8D', '#9EDAE5'
        ]
        
        # 循环使用配色
        colors = [tableau_colors[i % len(tableau_colors)] for i in range(num_communities)]
        community_ids = sorted(self.communities.keys())
        return {cid: colors[i] for i, cid in enumerate(community_ids)}
    
    def plot_matplotlib(self, G: nx.Graph, color_map: dict, filename: str = "community_graph.png"):
        """使用 Matplotlib 生成静态高清图"""
        print("\n🎨 生成静态网络图 (Matplotlib)...")
        
        # 布局算法（尝试多种，选择效果最好的）
        # 优先使用 Kamada-Kawai（适合中小规模图）
        try:
            pos = nx.kamada_kawai_layout(G, scale=2)
        except:
            # 备选：Fruchterman-Reingold
            pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        
        # 创建图形
        plt.figure(figsize=(20, 16), dpi=150)
        # 设置字体为黑体，以支持中文显示
        plt.rcParams['font.family'] = 'SimHei'
        # 设置负号显示
        plt.rcParams['axes.unicode_minus'] = False
        # 准备节点属性
        node_colors = [color_map.get(G.nodes[n]['community'], '#CCCCCC') for n in G.nodes()]
        node_sizes = [G.nodes[n]['size'] * 20 + 50 for n in G.nodes()]  # 基础大小50，按出现次数缩放
        
        # 绘制边（浅灰色，宽度按权重）
        edge_weights = [G[u][v]['weight'] * 0.8 for u, v in G.edges()]
        nx.draw_networkx_edges(
            G, pos,
            width=edge_weights,
            alpha=0.4,
            edge_color='gray'
        )
        
        # 绘制节点
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors='black',
            linewidths=0.5
        )
        
        # 绘制标签（仅显示高频实体）
        labels = {
            n: G.nodes[n]['name'] 
            for n in G.nodes() 
            if G.nodes[n]['size'] >= 3  # 只显示出现≥3次的实体
        }
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=8,
            font_family='SimHei',  # 中文支持
            font_weight='bold'
        )
        
        # 图例：社区颜色说明
        legend_elements = []
        for cid, color in sorted(color_map.items()):
            if any(G.nodes[n]['community'] == cid for n in G.nodes()):
                sample_ent = next((G.nodes[n]['name'] for n in G.nodes() if G.nodes[n]['community'] == cid), "实体")
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                 markerfacecolor=color, markersize=10, 
                                                 label=f'社区 #{cid}: {sample_ent}...'))
        
        plt.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)
        plt.title('GraphRAG 知识图谱 - 社区分布可视化', fontsize=16, pad=20)
        plt.axis('off')
        plt.tight_layout()
        
        # 保存
        output_path = self.output_dir / filename
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"✓ 静态图已保存: {output_path.absolute()}")
        return output_path
    
    def plot_plotly(self, G: nx.Graph, color_map: dict, filename: str = "community_graph_interactive.html"):
        """使用 Plotly 生成交互式HTML图（可缩放/悬停）"""
        if not PLOTLY_AVAILABLE:
            print("⚠️  Plotly 未安装，跳过交互式图生成")
            return None
        
        print("\n🌐 生成交互式网络图 (Plotly)...")
        
        # 布局
        pos = nx.spring_layout(G, k=1.2, iterations=50, seed=42)
        
        # 边坐标
        edge_x = []
        edge_y = []
        edge_weights = []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(G[u][v]['weight'])
        
        # 边迹
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            opacity=0.4
        )
        
        # 节点坐标
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            ent = G.nodes[node]
            node_text.append(
                f"<b>{ent['name']}</b><br>"
                f"类型: {ent['type']}<br>"
                f"出现: {ent['size']} 次<br>"
                f"社区: #{ent['community']}"
            )
            node_color.append(color_map.get(ent['community'], '#CCCCCC'))
            node_size.append(ent['size'] * 3 + 10)  # 基础大小10
        
        # 节点迹
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[G.nodes[n]['name'] for n in G.nodes()],
            textposition='top center',
            textfont=dict(size=8, color='black'),
            marker=dict(
                showscale=False,
                color=node_color,
                size=node_size,
                line_width=1.5,
                line_color='black',
                opacity=0.95
            )
        )
        node_trace.hovertext = node_text
        
        # 创建图形
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='<b>GraphRAG 知识图谱 - 社区分布 (交互式)</b>',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           width=1200,
                           height=900,
                           plot_bgcolor='white'
                       ))
        
        # 保存
        output_path = self.output_dir / filename
        pio.write_html(fig, file=output_path, auto_open=False)
        
        print(f"✓ 交互式图已保存: {output_path.absolute()}")
        print("   💡 提示: 用浏览器打开HTML文件可交互缩放/悬停查看实体详情")
        return output_path
    
    def visualize(self, output_prefix: str = "graphrag_community"):
        """主可视化流程"""
        print("\n" + "="*70)
        print("📊 GraphRAG 社区网络图可视化")
        print("="*70)
        
        # 构建图
        G = self.build_networkx_graph()
        
        # 生成颜色映射
        color_map = self.generate_color_map(len(self.communities))
        
        # 生成静态图
        png_path = self.plot_matplotlib(G, color_map, f"{output_prefix}.png")
        
        # 生成交互式图（如果可用）
        html_path = None
        if PLOTLY_AVAILABLE:
            html_path = self.plot_plotly(G, color_map, f"{output_prefix}_interactive.html")
        
        # 生成统计报告
        self._generate_stats_report(G, color_map, png_path, html_path)
        
        print("\n" + "="*70)
        print("✅ 可视化完成！")
        print("="*70)
        print(f"   静态图: {png_path}")
        if html_path:
            print(f"   交互图: {html_path}")
        print(f"   输出目录: {self.output_dir.absolute()}")
        print("="*70)
    
    def _generate_stats_report(self, G: nx.Graph, color_map: dict, png_path: Path, html_path: Path):
        """生成可视化统计报告"""
        report_path = self.output_dir / "visualization_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("GraphRAG 社区网络图 - 可视化统计报告\n")
            f.write("="*70 + "\n\n")
            
            # 基础统计
            f.write("【图谱规模】\n")
            f.write(f"  • 节点数: {G.number_of_nodes()}\n")
            f.write(f"  • 边数:   {G.number_of_edges()}\n")
            f.write(f"  • 社区数: {len(self.communities)} (Level 0)\n")
            f.write(f"  • 密度:   {nx.density(G):.4f}\n\n")
            
            # 社区分布
            f.write("【社区分布】\n")
            community_sizes = {}
            for cid in self.communities:
                size = sum(1 for n in G.nodes() if G.nodes[n].get('community') == cid)
                if size > 0:
                    community_sizes[cid] = size
            
            for cid, size in sorted(community_sizes.items(), key=lambda x: x[1], reverse=True):
                color = color_map.get(cid, '#CCCCCC')
                f.write(f"  • 社区 #{cid:2d} | 节点: {size:3d} | 颜色: {color}\n")
            f.write("\n")
            
            # 实体类型分布
            f.write("【实体类型分布】\n")
            type_counts = {}
            for n in G.nodes():
                ent_type = G.nodes[n]['type']
                type_counts[ent_type] = type_counts.get(ent_type, 0) + 1
            
            for ent_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  • {ent_type:10s}: {count:3d} 个实体\n")
            f.write("\n")
            
            # 输出文件
            f.write("【输出文件】\n")
            f.write(f"  • 静态图: {png_path.name}\n")
            if html_path:
                f.write(f"  • 交互图: {html_path.name}\n")
            f.write(f"  • 本报告: {report_path.name}\n")
            f.write("\n" + "="*70)
        
        print(f"\n📄 统计报告已生成: {report_path.name}")


# ============ 独立运行入口 ============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GraphRAG 社区网络图可视化")
    parser.add_argument("--db", default="./knowledge.db", help="数据库路径")
    parser.add_argument("--output", default="./visualizations", help="输出目录")
    parser.add_argument("--prefix", default="graphrag_community", help="输出文件前缀")
    args = parser.parse_args()
    
    print("="*70)
    print("GraphRAG 社区网络图可视化工具")
    print("="*70)
    print(f"  数据库: {Path(args.db).absolute()}")
    print(f"  输出目录: {Path(args.output).absolute()}")
    print("="*70)
    
    try:
        # 检查依赖
        if not PLOTLY_AVAILABLE:
            print("\n⚠️  提示: 安装 plotly 可获得交互式可视化体验")
            print("   pip install plotly")
        
        # 执行可视化
        viz = CommunityGraphVisualizer(db_path=args.db, output_dir=args.output)
        viz.visualize(output_prefix=args.prefix)
        
        print("\n💡 使用建议:")
        print("   • 静态图 (PNG): 用于论文/报告插图")
        print("   • 交互图 (HTML): 用浏览器打开，支持缩放/悬停查看实体详情")
        print("   • 统计报告: 查看 community_distribution.txt 了解社区结构")
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 可视化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)