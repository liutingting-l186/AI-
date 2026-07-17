"""
知识库数据库动态管理工具
功能：
  1. 合并：将新数据DB安全合并到主库（自动去重+冲突检测）
  2. 删除：按文档名/领域/权限标签精准删除
  3. 审计：操作日志记录 + 数据变更统计
  4. 回滚：自动创建操作前快照（.backup）
"""
import sqlite3
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import json


class KnowledgeDBManager:
    def __init__(self, main_db: str = "./knowledge.db"):
        self.main_db = Path(main_db)
        self.backup_dir = Path("./db_backups")
        self.backup_dir.mkdir(exist_ok=True)
        
        if not self.main_db.exists():
            raise FileNotFoundError(f"主数据库不存在: {self.main_db}")
    
    def _create_backup(self, operation: str) -> Path:
        """创建操作前快照（保留最近5个备份）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{self.main_db.stem}_{operation}_{timestamp}{self.main_db.suffix}"
        backup_path = self.backup_dir / backup_name
        
        # 复制数据库文件
        shutil.copy2(self.main_db, backup_path)
        
        # 清理旧备份（保留最近5个）
        backups = sorted(self.backup_dir.glob(f"{self.main_db.stem}_{operation}_*{self.main_db.suffix}"), reverse=True)
        for old_backup in backups[5:]:
            old_backup.unlink()
            print(f"  → 清理旧备份: {old_backup.name}")
        
        print(f"  ✓ 创建备份: {backup_path.name}")
        return backup_path
    
    def _get_db_stats(self, conn: sqlite3.Connection) -> dict:
        """获取数据库统计信息"""
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), COUNT(DISTINCT doc_id), COUNT(DISTINCT business_domain) FROM chunks")
        total, docs, domains = cursor.fetchone()
        
        cursor.execute("SELECT business_domain, COUNT(*) FROM chunks GROUP BY business_domain ORDER BY COUNT(*) DESC")
        domain_stats = dict(cursor.fetchall())
        
        return {
            "total_chunks": total,
            "total_docs": docs,
            "domains": domain_stats
        }
    
    def merge(self, source_db: str, skip_duplicates: bool = True, force_update: bool = False):
        """
        合并源数据库到主库
        
        参数:
            source_db: 源数据库路径
            skip_duplicates: True=跳过重复chunk_id, False=覆盖（默认安全模式）
            force_update: 强制覆盖重复项（需显式指定）
        """
        source_path = Path(source_db)
        if not source_path.exists():
            raise FileNotFoundError(f"源数据库不存在: {source_path}")
        
        print(f"\n{'='*60}")
        print(f"📦 合并操作: {source_path.name} → {self.main_db.name}")
        print(f"{'='*60}")
        
        # 1. 创建备份
        self._create_backup("merge")
        
        # 2. 连接数据库
        main_conn = sqlite3.connect(self.main_db)
        source_conn = sqlite3.connect(source_path)
        
        try:
            # 3. 获取统计信息
            main_stats = self._get_db_stats(main_conn)
            source_stats = self._get_db_stats(source_conn)
            
            print(f"\n📊 源库统计: {source_stats['total_chunks']} chunks | {source_stats['total_docs']} docs | 领域: {list(source_stats['domains'].keys())}")
            print(f"📊 主库统计: {main_stats['total_chunks']} chunks | {main_stats['total_docs']} docs | 领域: {list(main_stats['domains'].keys())}")
            
            # 4. 检测重复项
            source_cursor = source_conn.cursor()
            source_cursor.execute("SELECT chunk_id, doc_id, doc_name FROM chunks")
            source_chunks = {row[0]: (row[1], row[2]) for row in source_cursor.fetchall()}
            
            main_cursor = main_conn.cursor()
            main_cursor.execute("SELECT chunk_id FROM chunks WHERE chunk_id IN ({})".format(','.join('?'*len(source_chunks))), list(source_chunks.keys()))
            existing_ids = set(row[0] for row in main_cursor.fetchall())
            
            duplicates = existing_ids & source_chunks.keys()
            new_chunks = len(source_chunks) - len(duplicates)
            
            print(f"\n🔍 重复检测: {len(duplicates)} 个重复chunk_id | {new_chunks} 个新chunks")
            
            if duplicates:
                print(f"⚠️  重复文档示例:")
                for i, chunk_id in enumerate(list(duplicates)[:3], 1):
                    doc_id, doc_name = source_chunks[chunk_id]
                    print(f"    [{i}] {doc_name} (doc_id: {doc_id}) - chunk_id: {chunk_id}")
                
                if not force_update and skip_duplicates:
                    print(f"\n💡 安全模式: 跳过 {len(duplicates)} 个重复项 (使用 --force-update 覆盖)")
                elif force_update:
                    print(f"\n⚠️  警告: 将覆盖 {len(duplicates)} 个重复项!")
            
            # 5. 执行合并
            source_cursor.execute("SELECT * FROM chunks")
            chunks_to_insert = []
            
            for row in source_cursor.fetchall():
                chunk_id = row[0]
                if chunk_id in duplicates:
                    if not force_update:
                        continue  # 跳过重复
                chunks_to_insert.append(row)
            
            if not chunks_to_insert:
                print("\n❌ 无新数据可合并")
                return
            
            # 使用参数化查询防止SQL注入
            placeholders = ','.join(['?']*11)
            main_cursor.executemany(
                f"INSERT {'OR REPLACE' if force_update else 'OR IGNORE'} INTO chunks VALUES ({placeholders})",
                chunks_to_insert
            )
            main_conn.commit()
            
            # 6. 验证结果
            after_stats = self._get_db_stats(main_conn)
            added = after_stats['total_chunks'] - main_stats['total_chunks']
            
            print(f"\n✅ 合并成功!")
            print(f"   新增 chunks: {added} (目标: {len(chunks_to_insert)})")
            print(f"   主库总计: {after_stats['total_chunks']} chunks | {after_stats['total_docs']} docs")
            
            # 7. 生成操作报告
            report = {
                "operation": "merge",
                "timestamp": datetime.now().isoformat(),
                "source_db": source_path.name,
                "before": main_stats,
                "after": after_stats,
                "duplicates": len(duplicates),
                "new_chunks": added,
                "force_update": force_update
            }
            self._save_audit_log(report)
            
        finally:
            main_conn.close()
            source_conn.close()
    
    def delete(self, condition: dict, dry_run: bool = True):
        """
        按条件删除chunks
        
        参数:
            condition: 删除条件字典，支持:
                - {"doc_name": "xxx.pdf"}
                - {"business_domain": "危化品运输"}
                - {"permission_tag": "internal"}
                - {"chunk_id": "doc123_45"}
            dry_run: True=仅预览不删除（默认安全模式）
        """
        print(f"\n{'='*60}")
        print(f"🗑️  删除操作: 条件 {condition}")
        print(f"{'='*60}")
        
        # 构建WHERE子句
        where_clauses = []
        params = []
        for key, value in condition.items():
            if key not in ["chunk_id", "doc_id", "doc_name", "business_domain", "permission_tag"]:
                raise ValueError(f"不支持的删除条件字段: {key}")
            where_clauses.append(f"{key} = ?")
            params.append(value)
        
        where_sql = " AND ".join(where_clauses)
        
        conn = sqlite3.connect(self.main_db)
        cursor = conn.cursor()
        
        try:
            # 1. 预览将删除的数据
            cursor.execute(f"SELECT chunk_id, doc_name, business_domain, page_num FROM chunks WHERE {where_sql}", params)
            to_delete = cursor.fetchall()
            
            if not to_delete:
                print("\n❌ 未找到匹配的数据")
                return
            
            print(f"\n🔍 匹配 {len(to_delete)} 个chunks:")
            print(f"{'Chunk ID':<20} {'文档':<30} {'领域':<15} {'页码'}")
            print("-" * 70)
            for i, (cid, doc, domain, page) in enumerate(to_delete[:10], 1):
                print(f"{cid:<20} {doc:<30} {domain:<15} {page}")
            if len(to_delete) > 10:
                print(f"... 还有 {len(to_delete)-10} 个chunks")
            
            # 2. 统计影响范围
            cursor.execute(f"SELECT COUNT(DISTINCT doc_id), COUNT(DISTINCT business_domain) FROM chunks WHERE {where_sql}", params)
            affected_docs, affected_domains = cursor.fetchone()
            print(f"\n📊 影响范围: {affected_docs} 个文档 | {affected_domains} 个业务领域")
            
            if dry_run:
                print(f"\n💡 安全模式: 以上为预览结果 (使用 --confirm 确认删除)")
                return
            
            # 3. 创建备份
            self._create_backup("delete")
            
            # 4. 执行删除
            cursor.execute(f"DELETE FROM chunks WHERE {where_sql}", params)
            conn.commit()
            
            # 5. 验证结果
            print(f"\n✅ 删除成功! 移除 {cursor.rowcount} 个chunks")
            
            # 6. 生成操作报告
            report = {
                "operation": "delete",
                "timestamp": datetime.now().isoformat(),
                "condition": condition,
                "deleted_count": cursor.rowcount,
                "affected_docs": affected_docs,
                "affected_domains": affected_domains,
                "dry_run": dry_run
            }
            self._save_audit_log(report)
            
        finally:
            conn.close()
    
    def _save_audit_log(self, report: dict):
        """保存操作审计日志"""
        log_file = self.backup_dir / "audit_log.jsonl"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(report, ensure_ascii=False) + '\n')
        print(f"\n📝 操作日志已记录: {log_file.name}")
    
    def list_documents(self):
        """列出数据库中所有文档及其统计信息"""
        conn = sqlite3.connect(self.main_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT doc_name, doc_id, business_domain, 
                   COUNT(*) as chunk_count,
                   MIN(page_num) as start_page,
                   MAX(page_num) as end_page
            FROM chunks 
            GROUP BY doc_id 
            ORDER BY business_domain, doc_name
        """)
        
        docs = cursor.fetchall()
        conn.close()
        
        print(f"\n{'='*80}")
        print(f"📚 知识库文档清单 ({len(docs)} 个文档)")
        print(f"{'='*80}")
        print(f"{'文档名':<40} {'领域':<15} {'Chunks':<8} {'页码范围'}")
        print("-" * 80)
        
        domain_totals = {}
        for doc_name, doc_id, domain, count, start, end in docs:
            domain_totals[domain] = domain_totals.get(domain, 0) + 1
            pages = f"P{start}-{end}" if start != -1 and end != -1 else "N/A"
            print(f"{doc_name[:38]:<40} {domain:<15} {count:<8} {pages}")
        
        print("-" * 80)
        print(f"📊 领域分布: " + " | ".join([f"{d}({c})" for d, c in sorted(domain_totals.items())]))
        print(f"✅ 总计: {len(docs)} 个文档 | {sum(domain_totals.values())} 个领域分类")


def main():
    parser = argparse.ArgumentParser(description="知识库数据库动态管理工具", 
                                     formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', help='操作命令')
    
    # 合并命令
    merge_parser = subparsers.add_parser('merge', help='合并新数据到主库')
    merge_parser.add_argument('--source', required=True, help='源数据库路径 (如: new_data.db)')
    merge_parser.add_argument('--force-update', action='store_true', 
                             help='强制覆盖重复chunk_id (默认跳过重复项)')
    
    # 删除命令
    delete_parser = subparsers.add_parser('delete', help='删除指定数据')
    delete_parser.add_argument('--doc-name', help='按文档名删除 (支持通配符%)')
    delete_parser.add_argument('--domain', help='按业务领域删除 (如: 危化品运输)')
    delete_parser.add_argument('--permission', help='按权限标签删除 (如: internal)')
    delete_parser.add_argument('--chunk-id', help='按chunk_id精确删除')
    delete_parser.add_argument('--confirm', action='store_true', 
                              help='确认执行删除 (默认仅预览)')
    
    # 列出文档
    subparsers.add_parser('list', help='列出所有文档及统计信息')
    
    # 审计日志
    subparsers.add_parser('audit', help='查看操作审计日志')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = KnowledgeDBManager()
    
    try:
        if args.command == 'merge':
            manager.merge(
                source_db=args.source,
                force_update=args.force_update
            )
        
        elif args.command == 'delete':
            # 构建删除条件
            condition = {}
            if args.doc_name:
                condition['doc_name'] = args.doc_name
            if args.domain:
                condition['business_domain'] = args.domain
            if args.permission:
                condition['permission_tag'] = args.permission
            if args.chunk_id:
                condition['chunk_id'] = args.chunk_id
            
            if not condition:
                print("❌ 错误: 至少指定一个删除条件 (--doc-name / --domain / --permission / --chunk-id)")
                return
            
            manager.delete(
                condition=condition,
                dry_run=not args.confirm
            )
        
        elif args.command == 'list':
            manager.list_documents()
        
        elif args.command == 'audit':
            log_file = Path("./db_backups/audit_log.jsonl")
            if not log_file.exists():
                print("❌ 无审计日志")
                return
            
            print(f"\n{'='*80}")
            print(f"📝 操作审计日志")
            print(f"{'='*80}")
            with open(log_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f.readlines()[-10:], 1):  # 显示最近10条
                    log = json.loads(line)
                    ts = log['timestamp'].split('T')[0]
                    op = log['operation'].upper()
                    details = ""
                    if op == 'MERGE':
                        details = f"源库: {log['source_db']} | 新增: {log['new_chunks']} chunks"
                    elif op == 'DELETE':
                        cond = ' AND '.join([f"{k}={v}" for k,v in log['condition'].items()])
                        details = f"条件: {cond} | 删除: {log['deleted_count']} chunks"
                    print(f"[{i}] {ts} | {op:6} | {details}")
    
    except Exception as e:
        print(f"\n❌ 操作失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()