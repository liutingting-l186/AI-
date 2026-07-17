#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
车辆常跑路线分析 - 自动定时任务版
自动处理上一个月的轨迹数据
"""

import clickhouse_connect
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from scipy.spatial import cKDTree
import os
import time
import json
import gc
import logging
from multiprocessing import Pool, current_process
from typing import List, Dict, Set
import signal
import sys

# ============ 配置 ============
CLICKHOUSE_CONFIG = {
    "host": "192.168.128.10",
    "port": 8123,
    "user": "default",
    "password": "xaxn@2021.com",
    "database": "wlpt_01"
}

TARGET_DATABASE = "luxian"
TARGET_TABLE = "vehicle_info"

# 处理配置
MAX_WORKERS = 2                  # 并行进程数
BATCH_SIZE = 50                  # 每批处理的车辆数
DB_BATCH_SIZE = 200              # 数据库批量写入大小
CHECKPOINT_INTERVAL = 5          # 每处理多少批保存一次断点

# 算法参数
THRESHOLD_METER = 500
MIN_ROUTE_COUNT = 3
MAX_TOP_ROUTES = 10
EARTH_RADIUS = 6371000

# 内存控制
MAX_POINTS_PER_VEHICLE = 50000
FORCE_GC_EVERY = 100

# 路径配置
BASE_DIR = "/opt/bigdata/luxian"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 全局变量
_global_client = None
_global_worker_id = None

def get_last_month() -> str:
    """获取上个月的年月字符串（YYYY-MM格式）"""
    today = datetime.now()
    # 获取上个月
    if today.month == 1:
        last_month = today.replace(year=today.year - 1, month=12)
    else:
        last_month = today.replace(month=today.month - 1)
    
    return last_month.strftime("%Y-%m")

def init_worker(worker_id: int):
    """子进程初始化"""
    global _global_client, _global_worker_id
    _global_worker_id = worker_id
    _global_client = clickhouse_connect.get_client(**CLICKHOUSE_CONFIG)
    logging.info(f"[Worker-{worker_id}] 初始化完成")

def setup_logging(month: str):
    """配置日志"""
    log_file = os.path.join(LOG_DIR, f"processor_{month}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(processName)s] - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def fast_distance(lon1, lat1, lon2, lat2):
    """简化Haversine距离计算"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * EARTH_RADIUS * np.arcsin(np.sqrt(a))

def trajectory_distance_fast(traj1: List[Dict], traj2: List[Dict]) -> float:
    """使用KD-Tree加速的轨迹距离计算"""
    if len(traj1) == 0 or len(traj2) == 0:
        return float('inf')
    
    def sample(traj, max_points=50):
        if len(traj) <= max_points:
            return traj
        indices = np.linspace(0, len(traj)-1, max_points, dtype=int)
        return [traj[i] for i in indices]
    
    t1 = sample(traj1, 50)
    t2 = sample(traj2, 50)
    
    coords1 = np.array([[p['lon'], p['lat']] for p in t1], dtype=np.float32)
    coords2 = np.array([[p['lon'], p['lat']] for p in t2], dtype=np.float32)
    
    tree2 = cKDTree(coords2)
    dists1, _ = tree2.query(coords1, k=1)
    mean_dist = np.mean(dists1) * 111000
    
    return mean_dist

def split_trips(day_df: pd.DataFrame, gap_minutes: int = 30) -> List[List[Dict]]:
    """按时间间隔分割行程"""
    if len(day_df) == 0:
        return []
    
    day_df = day_df.sort_values('create_dt')
    times = day_df['create_dt'].values
    gaps = np.diff(times).astype('timedelta64[s]').astype(float) / 60
    
    split_indices = np.where(gaps > gap_minutes)[0] + 1
    trips = np.split(day_df, split_indices)
    
    result = []
    for t in trips:
        if len(t) >= 5:
            trip_data = t[['lon', 'lat', 'create_dt']].to_dict('records')
            result.append(trip_data)
    
    return result

def analyze_single_vehicle(vehicleno: str, month: str) -> Dict:
    """分析单辆车的常跑路线"""
    client = _global_client
    worker_id = _global_worker_id
    
    try:
        # 查询数据 - 使用 up_date 分区键
        query = f"""
        SELECT vehicleno, vehiclecolor, lon, lat, create_dt
        FROM t_plt_vehicle_location
        WHERE toStartOfMonth(up_date) = '{month}-01'
          AND vehicleno = '{vehicleno}'
        ORDER BY create_dt ASC
        """
        
        df = client.query_df(query)
        
        if len(df) == 0:
            return {
                'vehicleno': vehicleno,
                'status': 'no_data',
                'frequent_route_count': 0,
                'via_frequent_route_count': 0,
                'route_total_count': 0
            }
        
        # 限制数据量
        if len(df) > MAX_POINTS_PER_VEHICLE:
            logging.warning(f"[Worker-{worker_id}] {vehicleno}@{month}: 数据量过大({len(df)}), 采样处理")
            keep_indices = [0] + list(range(1, len(df)-1, len(df)//MAX_POINTS_PER_VEHICLE)) + [len(df)-1]
            df = df.iloc[keep_indices].copy()
        
        vehicle_color = df['vehiclecolor'].iloc[0] if 'vehiclecolor' in df.columns else None
        
        # 按天分组合
        df['date'] = df['create_dt'].dt.date
        all_trips = []
        
        for date, day_df in df.groupby('date'):
            trips = split_trips(day_df)
            all_trips.extend(trips)
        
        del df
        
        if len(all_trips) == 0:
            return {
                'vehicleno': vehicleno,
                'vehiclecolor': vehicle_color,
                'status': 'no_valid_trips',
                'frequent_route_count': 0,
                'via_frequent_route_count': 0,
                'route_total_count': 0
            }
        
        # 路线聚类
        base_routes = [all_trips[0]]
        route_counts = {0: 1}
        
        for trip in all_trips[1:]:
            min_dist = float('inf')
            matched_route_id = None
            
            for route_id, base_trip in enumerate(base_routes):
                dist = trajectory_distance_fast(trip, base_trip)
                if dist < min_dist:
                    min_dist = dist
                    matched_route_id = route_id
            
            if min_dist < THRESHOLD_METER:
                route_counts[matched_route_id] = route_counts.get(matched_route_id, 0) + 1
            else:
                matched_route_id = len(base_routes)
                base_routes.append(trip)
                route_counts[matched_route_id] = 1
        
        # 统计结果
        route_total_count = len(route_counts)
        frequent = [(rid, cnt) for rid, cnt in route_counts.items() if cnt >= MIN_ROUTE_COUNT]
        frequent.sort(key=lambda x: x[1], reverse=True)
        top10 = frequent[:MAX_TOP_ROUTES]
        
        frequent_route_count = len(top10)
        via_frequent_route_count = sum(cnt for _, cnt in top10)
        
        del base_routes, all_trips
        
        return {
            'vehicleno': vehicleno,
            'vehiclecolor': vehicle_color,
            'status': 'success',
            'frequent_route_count': frequent_route_count,
            'via_frequent_route_count': via_frequent_route_count,
            'route_total_count': route_total_count
        }
        
    except Exception as e:
        logging.error(f"[Worker-{worker_id}] {vehicleno}@{month}: 处理失败 - {e}")
        return {
            'vehicleno': vehicleno,
            'status': f'error:{str(e)}',
            'frequent_route_count': 0,
            'via_frequent_route_count': 0,
            'route_total_count': 0
        }

def batch_insert_to_db(client, results: List[Dict], month: str):
    """批量写入数据库"""
    if not results:
        return 0
    
    values_list = []
    for result in results:
        if result['status'] not in ['success', 'no_valid_trips']:
            continue
        
        try:
            color_code = int(result.get('vehiclecolor', 0)) if result.get('vehiclecolor') else 0
        except:
            color_code = 0
        
        plate = str(result['vehicleno']).replace("'", "\\'")
        values_list.append(
            f"('{plate}', {color_code}, '', {result['frequent_route_count']}, "
            f"{result['via_frequent_route_count']}, {result['route_total_count']}, '', '{month}-01')"
        )

    if not values_list:
        return 0

    total_inserted = 0
    for i in range(0, len(values_list), DB_BATCH_SIZE):
        batch = values_list[i:i+DB_BATCH_SIZE]
        query = f"""
        INSERT INTO {TARGET_DATABASE}.{TARGET_TABLE}
        (plate_number, color_code, color_name, frequent_route_count, via_frequent_route_count,
         route_total_count, company_name, month)
        VALUES {','.join(batch)}
        """
        
        try:
            client.command(query)
            total_inserted += len(batch)
        except Exception as e:
            logging.error(f"批量写入失败: {e}")
    
    return total_inserted

def get_checkpoint_file(month: str) -> str:
    """获取指定月份的断点文件路径"""
    return os.path.join(OUTPUT_DIR, f"checkpoint_{month}.json")

def load_month_checkpoint(month: str) -> Set[str]:
    """加载指定月份的断点"""
    checkpoint_file = get_checkpoint_file(month)
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data.get('completed', []))
        except Exception as e:
            logging.error(f"加载断点失败 [{month}]: {e}")
    return set()

def save_month_checkpoint(month: str, completed: Set[str], processed_count: int):
    """保存指定月份的断点"""
    checkpoint_file = get_checkpoint_file(month)
    try:
        checkpoint = {
            'month': month,
            'completed': list(completed),
            'processed_count': processed_count,
            'timestamp': datetime.now().isoformat()
        }
        tmp_file = checkpoint_file + '.tmp'
        with open(tmp_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f)
        os.replace(tmp_file, checkpoint_file)
    except Exception as e:
        logging.error(f"保存断点失败 [{month}]: {e}")

def cleanup_month_checkpoint(month: str):
    """清理指定月份的断点文件"""
    checkpoint_file = get_checkpoint_file(month)
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

def get_vehicle_list(client, month: str) -> List[str]:
    """获取指定月份的待处理车辆列表"""
    query = f"""
    SELECT DISTINCT vehicleno
    FROM t_plt_vehicle_location
    WHERE toStartOfMonth(up_date) = '{month}-01'
    ORDER BY vehicleno
    """
    
    result = client.query(query)
    return [row[0] for row in result.result_rows]

def process_batch_wrapper(args):
    """处理一批车辆的包装函数"""
    batch_vehicles, month = args
    results = []
    
    for vehicleno in batch_vehicles:
        result = analyze_single_vehicle(vehicleno, month)
        results.append(result)
    
    return results

def process_single_month(main_client, month: str) -> Dict:
    """
    处理单个月份
    """
    logging.info("="*80)
    logging.info(f"[MONTH] 开始处理月份: {month}")
    logging.info("="*80)
    
    month_start_time = time.time()
    
    # 加载该月断点
    completed_set = load_month_checkpoint(month)
    logging.info(f"[CHECKPOINT-{month}] 已处理: {len(completed_set)} 辆车")
    
    # 获取该月车辆列表
    logging.info(f"[{month}] 获取车辆列表...")
    all_vehicles = get_vehicle_list(main_client, month)
    total_vehicles = len(all_vehicles)
    logging.info(f"[{month}] 总车辆数: {total_vehicles}")
    
    # 过滤已处理
    pending_vehicles = [v for v in all_vehicles if v not in completed_set]
    pending_count = len(pending_vehicles)
    logging.info(f"[{month}] 待处理: {pending_count} 辆车")
    
    if pending_count == 0:
        logging.info(f"[{month}] 该月份已全部完成")
        cleanup_month_checkpoint(month)
        return {
            'month': month,
            'total_vehicles': total_vehicles,
            'processed': 0,
            'success': 0,
            'skipped': total_vehicles
        }
    
    # 分批
    batches = []
    for i in range(0, pending_count, BATCH_SIZE):
        batch = pending_vehicles[i:i+BATCH_SIZE]
        batches.append((batch, month))
    
    total_batches = len(batches)
    logging.info(f"[{month}] 分成 {total_batches} 批处理")
    
    # 多进程处理
    processed_count = 0
    success_count = 0
    batch_results_buffer = []
    
    with Pool(processes=MAX_WORKERS, initializer=init_worker, 
              initargs=(0,)) as pool:
        
        for batch_idx, batch_results in enumerate(
            pool.imap_unordered(process_batch_wrapper, batches), 1
        ):
            batch_results_buffer.extend(batch_results)
            
            # 更新统计
            for result in batch_results:
                processed_count += 1
                completed_set.add(result['vehicleno'])
                
                if result['status'] == 'success':
                    success_count += 1
            
            # 批量写入数据库
            if len(batch_results_buffer) >= DB_BATCH_SIZE:
                inserted = batch_insert_to_db(main_client, batch_results_buffer, month)
                logging.info(f"[{month}-DB] 写入 {inserted} 条记录")
                batch_results_buffer.clear()
            
            # 保存断点
            if batch_idx % CHECKPOINT_INTERVAL == 0:
                save_month_checkpoint(month, completed_set, processed_count)
                progress = processed_count / pending_count * 100
                logging.info(f"[{month}-CHECKPOINT] 进度: {processed_count}/{pending_count} ({progress:.1f}%)")
            
            # 进度报告
            if batch_idx % 5 == 0 or batch_idx == total_batches:
                elapsed = time.time() - month_start_time
                speed = processed_count / elapsed if elapsed > 0 else 0
                progress = processed_count / pending_count * 100
                logging.info(f"[{month}-PROGRESS] 批次 {batch_idx}/{total_batches} | "
                           f"车辆 {processed_count}/{pending_count} ({progress:.1f}%) | "
                           f"速度: {speed:.2f}辆/s")
            
            # 定期GC
            if batch_idx % 10 == 0:
                gc.collect()
    
    # 写入剩余数据
    if batch_results_buffer:
        inserted = batch_insert_to_db(main_client, batch_results_buffer, month)
        logging.info(f"[{month}-DB] 最后写入 {inserted} 条记录")
    
    # 保存最终断点并清理
    save_month_checkpoint(month, completed_set, processed_count)
    cleanup_month_checkpoint(month)
    
    # 月份统计
    month_elapsed = time.time() - month_start_time
    logging.info("="*80)
    logging.info(f"[{month}-COMPLETE] 月份处理完成")
    logging.info(f"  总车辆: {total_vehicles}")
    logging.info(f"  已处理: {processed_count}")
    logging.info(f"  成功: {success_count}")
    logging.info(f"  失败: {processed_count - success_count}")
    logging.info(f"  耗时: {month_elapsed:.1f}s ({month_elapsed/60:.1f}min)")
    logging.info("="*80)
    
    return {
        'month': month,
        'total_vehicles': total_vehicles,
        'processed': processed_count,
        'success': success_count,
        'failed': processed_count - success_count,
        'elapsed_seconds': month_elapsed
    }

def signal_handler(sig, frame):
    """信号处理"""
    logging.info("[SIGNAL] 接收到中断信号，保存当前进度后退出...")
    sys.exit(0)

def main():
    """主流程 - 自动处理上一个月"""
    # 获取上一个月
    target_month = get_last_month()
    
    # 设置日志
    log_file = setup_logging(target_month)
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logging.info("="*80)
    logging.info(f"[START] 车辆常跑路线自动处理 - {datetime.now()}")
    logging.info(f"[CONFIG] 目标月份: {target_month}")
    logging.info(f"[CONFIG] 并行数: {MAX_WORKERS}, 批次大小: {BATCH_SIZE}")
    logging.info(f"[CONFIG] 日志文件: {log_file}")
    logging.info("="*80)
    
    # 主进程连接数据库
    main_client = clickhouse_connect.get_client(**CLICKHOUSE_CONFIG)
    
    # 处理单个月份
    overall_start_time = time.time()
    month_stats = process_single_month(main_client, target_month)
    
    # 总体统计
    overall_elapsed = time.time() - overall_start_time
    
    logging.info("="*80)
    logging.info("[FINAL REPORT] 处理完成")
    logging.info("="*80)
    logging.info(f"处理月份: {target_month}")
    logging.info(f"总车辆数: {month_stats['total_vehicles']}")
    logging.info(f"总处理数: {month_stats['processed']}")
    logging.info(f"总成功数: {month_stats['success']}")
    logging.info(f"总耗时: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f}min)")
    logging.info(f"平均速度: {month_stats['processed']/overall_elapsed:.2f}辆/s")
    logging.info("="*80)
    
    main_client.close()
    logging.info("[ALL COMPLETE] 全部完成")
    
    # 返回状态码（供定时任务判断）
    return 0 if month_stats['failed'] == 0 else 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
