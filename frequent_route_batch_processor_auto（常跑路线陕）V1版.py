#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
车辆常跑路线分析 - 方案一：每日处理 + 月度合并

使用方式:
1. 每日处理（定时任务每天执行）:
   python frequent_route_batch_processor_auto.py --mode daily
   python frequent_route_batch_processor_auto.py --mode daily --date 2026-04-15

2. 月度合并（每月1号凌晨执行）:
   python frequent_route_batch_processor_auto.py --mode merge
   python frequent_route_batch_processor_auto.py --mode merge --month 2026-03

3. 批量处理（历史数据补录）:
   python frequent_route_batch_processor_auto.py --mode batch --start-date 2026-04-01 --end-date 2026-04-14

部署建议:
- 首次部署时，先使用 batch 模式补录当月历史数据
- 然后配置每日定时任务处理前一天数据
- 每月1号凌晨执行 merge 模式合并整月数据

特点:
- 每天处理前一天的轨迹数据，避免一次性读取整月数据导致服务器压力过大
- 每日结果保存为JSON文件，支持断点续传
- 每月1号凌晨合并整月数据，写入ClickHouse后清理中间文件
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
    "host": "192.168.110.90",
    "port": 8123,
    "user": "default",
    "password": "xaxn@2021.com",
    "database": "south"
}

# 目标数据库配置（写入结果）
TARGET_CLICKHOUSE_CONFIG = {
    "host": "192.168.110.23",
    "port": 18123,
    "user": "default",
    "password": "xaxn@2021.com",
    "database": "luxian"
}

TARGET_DATABASE = "luxian"
TARGET_TABLE = "vehicle_info"

# 处理配置
MAX_WORKERS = 4                  # 并行进程数（优化：2→4）
BATCH_SIZE = 50                  # 每批处理的车辆数
DB_BATCH_SIZE = 200              # 数据库批量写入大小
CHECKPOINT_INTERVAL = 5          # 每处理多少批保存一次断点
QUERY_BATCH_SIZE = 20            # 批量查询车辆数（新增）

# 算法参数
THRESHOLD_METER = 500
MIN_ROUTE_COUNT = 3
MAX_TOP_ROUTES = 10
EARTH_RADIUS = 6371000

# 内存控制
MAX_POINTS_PER_VEHICLE = 50000
FORCE_GC_EVERY = 100

# 路径配置
BASE_DIR = "/opt/java/luxian"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")
DAILY_DIR = os.path.join(BASE_DIR, "daily")  # 每日中间结果目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DAILY_DIR, exist_ok=True)

# 全局变量
_global_client = None
_global_worker_id = None

# 运行模式
MODE_DAILY = "daily"           # 每日处理模式
MODE_MONTHLY_MERGE = "merge"   # 月度合并模式

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
    
    coords1 = np.array([[p['longitude'], p['latitude']] for p in t1], dtype=np.float32)
    coords2 = np.array([[p['longitude'], p['latitude']] for p in t2], dtype=np.float32)
    
    tree2 = cKDTree(coords2)
    dists1, _ = tree2.query(coords1, k=1)
    mean_dist = np.mean(dists1) * 111000
    
    return mean_dist

def split_trips(day_df: pd.DataFrame, gap_minutes: int = 30) -> List[List[Dict]]:
    """按时间间隔分割行程"""
    if len(day_df) == 0:
        return []
    
    day_df = day_df.sort_values('trajectory_time')
    times = day_df['trajectory_time'].values
    gaps = np.diff(times).astype('timedelta64[s]').astype(float) / 60
    
    split_indices = np.where(gaps > gap_minutes)[0] + 1
    trips = np.split(day_df, split_indices)
    
    result = []
    for t in trips:
        if len(t) >= 5:
            trip_data = t[['longitude', 'latitude', 'trajectory_time']].to_dict('records')
            result.append(trip_data)
    
    return result

def batch_query_vehicles(client, vehicle_list: List[str], month: str) -> Dict[str, pd.DataFrame]:
    """批量查询多辆车的数据（优化：减少查询次数）"""
    if not vehicle_list:
        return {}
    
    vehicle_str = "','".join(vehicle_list)
    query = f"""
    SELECT vehicle_num, vehicle_color_code, longitude, latitude, trajectory_time
    FROM trajectory_history_wl_809
    WHERE trajectory_time >= toDate('{month}-01')
      AND trajectory_time < toDate('{month}-01') + INTERVAL 1 MONTH
      AND vehicle_num IN ('{vehicle_str}')
      AND vehicle_num LIKE '%陕%'
    ORDER BY vehicle_num, trajectory_time ASC
    """
    
    df = client.query_df(query)
    
    if len(df) == 0:
        return {}
    
    result = {}
    for vehicle_num, group_df in df.groupby('vehicle_num'):
        result[vehicle_num] = group_df.copy()
    
    return result

def analyze_single_vehicle(vehicle_num: str, month: str, df: pd.DataFrame = None) -> Dict:
    """分析单辆车的常跑路线"""
    client = _global_client
    worker_id = _global_worker_id
    
    try:
        # 如果传入了预处理的数据，直接使用；否则查询数据库
        if df is None:
            query = f"""
            SELECT vehicle_num, vehicle_color_code, longitude, latitude, trajectory_time
            FROM trajectory_history_wl_809
            WHERE trajectory_time >= toDate('{month}-01')
              AND trajectory_time < toDate('{month}-01') + INTERVAL 1 MONTH
              AND vehicle_num = '{vehicle_num}'
              AND vehicle_num LIKE '%陕%'
            ORDER BY trajectory_time ASC
            """
            df = client.query_df(query)
        
        if df is None or len(df) == 0:
            return {
                'vehicle_num': vehicle_num,
                'status': 'no_data',
                'frequent_route_count': 0,
                'via_frequent_route_count': 0,
                'route_total_count': 0
            }
        
        # 限制数据量
        if len(df) > MAX_POINTS_PER_VEHICLE:
            logging.warning(f"[Worker-{worker_id}] {vehicle_num}@{month}: 数据量过大({len(df)}), 采样处理")
            keep_indices = [0] + list(range(1, len(df)-1, len(df)//MAX_POINTS_PER_VEHICLE)) + [len(df)-1]
            df = df.iloc[keep_indices].copy()
        
        vehicle_color = df['vehicle_color_code'].iloc[0] if 'vehicle_color_code' in df.columns else None
        
        # 按天分组合
        df['date'] = df['trajectory_time'].dt.date
        all_trips = []
        
        for date, day_df in df.groupby('date'):
            trips = split_trips(day_df)
            all_trips.extend(trips)
        
        del df
        
        if len(all_trips) == 0:
            return {
                'vehicle_num': vehicle_num,
                'vehicle_color_code': vehicle_color,
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
            'vehicle_num': vehicle_num,
            'vehicle_color_code': vehicle_color,
            'status': 'success',
            'frequent_route_count': frequent_route_count,
            'via_frequent_route_count': via_frequent_route_count,
            'route_total_count': route_total_count
        }
        
    except Exception as e:
        logging.error(f"[Worker-{worker_id}] {vehicle_num}@{month}: 处理失败 - {e}")
        return {
            'vehicle_num': vehicle_num,
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
            color_code = int(result.get('vehicle_color_code', 0)) if result.get('vehicle_color_code') else 0
        except:
            color_code = 0
        
        plate = str(result['vehicle_num']).replace("'", "\\'")
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


# ============ 每日处理相关函数 ============

def get_daily_dir(date_str: str) -> str:
    """获取指定日期的每日数据目录"""
    daily_path = os.path.join(DAILY_DIR, date_str)
    os.makedirs(daily_path, exist_ok=True)
    return daily_path

def get_daily_checkpoint_file(date_str: str) -> str:
    """获取每日断点文件路径"""
    daily_path = get_daily_dir(date_str)
    return os.path.join(daily_path, "checkpoint.json")

def load_daily_checkpoint(date_str: str) -> Set[str]:
    """加载指定日期的断点"""
    checkpoint_file = get_daily_checkpoint_file(date_str)
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data.get('completed', []))
        except Exception as e:
            logging.error(f"加载每日断点失败 [{date_str}]: {e}")
    return set()

def save_daily_checkpoint(date_str: str, completed: Set[str], processed_count: int):
    """保存指定日期的断点"""
    checkpoint_file = get_daily_checkpoint_file(date_str)
    try:
        checkpoint = {
            'date': date_str,
            'completed': list(completed),
            'processed_count': processed_count,
            'timestamp': datetime.now().isoformat()
        }
        tmp_file = checkpoint_file + '.tmp'
        with open(tmp_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f)
        os.replace(tmp_file, checkpoint_file)
    except Exception as e:
        logging.error(f"保存每日断点失败 [{date_str}]: {e}")

def get_daily_result_file(date_str: str) -> str:
    """获取每日结果文件路径"""
    daily_path = get_daily_dir(date_str)
    return os.path.join(daily_path, "vehicle_routes.json")

def convert_to_serializable(obj):
    """将numpy/pandas类型转换为Python原生类型，用于JSON序列化"""
    if hasattr(obj, 'item'):  # numpy scalar (int16, int32, float32, etc.)
        return obj.item()
    elif hasattr(obj, 'isoformat'):  # datetime, Timestamp
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj

def save_daily_results(date_str: str, vehicle_results: Dict):
    """保存每日处理结果到JSON文件"""
    result_file = get_daily_result_file(date_str)
    try:
        # 转换numpy类型为Python原生类型
        serializable_results = convert_to_serializable(vehicle_results)
        data = {
            'date': date_str,
            'generated_at': datetime.now().isoformat(),
            'vehicles': serializable_results
        }
        tmp_file = result_file + '.tmp'
        with open(tmp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, result_file)
        logging.info(f"[{date_str}] 每日结果已保存: {result_file}")
        return True
    except Exception as e:
        logging.error(f"保存每日结果失败 [{date_str}]: {e}")
        return False

def load_daily_results(date_str: str) -> Dict:
    """加载指定日期的处理结果"""
    result_file = get_daily_result_file(date_str)
    if not os.path.exists(result_file):
        return {}
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('vehicles', {})
    except Exception as e:
        logging.error(f"加载每日结果失败 [{date_str}]: {e}")
        return {}

def get_vehicle_list_by_date(client, date_str: str) -> List[str]:
    """获取指定日期的待处理车辆列表"""
    query = f"""
    SELECT DISTINCT vehicle_num
    FROM trajectory_history_wl_809
    WHERE toDate(trajectory_time) = toDate('{date_str}')
      AND vehicle_num LIKE '%陕%'
    ORDER BY vehicle_num
    """
    
    result = client.query(query)
    return [row[0] for row in result.result_rows]

def batch_query_vehicles_by_date(client, vehicle_list: List[str], date_str: str) -> Dict[str, pd.DataFrame]:
    """批量查询指定日期的多辆车数据"""
    if not vehicle_list:
        return {}
    
    vehicle_str = "','".join(vehicle_list)
    query = f"""
    SELECT vehicle_num, vehicle_color_code, longitude, latitude, trajectory_time
    FROM trajectory_history_wl_809
    WHERE toDate(trajectory_time) = toDate('{date_str}')
      AND vehicle_num IN ('{vehicle_str}')
      AND vehicle_num LIKE '%陕%'
    ORDER BY vehicle_num, trajectory_time ASC
    """
    
    df = client.query_df(query)
    
    if len(df) == 0:
        return {}
    
    result = {}
    for vehicle_num, group_df in df.groupby('vehicle_num'):
        result[vehicle_num] = group_df.copy()
    
    return result

def analyze_single_vehicle_daily(vehicle_num: str, date_str: str, df: pd.DataFrame = None) -> Dict:
    """分析单辆车单日的常跑路线（返回详细路线信息用于月度合并）"""
    client = _global_client
    worker_id = _global_worker_id
    
    try:
        # 如果传入了预处理的数据，直接使用；否则查询数据库
        if df is None:
            query = f"""
            SELECT vehicle_num, vehicle_color_code, longitude, latitude, trajectory_time
            FROM trajectory_history_wl_809
            WHERE toDate(trajectory_time) = toDate('{date_str}')
              AND vehicle_num = '{vehicle_num}'
              AND vehicle_num LIKE '%陕%'
            ORDER BY trajectory_time ASC
            """
            df = client.query_df(query)
        
        if df is None or len(df) == 0:
            return {
                'vehicle_num': vehicle_num,
                'status': 'no_data',
                'vehicle_color_code': None,
                'routes': [],
                'frequent_route_count': 0,
                'via_frequent_route_count': 0,
                'route_total_count': 0
            }
        
        # 限制数据量
        if len(df) > MAX_POINTS_PER_VEHICLE:
            logging.warning(f"[Worker-{worker_id}] {vehicle_num}@{date_str}: 数据量过大({len(df)}), 采样处理")
            keep_indices = [0] + list(range(1, len(df)-1, len(df)//MAX_POINTS_PER_VEHICLE)) + [len(df)-1]
            df = df.iloc[keep_indices].copy()
        
        vehicle_color = df['vehicle_color_code'].iloc[0] if 'vehicle_color_code' in df.columns else None
        
        # 按天分组合（单日数据，实际上只有一天）
        df['date'] = df['trajectory_time'].dt.date
        all_trips = []
        
        for date, day_df in df.groupby('date'):
            trips = split_trips(day_df)
            all_trips.extend(trips)
        
        del df
        
        if len(all_trips) == 0:
            return {
                'vehicle_num': vehicle_num,
                'vehicle_color_code': vehicle_color,
                'status': 'no_valid_trips',
                'routes': [],
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
        
        # 保存路线详情用于月度合并
        routes_detail = []
        for rid, cnt in top10:
            routes_detail.append({
                'route_id': rid,
                'count': cnt,
                'points': base_routes[rid][:10]  # 只保存前10个点作为特征
            })
        
        del base_routes, all_trips
        
        return {
            'vehicle_num': vehicle_num,
            'vehicle_color_code': vehicle_color,
            'status': 'success',
            'routes': routes_detail,
            'frequent_route_count': frequent_route_count,
            'via_frequent_route_count': via_frequent_route_count,
            'route_total_count': route_total_count
        }
        
    except Exception as e:
        logging.error(f"[Worker-{worker_id}] {vehicle_num}@{date_str}: 处理失败 - {e}")
        return {
            'vehicle_num': vehicle_num,
            'vehicle_color_code': None,
            'status': f'error:{str(e)}',
            'routes': [],
            'frequent_route_count': 0,
            'via_frequent_route_count': 0,
            'route_total_count': 0
        }

def process_daily_batch_wrapper(args):
    """处理每日一批车辆的包装函数"""
    batch_vehicles, date_str = args
    results = []
    
    # 批量查询数据
    client = _global_client
    vehicle_data = batch_query_vehicles_by_date(client, batch_vehicles, date_str)
    
    # 逐车处理
    for vehicle_num in batch_vehicles:
        df = vehicle_data.get(vehicle_num)
        result = analyze_single_vehicle_daily(vehicle_num, date_str, df)
        results.append(result)
    
    return results

def process_single_day(main_client, date_str: str) -> Dict:
    """处理单日数据"""
    logging.info("="*80)
    logging.info(f"[DAY] 开始处理日期: {date_str}")
    logging.info("="*80)
    
    day_start_time = time.time()
    
    # 加载该日断点
    completed_set = load_daily_checkpoint(date_str)
    logging.info(f"[CHECKPOINT-{date_str}] 已处理: {len(completed_set)} 辆车")
    
    # 获取该日车辆列表
    logging.info(f"[{date_str}] 获取车辆列表...")
    all_vehicles = get_vehicle_list_by_date(main_client, date_str)
    total_vehicles = len(all_vehicles)
    logging.info(f"[{date_str}] 总车辆数: {total_vehicles}")
    
    # 过滤已处理
    pending_vehicles = [v for v in all_vehicles if v not in completed_set]
    pending_count = len(pending_vehicles)
    logging.info(f"[{date_str}] 待处理: {pending_count} 辆车")
    
    if pending_count == 0:
        logging.info(f"[{date_str}] 该日期已全部完成")
        return {
            'date': date_str,
            'total_vehicles': total_vehicles,
            'processed': 0,
            'success': 0,
            'failed': 0,
            'skipped': total_vehicles,
            'elapsed_seconds': 0
        }
    
    # 分批
    batches = []
    for i in range(0, pending_count, BATCH_SIZE):
        batch = pending_vehicles[i:i+BATCH_SIZE]
        batches.append((batch, date_str))
    
    total_batches = len(batches)
    logging.info(f"[{date_str}] 分成 {total_batches} 批处理")
    
    # 多进程处理
    processed_count = 0
    success_count = 0
    vehicle_results = {}
    
    with Pool(processes=MAX_WORKERS, initializer=init_worker, 
              initargs=(0,)) as pool:
        
        for batch_idx, batch_results in enumerate(
            pool.imap_unordered(process_daily_batch_wrapper, batches), 1
        ):
            # 更新统计和结果
            for result in batch_results:
                processed_count += 1
                completed_set.add(result['vehicle_num'])
                
                # 保存成功的结果
                if result['status'] in ['success', 'no_valid_trips']:
                    success_count += 1
                    vehicle_results[result['vehicle_num']] = {
                        'vehicle_color_code': result.get('vehicle_color_code'),
                        'routes': result.get('routes', []),
                        'frequent_route_count': result['frequent_route_count'],
                        'via_frequent_route_count': result['via_frequent_route_count'],
                        'route_total_count': result['route_total_count']
                    }
            
            # 保存断点
            if batch_idx % CHECKPOINT_INTERVAL == 0:
                save_daily_checkpoint(date_str, completed_set, processed_count)
                progress = processed_count / pending_count * 100
                logging.info(f"[{date_str}-CHECKPOINT] 进度: {processed_count}/{pending_count} ({progress:.1f}%)")
            
            # 进度报告
            if batch_idx % 5 == 0 or batch_idx == total_batches:
                elapsed = time.time() - day_start_time
                speed = processed_count / elapsed if elapsed > 0 else 0
                progress = processed_count / pending_count * 100
                logging.info(f"[{date_str}-PROGRESS] 批次 {batch_idx}/{total_batches} | "
                           f"车辆 {processed_count}/{pending_count} ({progress:.1f}%) | "
                           f"速度: {speed:.2f}辆/s")
            
            # 定期GC
            if batch_idx % 10 == 0:
                gc.collect()
    
    # 保存每日结果到JSON
    if vehicle_results:
        save_daily_results(date_str, vehicle_results)
    
    # 保存最终断点
    save_daily_checkpoint(date_str, completed_set, processed_count)
    
    # 日期统计
    day_elapsed = time.time() - day_start_time
    logging.info("="*80)
    logging.info(f"[{date_str}-COMPLETE] 日期处理完成")
    logging.info(f"  总车辆: {total_vehicles}")
    logging.info(f"  已处理: {processed_count}")
    logging.info(f"  成功: {success_count}")
    logging.info(f"  失败: {processed_count - success_count}")
    logging.info(f"  耗时: {day_elapsed:.1f}s ({day_elapsed/60:.1f}min)")
    logging.info("="*80)
    
    return {
        'date': date_str,
        'total_vehicles': total_vehicles,
        'processed': processed_count,
        'success': success_count,
        'failed': processed_count - success_count,
        'elapsed_seconds': day_elapsed
    }


def get_vehicle_list(client, month: str) -> List[str]:
    """获取指定月份的待处理车辆列表"""
    query = f"""
    SELECT DISTINCT vehicle_num
    FROM trajectory_history_wl_809
    WHERE trajectory_time >= toDate('{month}-01')
      AND trajectory_time < toDate('{month}-01') + INTERVAL 1 MONTH
      AND vehicle_num LIKE '%陕%'
    ORDER BY vehicle_num
    """
    
    result = client.query(query)
    return [row[0] for row in result.result_rows]

def process_batch_wrapper(args):
    """处理一批车辆的包装函数（优化：批量查询）"""
    batch_vehicles, month = args
    results = []
    
    # 批量查询数据
    client = _global_client
    vehicle_data = batch_query_vehicles(client, batch_vehicles, month)
    
    # 逐车处理
    for vehicle_num in batch_vehicles:
        df = vehicle_data.get(vehicle_num)
        result = analyze_single_vehicle(vehicle_num, month, df)
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
            'failed': 0,
            'skipped': total_vehicles,
            'elapsed_seconds': 0
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
                completed_set.add(result['vehicle_num'])
                
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


# ============ 月度合并相关函数 ============

def get_dates_in_month(month: str) -> List[str]:
    """获取指定月份的所有日期列表"""
    dates = []
    year, mon = map(int, month.split('-'))
    
    # 获取该月第一天
    first_day = datetime(year, mon, 1)
    
    # 获取下月第一天
    if mon == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, mon + 1, 1)
    
    # 生成所有日期
    current = first_day
    while current < next_month:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    return dates

def merge_monthly_data(month: str) -> Dict[str, Dict]:
    """合并整月的每日数据"""
    logging.info(f"[{month}] 开始合并月度数据...")
    
    dates = get_dates_in_month(month)
    logging.info(f"[{month}] 共 {len(dates)} 天需要合并")
    
    # 存储所有车辆的月度统计
    monthly_stats = {}
    
    for date_str in dates:
        daily_results = load_daily_results(date_str)
        
        if not daily_results:
            logging.warning(f"[{date_str}] 无每日数据，跳过")
            continue
        
        logging.info(f"[{date_str}] 加载了 {len(daily_results)} 辆车的数据")
        
        for vehicle_num, daily_data in daily_results.items():
            if vehicle_num not in monthly_stats:
                monthly_stats[vehicle_num] = {
                    'vehicle_color_code': daily_data.get('vehicle_color_code'),
                    'daily_routes': {},  # 按日存储路线
                    'total_frequent_route_count': 0,
                    'total_via_frequent_route_count': 0,
                    'total_route_count': 0
                }
            
            # 累加统计
            monthly_stats[vehicle_num]['total_frequent_route_count'] += daily_data.get('frequent_route_count', 0)
            monthly_stats[vehicle_num]['total_via_frequent_route_count'] += daily_data.get('via_frequent_route_count', 0)
            monthly_stats[vehicle_num]['total_route_count'] += daily_data.get('route_total_count', 0)
            
            # 保存每日路线详情
            monthly_stats[vehicle_num]['daily_routes'][date_str] = daily_data.get('routes', [])
    
    logging.info(f"[{month}] 月度合并完成，共 {len(monthly_stats)} 辆车")
    return monthly_stats

def calculate_monthly_frequent_routes(monthly_stats: Dict) -> Dict[str, Dict]:
    """计算月度常跑路线（基于整月数据重新聚类）"""
    logging.info("开始计算月度常跑路线...")
    
    final_results = {}
    
    for vehicle_num, stats in monthly_stats.items():
        # 收集整月所有路线
        all_monthly_routes = []
        route_counts = {}
        
        for date_str, daily_routes in stats['daily_routes'].items():
            for route in daily_routes:
                all_monthly_routes.append({
                    'date': date_str,
                    'route_id': route['route_id'],
                    'count': route['count'],
                    'points': route.get('points', [])
                })
        
        if not all_monthly_routes:
            final_results[vehicle_num] = {
                'vehicle_color_code': stats['vehicle_color_code'],
                'frequent_route_count': 0,
                'via_frequent_route_count': 0,
                'route_total_count': stats['total_route_count']
            }
            continue
        
        # 基于出现次数统计（简化版：直接按出现天数统计）
        route_day_count = {}
        for route in all_monthly_routes:
            key = route['route_id']
            if key not in route_day_count:
                route_day_count[key] = {'count': 0, 'total_trips': 0}
            route_day_count[key]['count'] += 1
            route_day_count[key]['total_trips'] += route['count']
        
        # 筛选月度常跑路线（至少在3天出现）
        frequent = [(rid, data) for rid, data in route_day_count.items() if data['count'] >= MIN_ROUTE_COUNT]
        frequent.sort(key=lambda x: x[1]['count'], reverse=True)
        top10 = frequent[:MAX_TOP_ROUTES]
        
        frequent_route_count = len(top10)
        via_frequent_route_count = sum(data['total_trips'] for _, data in top10)
        
        final_results[vehicle_num] = {
            'vehicle_color_code': stats['vehicle_color_code'],
            'frequent_route_count': frequent_route_count,
            'via_frequent_route_count': via_frequent_route_count,
            'route_total_count': stats['total_route_count']
        }
    
    logging.info(f"月度常跑路线计算完成，共 {len(final_results)} 辆车")
    return final_results

def batch_insert_monthly_results(client, results: Dict[str, Dict], month: str):
    """批量写入月度结果到数据库"""
    if not results:
        logging.warning("无数据需要写入")
        return 0
    
    values_list = []
    for vehicle_num, result in results.items():
        try:
            color_code = int(result.get('vehicle_color_code', 0)) if result.get('vehicle_color_code') else 0
        except:
            color_code = 0
        
        plate = str(vehicle_num).replace("'", "\\'")
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
            logging.info(f"[DB] 已写入 {total_inserted}/{len(values_list)} 条记录")
        except Exception as e:
            logging.error(f"批量写入失败: {e}")
    
    return total_inserted

def cleanup_daily_files(month: str):
    """清理指定月份的每日中间文件"""
    dates = get_dates_in_month(month)
    cleaned_count = 0
    
    for date_str in dates:
        daily_path = os.path.join(DAILY_DIR, date_str)
        if os.path.exists(daily_path):
            try:
                # 删除结果文件
                result_file = os.path.join(daily_path, "vehicle_routes.json")
                if os.path.exists(result_file):
                    os.remove(result_file)
                    cleaned_count += 1
                
                # 可选：删除checkpoint文件（保留用于审计）
                # checkpoint_file = os.path.join(daily_path, "checkpoint.json")
                # if os.path.exists(checkpoint_file):
                #     os.remove(checkpoint_file)
                
                # 如果目录为空，删除目录
                if os.path.exists(daily_path) and not os.listdir(daily_path):
                    os.rmdir(daily_path)
                    
            except Exception as e:
                logging.error(f"清理文件失败 [{date_str}]: {e}")
    
    logging.info(f"[CLEANUP] 已清理 {cleaned_count} 个每日结果文件")
    return cleaned_count

def process_monthly_merge(main_client, month: str) -> Dict:
    """处理月度合并"""
    logging.info("="*80)
    logging.info(f"[MONTHLY-MERGE] 开始月度合并: {month}")
    logging.info("="*80)
    
    merge_start_time = time.time()
    
    # 1. 合并整月数据
    monthly_stats = merge_monthly_data(month)
    
    if not monthly_stats:
        logging.warning(f"[{month}] 无月度数据需要合并")
        return {
            'month': month,
            'total_vehicles': 0,
            'inserted': 0,
            'elapsed_seconds': 0
        }
    
    # 2. 计算月度常跑路线
    final_results = calculate_monthly_frequent_routes(monthly_stats)
    
    # 3. 写入目标数据库（使用单独的配置）
    logging.info(f"[{month}] 开始写入目标数据库 (192.168.110.23:18123)...")
    target_client = clickhouse_connect.get_client(**TARGET_CLICKHOUSE_CONFIG)
    inserted = batch_insert_monthly_results(target_client, final_results, month)
    target_client.close()
    
    # 4. 清理中间文件
    logging.info(f"[{month}] 开始清理中间文件...")
    cleanup_daily_files(month)
    
    # 统计
    merge_elapsed = time.time() - merge_start_time
    logging.info("="*80)
    logging.info(f"[MONTHLY-MERGE-COMPLETE] 月度合并完成")
    logging.info(f"  月份: {month}")
    logging.info(f"  总车辆: {len(final_results)}")
    logging.info(f"  写入记录: {inserted}")
    logging.info(f"  耗时: {merge_elapsed:.1f}s ({merge_elapsed/60:.1f}min)")
    logging.info("="*80)
    
    return {
        'month': month,
        'total_vehicles': len(final_results),
        'inserted': inserted,
        'elapsed_seconds': merge_elapsed
    }

def get_yesterday() -> str:
    """获取昨天的日期字符串（YYYY-MM-DD格式）"""
    yesterday = datetime.now() - timedelta(days=1)
    return yesterday.strftime("%Y-%m-%d")

def get_target_month_from_date(date_str: str) -> str:
    """从日期字符串获取月份"""
    return date_str[:7]

def get_date_range(start_date: str, end_date: str) -> List[str]:
    """获取日期范围内的所有日期列表"""
    dates = []
    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    while current <= end:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    return dates

def process_date_range(main_client, start_date: str, end_date: str) -> Dict:
    """处理指定日期范围内的所有数据（用于历史数据补录）"""
    dates = get_date_range(start_date, end_date)
    
    logging.info("="*80)
    logging.info(f"[BATCH] 开始批量处理日期范围: {start_date} 至 {end_date}")
    logging.info(f"[BATCH] 共 {len(dates)} 天需要处理")
    logging.info("="*80)
    
    total_stats = {
        'total_days': len(dates),
        'processed_days': 0,
        'failed_days': 0,
        'total_vehicles': 0,
        'total_processed': 0,
        'total_success': 0
    }
    
    for idx, date_str in enumerate(dates, 1):
        logging.info(f"\n[BATCH-PROGRESS] 处理第 {idx}/{len(dates)} 天: {date_str}")
        
        try:
            day_stats = process_single_day(main_client, date_str)
            
            total_stats['processed_days'] += 1
            total_stats['total_vehicles'] += day_stats['total_vehicles']
            total_stats['total_processed'] += day_stats['processed']
            total_stats['total_success'] += day_stats['success']
            
            # 每处理5天强制GC一次
            if idx % 5 == 0:
                gc.collect()
                
        except Exception as e:
            logging.error(f"[BATCH-ERROR] 处理 {date_str} 失败: {e}")
            total_stats['failed_days'] += 1
    
    logging.info("="*80)
    logging.info("[BATCH-COMPLETE] 批量处理完成")
    logging.info(f"  总天数: {total_stats['total_days']}")
    logging.info(f"  成功: {total_stats['processed_days']}")
    logging.info(f"  失败: {total_stats['failed_days']}")
    logging.info(f"  总车辆: {total_stats['total_vehicles']}")
    logging.info(f"  总处理: {total_stats['total_processed']}")
    logging.info(f"  总成功: {total_stats['total_success']}")
    logging.info("="*80)
    
    return total_stats

def main():
    """主流程 - 支持每日处理、批量处理和月度合并三种模式"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='车辆常跑路线分析')
    parser.add_argument('--mode', type=str, choices=[MODE_DAILY, MODE_MONTHLY_MERGE, 'batch'],
                        default=MODE_DAILY,
                        help=f'运行模式: {MODE_DAILY}=每日处理, {MODE_MONTHLY_MERGE}=月度合并, batch=批量处理日期范围')
    parser.add_argument('--date', type=str, default=None,
                        help='指定日期 (YYYY-MM-DD格式)，默认为昨天')
    parser.add_argument('--month', type=str, default=None,
                        help='指定月份 (YYYY-MM格式)，月度合并时使用，默认为上个月')
    parser.add_argument('--start-date', type=str, default=None,
                        help='批量处理起始日期 (YYYY-MM-DD格式)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='批量处理结束日期 (YYYY-MM-DD格式)')
    
    args = parser.parse_args()
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 主进程连接数据库
    main_client = clickhouse_connect.get_client(**CLICKHOUSE_CONFIG)
    
    try:
        if args.mode == MODE_DAILY:
            # ============ 每日处理模式 ============
            target_date = args.date or get_yesterday()
            target_month = get_target_month_from_date(target_date)
            
            # 设置日志
            log_file = setup_logging(f"daily_{target_date}")
            
            logging.info("="*80)
            logging.info(f"[START] 车辆常跑路线每日处理 - {datetime.now()}")
            logging.info(f"[CONFIG] 目标日期: {target_date}")
            logging.info(f"[CONFIG] 所属月份: {target_month}")
            logging.info(f"[CONFIG] 并行数: {MAX_WORKERS}, 批次大小: {BATCH_SIZE}")
            logging.info(f"[CONFIG] 日志文件: {log_file}")
            logging.info("="*80)
            
            # 处理单日数据
            overall_start_time = time.time()
            day_stats = process_single_day(main_client, target_date)
            
            # 总体统计
            overall_elapsed = time.time() - overall_start_time
            
            logging.info("="*80)
            logging.info("[FINAL REPORT] 每日处理完成")
            logging.info("="*80)
            logging.info(f"处理日期: {target_date}")
            logging.info(f"总车辆数: {day_stats['total_vehicles']}")
            logging.info(f"已处理: {day_stats['processed']}")
            logging.info(f"成功: {day_stats['success']}")
            logging.info(f"失败: {day_stats['failed']}")
            logging.info(f"总耗时: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f}min)")
            logging.info("="*80)
            
            main_client.close()
            logging.info("[ALL COMPLETE] 每日处理全部完成")
            
            # 返回状态码
            return 0 if day_stats['failed'] == 0 else 1
            
        elif args.mode == 'batch':
            # ============ 批量处理模式（历史数据补录） ============
            if not args.start_date or not args.end_date:
                logging.error("[ERROR] 批量处理模式需要指定 --start-date 和 --end-date")
                return 1
            
            # 设置日志
            log_file = setup_logging(f"batch_{args.start_date}_{args.end_date}")
            
            logging.info("="*80)
            logging.info(f"[START] 车辆常跑路线批量处理 - {datetime.now()}")
            logging.info(f"[CONFIG] 日期范围: {args.start_date} 至 {args.end_date}")
            logging.info(f"[CONFIG] 日志文件: {log_file}")
            logging.info("="*80)
            
            # 处理批量数据
            overall_start_time = time.time()
            batch_stats = process_date_range(main_client, args.start_date, args.end_date)
            
            # 总体统计
            overall_elapsed = time.time() - overall_start_time
            
            logging.info("="*80)
            logging.info("[FINAL REPORT] 批量处理完成")
            logging.info("="*80)
            logging.info(f"日期范围: {args.start_date} 至 {args.end_date}")
            logging.info(f"总天数: {batch_stats['total_days']}")
            logging.info(f"成功天数: {batch_stats['processed_days']}")
            logging.info(f"失败天数: {batch_stats['failed_days']}")
            logging.info(f"总车辆: {batch_stats['total_vehicles']}")
            logging.info(f"总处理: {batch_stats['total_processed']}")
            logging.info(f"总成功: {batch_stats['total_success']}")
            logging.info(f"总耗时: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f}min)")
            logging.info("="*80)
            
            main_client.close()
            logging.info("[ALL COMPLETE] 批量处理全部完成")
            
            # 返回状态码
            return 0 if batch_stats['failed_days'] == 0 else 1
            
        elif args.mode == MODE_MONTHLY_MERGE:
            # ============ 月度合并模式 ============
            target_month = args.month or get_last_month()
            
            # 设置日志
            log_file = setup_logging(f"merge_{target_month}")
            
            logging.info("="*80)
            logging.info(f"[START] 车辆常跑路线月度合并 - {datetime.now()}")
            logging.info(f"[CONFIG] 目标月份: {target_month}")
            logging.info(f"[CONFIG] 日志文件: {log_file}")
            logging.info("="*80)
            
            # 处理月度合并
            overall_start_time = time.time()
            merge_stats = process_monthly_merge(main_client, target_month)
            
            # 总体统计
            overall_elapsed = time.time() - overall_start_time
            
            logging.info("="*80)
            logging.info("[FINAL REPORT] 月度合并完成")
            logging.info("="*80)
            logging.info(f"处理月份: {target_month}")
            logging.info(f"总车辆数: {merge_stats['total_vehicles']}")
            logging.info(f"写入记录: {merge_stats['inserted']}")
            logging.info(f"总耗时: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f}min)")
            logging.info("="*80)
            
            main_client.close()
            logging.info("[ALL COMPLETE] 月度合并全部完成")
            
            # 返回状态码
            return 0
            
    except Exception as e:
        logging.error(f"[ERROR] 处理失败: {e}")
        main_client.close()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
