import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def load_and_report_quality(file_path):
    df = pd.read_parquet(file_path)
    print(f"原始数据总行数: {len(df)}")
    # 1. 缺失率统计
    missing_rate = df.isnull().sum() / len(df) * 100
    print("\n各字段缺失率 (%):")
    print(missing_rate[missing_rate > 0])
    # 2. 异常值统计
    print("\n异常值概览 (车费和距离):")
    print(df[['trip_distance', 'fare_amount', 'passenger_count']].describe())
    return df

def clean_data(df):
    """
    清洗策略与理由：
    1. 车费金额必须>0
    2. 行程距离必须>0且<1000，距离为0的订单没有分析价值，过长的距离可能是数据错误
    3. 乘客人数过滤掉0人和>9人的异常记录。
    4. 行程时间: 剔除接单时间晚于下车时间的错误数据，以及异常漫长的订单。
    """
    initial_len = len(df)
    
    # 执行清洗
    df_clean = df[
        (df['fare_amount'] > 0) &
        (df['trip_distance'] > 0) & (df['trip_distance'] < 1000) &
        (df['passenger_count'] > 0) & (df['passenger_count'] <= 9) &
        (df['tpep_dropoff_datetime'] > df['tpep_pickup_datetime'])
    ].copy()
    # 时间异常值清洗（过滤持续时间大于10小时的）
    duration = (df_clean['tpep_dropoff_datetime'] - df_clean['tpep_pickup_datetime']).dt.total_seconds() / 3600
    df_clean = df_clean[duration <= 10]
    print(f"清洗完毕。剔除了 {initial_len - len(df_clean)} 条异常记录。剩余 {len(df_clean)} 条。")
    return df_clean
def feature_engineering(df):
    # 基础时间特征提取
    pickup_dt = df['tpep_pickup_datetime']
    df['pickup_hour'] = pickup_dt.dt.hour
    df['pickup_weekday'] = pickup_dt.dt.weekday # 0=周一, 6=周日
    df['is_weekend'] = df['pickup_weekday'].apply(lambda x: 1 if x >= 5 else 0)
    # 定义高峰期时间段：工作日的早高峰7-9，晚高峰17-19
    def is_peak(row):
        if row['is_weekend'] == 0:
            if (7 <= row['pickup_hour'] <= 9) or (17 <= row['pickup_hour'] <= 19):
                return 1
        return 0
    df['is_peak_hour'] = df.apply(is_peak, axis=1)
    # 衍生特征1:trip_duration_mins，行程耗时/分钟
    df['trip_duration_mins'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60.0
    # 衍生特征 2: average_speed_mph，英里/小时
    df['average_speed_mph'] = df['trip_distance'] / (df['trip_duration_mins'] / 60.0)
    df = df[df['average_speed_mph'] <= 120] # 过滤极速异常
    return df

def main():
    print("城市出租车出行数据分析与智能问答系统")
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('data'):
        os.makedirs('data')
    data_path = './data/yellow_tripdata_2023-01.parquet'
    df_raw = load_and_report_quality(data_path)
    df_clean = clean_data(df_raw)
    df_final = feature_engineering(df_clean)
    # 预览处理后的数据
    print("\n数据处理完毕，前 3 行预览:")
    print(df_final[['tpep_pickup_datetime', 'trip_distance', 'fare_amount', 'is_peak_hour', 'average_speed_mph']].head(3))
if __name__ == "__main__":
    main()