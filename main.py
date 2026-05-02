import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import re
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

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
def run_m2_analysis(df):
    print("\n分析与可视化")
    out_dir = 'outputs/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # 图表1:出行需求时间规律
    plt.figure(figsize=(10, 6))
    #按日期和小时统计每天每小时的单量
    df['pickup_date'] = df['tpep_pickup_datetime'].dt.date
    daily_hourly_vol = df.groupby(['pickup_date', 'pickup_hour', 'is_weekend']).size().reset_index(name='trip_count')
    # 工作日/周末在每个小时的平均单量
    avg_hourly_vol = daily_hourly_vol.groupby(['is_weekend', 'pickup_hour'])['trip_count'].mean().reset_index()
    
    sns.lineplot(data=avg_hourly_vol, x='pickup_hour', y='trip_count', hue='is_weekend', marker='o', palette=['blue', 'orange'])
    plt.title('分小时平均订单量规律 (工作日 vs 周末)')
    plt.xlabel('一天中的小时 (0-23)')
    plt.ylabel('平均单小时订单量')
    plt.xticks(range(0, 24))
    plt.legend(title='日期类型', labels=['工作日', '周末'])
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, '1_time_pattern.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图表 2:区域热度分析(上客量最高的TOP10区域)
    plt.figure(figsize=(12, 6))
    
    # 提取TOP10上客区域
    top10_locations = df['PULocationID'].value_counts().nlargest(10).index
    df_top10 = df[df['PULocationID'].isin(top10_locations)]
    # 统计这些区域在高峰期和非高峰期的单量
    loc_peak_counts = df_top10.groupby(['PULocationID', 'is_peak_hour']).size().unstack(fill_value=0)
    # 按照总单量排序
    loc_peak_counts['total'] = loc_peak_counts.sum(axis=1)
    loc_peak_counts = loc_peak_counts.sort_values('total', ascending=False).drop(columns='total')
    loc_peak_counts.plot(kind='bar', stacked=True, figsize=(12,6), color=['#87CEFA', '#FF6347'])
    plt.title('Top 10 上客区域热度分布 (高峰期与非高峰期)')
    plt.xlabel('上客区域 ID (PULocationID)')
    plt.ylabel('总订单量')
    plt.legend(title='时段', labels=['非高峰期', '高峰期'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '2_top10_regions.png'), dpi=300)
    plt.close()

    # 图表 3: 车费影响因素分析 (行程距离 vs 车费散点图)
    plt.figure(figsize=(10, 6))
    # 300万数据画散点图太密，随机抽样10000条记录进行展示
    sample_df = df.sample(n=10000, random_state=42)
    
    sns.scatterplot(data=sample_df, x='trip_distance', y='fare_amount', hue='passenger_count', 
                    palette='viridis', alpha=0.6, s=20)
    plt.title('行程距离与车费关系散点图 (1万条抽样数据)')
    plt.xlabel('行程距离 (Miles)')
    plt.ylabel('车费金额 ($)')
    # 限制坐标轴范围以剔除极个别离群点，让图表更清晰
    plt.xlim(0, sample_df['trip_distance'].quantile(0.99)) 
    plt.ylim(0, sample_df['fare_amount'].quantile(0.99))
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, '3_fare_factors.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # 图表 4:自选分析：出发地是机场占比随每小时变化图
    plt.figure(figsize=(10, 6))
    # 机场ID：132(JFK), 138(LGA), 1(EWR)
    airport_ids = [1, 132, 138]
    df['is_airport_pickup'] = df['PULocationID'].isin(airport_ids)
    
    # 按小时聚合，计算每个小时的总单量和机场单量
    hourly_stats = df.groupby('pickup_hour').agg(
        total_trips=('PULocationID', 'count'),
        airport_trips=('is_airport_pickup', 'sum')
    ).reset_index()
    # 计算该小时内机场订单的占比
    hourly_stats['airport_ratio_pct'] = (hourly_stats['airport_trips'] / hourly_stats['total_trips']) * 100
    # 绘制柱状+折线图组合
    sns.barplot(data=hourly_stats, x='pickup_hour', y='airport_ratio_pct', color='skyblue', alpha=0.7)
    sns.lineplot(data=hourly_stats, x='pickup_hour', y='airport_ratio_pct', color='darkblue', marker='o', linewidth=2)
    plt.title('24小时内出发地为机场的订单占比 (%)')
    plt.xlabel('一天中的小时 (0-23)')
    plt.ylabel('机场订单所占百分比 (%)')
    plt.xticks(range(0, 24))
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    # 添加数值标注
    for i in range(len(hourly_stats)):
        plt.text(i, hourly_stats['airport_ratio_pct'][i] + 0.2, 
                 f"{hourly_stats['airport_ratio_pct'][i]:.1f}%", 
                 ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '4_hourly_airport_ratio.png'), dpi=300)
    plt.close()

# 预测模型模块
# 定义PyTorch神经网络模型
class DemandPredictorNN(nn.Module):
    def __init__(self, input_dim):
        super(DemandPredictorNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # 输出层
        )
        
    def forward(self, x):
        return self.net(x)

def run_m3_prediction(df):
    print("\n开始构建和训练预测模型")
    # 构造训练数据
    # 按日期、小时、区域聚合，计算总单量
    df['pickup_date'] = df['tpep_pickup_datetime'].dt.date
    agg_df = df.groupby(['pickup_date', 'pickup_hour', 'PULocationID', 'pickup_weekday', 'is_peak_hour', 'is_weekend']).size().reset_index(name='demand')
    
    # 避免数据量过大,选取单量最活跃的Top 20区域进行训练预测
    top20_locs = agg_df.groupby('PULocationID')['demand'].sum().nlargest(20).index
    model_df = agg_df[agg_df['PULocationID'].isin(top20_locs)].copy()
    # 提取特征列和目标列
    feature_cols = ['PULocationID', 'pickup_hour', 'pickup_weekday', 'is_weekend', 'is_peak_hour']
    X = model_df[feature_cols].values
    y = model_df['demand'].values
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PyTorch神经网络训练
    # 转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # 初始化模型、损失函数(MSE)和优化器(Adam)
    input_dim = X_train.shape[1]
    nn_model = DemandPredictorNN(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=0.01)
    
    epochs = 200
    train_losses = []
    
    for epoch in range(epochs):
        # 前向传播
        outputs = nn_model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        if (epoch+1) % 50 == 0:
            print(f"     Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    # 绘制 Loss 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.title('PyTorch Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig('outputs/pytorch_loss_curve.png')
    plt.close()
    print("  -> Loss 曲线已保存至 outputs/pytorch_loss_curve.png")
    
    # 神经网络评估
    nn_model.eval()
    with torch.no_grad():
        nn_preds = nn_model(X_test_tensor).numpy()
    
    nn_mae = mean_absolute_error(y_test, nn_preds)
    nn_rmse = np.sqrt(mean_squared_error(y_test, nn_preds))
    
    # 3. 随机森林对比实验
    print("训练随机森林进行对比")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    
    # 4. 对比实验的情况分析
    print("M3 预测模型性能对比(测试集)")
    print(f"【PyTorch 神经网络】:")
    print(f"  MAE (平均绝对误差):  {nn_mae:.2f} 单/小时")
    print(f"  RMSE (均方根误差):   {nn_rmse:.2f} 单/小时")
    print(f"\n【Random Forest 随机森林】:")
    print(f"  MAE (平均绝对误差):  {rf_mae:.2f} 单/小时")
    print(f"  RMSE (均方根误差):   {rf_rmse:.2f} 单/小时")


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
    run_m2_analysis(df_final)
    run_m3_prediction(df_final)
    run_m4_qa_system(df_final)

if __name__ == "__main__":
    main()
