import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.api.types import CategoricalDtype

plt.rcParams['font.sans-serif'] = ['SimHei']  # 绘图中文设置
df = pd.read_csv('continuous dataset.csv', index_col=[0], parse_dates=[0])
cat_type = CategoricalDtype(categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)


def create_features(df, label=None):
    """
    Creates time series features from datetime index.
    """
    df = df.copy()
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekday'] = df['date'].dt.day_name()
    df['weekday'] = df['weekday'].astype(cat_type)
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week  # Use isocalendar().week
    df['date_offset'] = (df.date.dt.month * 100 + df.date.dt.day - 320) % 1300

    df['season'] = pd.cut(df['date_offset'], [0, 300, 602, 900, 1300], labels=['Spring', 'Summer', 'Fall', 'Winter'])
    X = df[['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X


X, y = create_features(df, label='nat_demand')
X = np.array(X.values)
y = np.array(y.values)

# 将数据转换为 LightGBM 的 Dataset 格式
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.005, shuffle=False)
train_data = lgb.Dataset(X_train, label=y_train)

# 设置参数，这里使用的是线性回归任务
params = {
    'objective': 'regression',  # 设置为回归任务
    'metric': 'l2',  # 使用均方误差作为评估指标
    'boosting_type': 'gbdt',  # 使用梯度提升决策树
    'learning_rate': 0.1,  # 学习率
    'num_leaves': 31,  # 叶子节点数
    'feature_fraction': 0.9,  # 每次迭代中随机选择 90% 的特征
    'bagging_fraction': 0.8,  # 每次迭代中随机选择 80% 的样本
    'bagging_freq': 5,  # 每 5 次迭代执行一次 bagging
    'verbose': -1,  # 不输出训练参数
}

# 训练模型
num_round = 10000  # 迭代轮数
bst = lgb.train(params, train_data, num_round)

# 预测
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
# print(X_test[1,:])

# 可视化结果
plt.figure()
plt.plot(y_test, label='真实值')
plt.plot(y_pred, label='预测值')
plt.legend()
plt.title('预测和真实值对比')
plt.savefig('预测和真实值对比.png')
