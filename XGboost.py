import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype
import pandas as pd
import numpy as np
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
# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.002, random_state=42)

# 转换数据为 XGBoost 的 DMatrix 格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置 XGBoost 的参数
params = {
    'objective': 'reg:squarederror',  # 使用均方误差作为回归任务的目标函数
    'eval_metric': 'rmse',  # 使用均方根误差作为评估指标
    'max_depth': 3,  # 决策树的最大深度
    'learning_rate': 0.1,  # 学习率
    'subsample': 0.8,  # 每次迭代中随机选择 80% 的样本
    'colsample_bytree': 0.8,  # 每次迭代中随机选择 80% 的特征
    'seed': 42  # 随机种子，用于复现结果
}

# 训练模型
num_round = 100  # 迭代轮数
bst = xgb.train(params, dtrain, num_round)

# 在测试集上进行预测
y_pred = bst.predict(dtest)
print(y_pred)
print(y_test)
# 计算均方根误差
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'Root Mean Squared Error (RMSE): {rmse}')
plt.figure()
plt.plot(y_test, label='真实值')
plt.plot(y_pred, label='预测值')
plt.legend()
plt.title('预测和真实值对比')
plt.savefig('预测和真实值对比.png')
