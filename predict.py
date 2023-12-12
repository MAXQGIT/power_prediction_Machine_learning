import joblib
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split
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
    X = df[['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear', 'weekday', 'season']]
    if label:
        y = df[label]
        return X, y
    return X


X, y = create_features(df, label='nat_demand')
features_and_target = pd.concat([X, y], axis=1)

Q1 = df['nat_demand'].quantile(0.25)
Q3 = df['nat_demand'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Filter the DataFramefeatures_and_target to exclude outliers
filtered_df = features_and_target[(features_and_target['nat_demand'] >= lower_bound) & (features_and_target['nat_demand'] <= upper_bound)]

df1 = pd.get_dummies(filtered_df, columns=['weekday', 'season'])
X = df1.drop('nat_demand', axis=1)
y = df1['nat_demand']
scler = joblib.load('scler.pkl')
X=scler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = joblib.load('best_gradient_boosting_model.pkl')
predictions = model.predict(X_test)
print(predictions)