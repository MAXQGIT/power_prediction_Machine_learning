import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

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
# sns.set(rc={'figure.figsize':(11, 4)})
# fig,ax =plt.subplots(figsize=(12,5))
# sns.boxplot(data=features_and_target.dropna(),
#             x='weekday',
#             y='nat_demand',
#             hue='season',
#             ax=ax,
#             linewidth=1)
# ax.set_title('Power Use MW by Day of Week')
# ax.set_xlabel('Day of Week')
# ax.set_ylabel('Energy (MW)')
# ax.legend(bbox_to_anchor=(1, 1))
# plt.show()
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
scler = MinMaxScaler()
X = scler.fit_transform(X)

import os

file_path = "my_model.pkl"

# Check if the file exists before attempting to delete
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"File '{file_path}' has been deleted.")
else:
    print(f"File '{file_path}' does not exist.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
models = [
    ("Linear Regression", LinearRegression()),
    ("Decision Tree", DecisionTreeRegressor()),
    ("Random Forest", RandomForestRegressor()),
    ("Gradient Boosting", GradientBoostingRegressor()),
    ("SVR", SVR()),
    ("KNN", KNeighborsRegressor())
]
for name, model in models:
    model.fit(X_train, y_train)

    # Make predictions and evaluate each model
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)
    print(f"Model: {name}")
    print('Root Mean Squared Error (RMSE):', rmse)
    print('R-squared (R2) Score:', r2)
    print()
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

# Define the hyperparameters and their possible values for tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
}

# Create the GradientBoostingRegressor model
model = GradientBoostingRegressor()

# Create the GridSearchCV object
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the GridSearchCV to the training data
grid_search.fit(X_train, y_train)
# Get the best hyperparameters and the best estimator
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Make predictions and evaluate the best model
predictions = best_model.predict(X_test)
rmse = mean_squared_error(y_test, predictions, squared=False)
r2 = r2_score(y_test, predictions)

print("Best Hyperparameters:", best_params)
print('Root Mean Squared Error (RMSE):', rmse)
print('R-squared (R2) Score:', r2)
import joblib

model_filename = 'best_gradient_boosting_model.pkl'
joblib.dump(best_model, model_filename)

print(f"Best model saved to {model_filename}")
