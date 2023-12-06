import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

df = pd.read_csv("data/NASA_planetary_data.csv", index_col = 'rowid')

df.dropna(subset = ['pl_orbper'], axis = 0, how = 'any', inplace = True)

df_x = df[['pl_orbper', 'pl_orbsmax', 'pl_orbeccen','st_radv','pl_trandep','st_vsin']] # independent features
df_y = df['pl_dens'] # target variable

df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state = 0)


df_x_train['pl_orbeccen'] = df_x_train['pl_orbeccen'].fillna(df_x_train['pl_orbeccen'].mean()) # filling null values with appropriate substitutes
df_x_train['pl_trandep'] = df_x_train['pl_trandep'].fillna(df_x_train['pl_trandep'].median())
df_x_train['pl_orbsmax'] = df_x_train['pl_orbsmax'].fillna(df_x_train['pl_orbsmax'].median())
df_x_train['st_radv'] = df_x_train['st_radv'].fillna(df_x_train['st_radv'].mean())
df_x_train['st_vsin'] = df_x_train['st_vsin'].fillna(df_x_train['st_vsin'].median())


df_x_test['pl_orbeccen'] = df_x_test['pl_orbeccen'].fillna(df_x_test['pl_orbeccen'].mean())
df_x_test['pl_trandep'] = df_x_test['pl_trandep'].fillna(df_x_test['pl_trandep'].median())
df_x_test['pl_orbsmax'] = df_x_test['pl_orbsmax'].fillna(df_x_test['pl_orbsmax'].median())
df_x_test['st_radv'] = df_x_test['st_radv'].fillna(df_x_test['st_radv'].mean())
df_x_test['st_vsin'] = df_x_test['st_vsin'].fillna(df_x_test['st_vsin'].median())

df_y_train = df_y_train.fillna(df_y_train.median())
df_y_test = df_y_test.fillna(df_y_test.median())

df_x_train.dropna(axis = 0, how = 'any', inplace = True)
df_x_test.dropna(axis = 0, how = 'any', inplace = True)

df_x_train['pl_orbper'] = [round(x) for x in (list(df_x_train['pl_orbper'].values))] # cleaning the 'orbital period' feature 
                                                                                     # as it is not sensible to fill the null values with anything
df_x_train['pl_orbper'] = list(df_x_train['pl_orbper'].values)

df_x_test['pl_orbper'] = [round(x) for x in (list(df_x_test['pl_orbper'].values))]
df_x_test['pl_orbper'] = list(df_x_test['pl_orbper'].values)

scaler = MinMaxScaler()
df_x_train_norm = scaler.fit_transform(df_x_train) # scaling the features
df_x_test_norm = scaler.fit_transform(df_x_test)

df_y_train_norm = scaler.fit_transform(df_y_train.values.reshape(-1, 1))
df_y_test_norm = scaler.fit_transform(df_y_test.values.reshape(-1, 1))

lr_model = LinearRegression() 
lr_model.fit(df_x_train_norm, df_y_train_norm)  # training a simple linear model
y_pred = lr_model.predict(df_x_test_norm)
mse_lr = mean_squared_error(df_y_test, y_pred)

rf_model = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf_model.fit(df_x_train_norm, df_y_train_norm.ravel()) # random forest 
y_pred_rf = rf_model.predict(df_x_test_norm)
mse_rf = mean_squared_error(df_y_test_norm, y_pred_rf)

xgb_model = XGBRegressor(objective = 'reg:squarederror', params = { 
         'learning_rate' : 0.1,
         'n_estimators' : 100,
         'max_depth' : 3,
         'subsmaple' : 0.8,
         'colsample_bytree' : 0.8})
xgb_model.fit(df_x_train_norm, df_y_train_norm)  # XGBoost
y_pred_xgb = xgb_model.predict(df_x_test_norm)
mse_xgb = mean_squared_error(df_y_test_norm, y_pred_xgb)

models = ['Linear Regression', 'Random Forest', 'XGBoost']
mse_values = [mse_lr, mse_rf, mse_xgb]
x_values = np.arange(len(models))

plt.bar(x_values-0.2, [mse_lr, 0, 0], width = 0.2, color = 'blue', label = 'Linear Regression')
plt.bar(x_values, [0, mse_rf, 0], width = 0.2, color='orange', label='Random Forest')
plt.bar(x_values+0.2, [0, 0, mse_xgb], width = 0.2, color='green', label='XGBoost')
plt.xticks(x_values, models)
plt.ylabel('Mean Squared Error (MSE)')
plt.yscale('log')
plt.title("Comparison of Mean Squared Error (MSE) among all 3 models")
plt.style.use('dark_background')
plt.tight_layout()
plt.savefig('model_comp.png')
plt.show()

ft_ls = ['orbital period', 'orbital semi-major axis', 'orbital eccentricity',
         'radial velocity', 'transit depth', 'system rotational vel.']

feature_names_rf = rf_model.feature_importances_ # getting feature importance according to random forest
feature_names_xgb = xgb_model.feature_importances_ # feature importance according to XGBoost

plt.bar(range(len(feature_names_rf)), feature_names_rf)
plt.xticks(range(len(feature_names_rf)), ft_ls, rotation = -90)
plt.ylabel('Feature Importance')
plt.title('Random Forest Regressor - Feature Importance')
plt.style.use('dark_background')
plt.tight_layout()
plt.savefig('rf_feature_importance plot.jpeg')
plt.show()

plt.bar(range(len(feature_names_xgb)), feature_names_xgb)
plt.xticks(range(len(feature_names_xgb)), ft_ls, rotation = -90)
plt.ylabel('Feature Importance')
plt.title('XG Boost - Feature Importance')
plt.style.use('dark_background')
plt.tight_layout()
plt.savefig('xgb_feature_importance plot.jpeg')
plt.show()

n = 6
r = np.arange(n)
width = 0.25

bar1 = plt.bar(r, feature_names_rf, color = 'w',
               width = width)
bar2 = plt.bar(r+width, feature_names_xgb, color = 'r',
               width = width)
plt.ylabel('Feature Importance')
plt.title('Random Forest vs XGBoost - Model Comparison')
plt.xticks(r+width, ft_ls, rotation = -90)
plt.legend((bar1, bar2), ('Random Forest', 'XGBoost'))
plt.style.use('dark_background')
plt.tight_layout()
plt.savefig('model_comparison.jpeg')
plt.show()