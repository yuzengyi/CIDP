import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import adfuller
# from sklearn.feature_selection import RFECV
# from sklearn.model_selection import TimeSeriesSplit

# Load the DataFrame from an Excel file
df = pd.read_excel('data_imp.xlsx')

# Convert 'year' to datetime for any time-based operations
df['year'] = pd.to_datetime(df['year'], format='%Y')

# Stationarity check example for a feature 'EM1'
result = adfuller(df['EM1'].dropna())  # Ensure no NA values are present
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

# Feature importance using Random Forest
X = df.drop(['Y', 'ShortName', 'year'], axis=1)  # Drop non-predictor and target variable
y = df['Y']
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)
# Export feature importances to an Excel file
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
importances.to_excel('feature_importances.xlsx')

print("Feature importances have been exported to 'feature_importances.xlsx'.")

# Feature importances
# importances = pd.Series(model.feature_importances_, index=X.columns)
# print(importances.sort_values(ascending=False))

# # Recursive Feature Elimination with Cross-Validation
# rfecv = RFECV(estimator=RandomForestRegressor(n_estimators=100), step=1, cv=TimeSeriesSplit(n_splits=5))
# rfecv.fit(X, y)
#
# print("Optimal number of features: %d" % rfecv.n_features_)
# print('Best features:', X.columns[rfecv.support_])
#
# # Export the final dataset with selected features to Excel
# df_selected_features = df[X.columns[rfecv.support_]]
# df_selected_features.to_excel('final_selected_features.xlsx')
