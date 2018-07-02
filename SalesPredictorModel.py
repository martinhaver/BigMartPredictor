import pandas as pd
import numpy as np
from sklearn import cross_validation, metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

train = pd.read_csv("trainClean.csv")
test = pd.read_csv("testClean.csv")

# Calculate mean
salesMean = train['Item_Outlet_Sales'].mean()

# Create a data frame
theBase = test[['Item_Identifier', 'Outlet_Identifier']]
theBase['Item_Outlet_Sales'] = salesMean

# Export results
theBase.to_csv("result1.csv", index=False)

# Define target and ID columns:
target = 'Item_Outlet_Sales'
IDcolumn = ['Item_Identifier', 'Outlet_Identifier']


# The function producing report of the model containing CV error for evaluation
def report(alg, dtrain, dtest, predictors, target, IDcolumn, filename):
    # Fitting function
    alg.fit(dtrain[predictors], dtrain[target])
    # Predicting function
    predictions = alg.predict(dtrain[predictors])
    # Cross-validation:
    cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    # Print model report:
    print("\nModel Report")
    print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, predictions)))
    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (
            np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))
    # Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    # Export the results:
    IDcolumn.append(target)
    submission = pd.DataFrame({x: dtest[x] for x in IDcolumn})
    submission.to_csv(filename, index=False)


# Linear regression model
predictors = [x for x in train.columns if x not in [target]+IDcolumn]
# print predictors
alg1 = LinearRegression(normalize=True)
report(alg1, train, test, predictors, target, IDcolumn, 'result1.csv')
varx1 = pd.Series(alg1.coef_, predictors).sort_values()
varx1.plot(kind='bar', title='Model Coefficients')

# Random forest model
predictors = [x for x in train.columns if x not in [target]+IDcolumn]
algr = RandomForestRegressor(n_estimators=500, max_depth=7, min_samples_leaf=100, n_jobs=4)
report(algr, train, test, predictors, target, IDcolumn, 'result2.csv')
varx2 = pd.Series(algr.feature_importances_, predictors).sort_values(ascending=False)
varx2.plot(kind='bar', title='Feature Importances')