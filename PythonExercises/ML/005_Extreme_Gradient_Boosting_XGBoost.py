# =========================================================================================== #
# ========                  Extreme Gradient Boosting with XGBoost                   ======== #
# =========================================================================================== #

# ============================== #
# =====    Ejercicio 1     ===== #
# ============================== #

# Import xgboost
import xgboost as xgb
# Create arrays for the features and the target: X, y
X, y = churn_data.iloc[:,:-1], churn_data.iloc[:,-1]
# Create the training and test sets
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = 0.2, random_state = 123)
# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier(objective = 'binary:logistic', n_estimators = 10, seed = 123)
# Fit the classifier to the training set
xg_cl.fit(X_train, y_train)
# Predict the labels of the test set: preds
preds = xg_cl.predict(X_test)
# Compute the accuracy: accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))

# ============================== #
# =====    Ejercicio 2     ===== #
# ============================== #

# Import the necessary modules
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
# Instantiate the classifier: dt_clf_4
dt_clf_4 = DecisionTreeClassifier(max_depth = 4)
# Fit the classifier to the training set
dt_clf_4.fit(X_train, y_train)
# Predict the labels of the test set: y_pred_4
y_pred_4 = dt_clf_4.predict(X_test)
# Compute the accuracy of the predictions: accuracy
accuracy = float(np.sum(y_pred_4==y_test))/y_test.shape[0]
print("accuracy:", accuracy)

# ============================== #
# =====    Ejercicio 3     ===== #
# ============================== #

# Create arrays for the features and the target: X, y
X, y = churn_data.iloc[:,:-1], churn_data.iloc[:,-1]
# Create the DMatrix from X and y: churn_dmatrix
churn_dmatrix = xgb.DMatrix(data = X, label = y)
# Create the parameter dictionary: params
params = {"objective":"reg:logistic", "max_depth":3}
# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain = churn_dmatrix, params = params, 
                    nfold = 3, num_boost_round = 5, 
                    metrics = "error", as_pandas = True, seed = 123)
# Print cv_results
print(cv_results)
# Print the accuracy
print(((1 - cv_results["test-error-mean"]).iloc[-1]))

# ============================== #
# =====    Ejercicio 4     ===== #
# ============================== #

# Perform cross_validation: cv_results
cv_results = xgb.cv(dtrain = churn_dmatrix, params = params, 
                    nfold = 3, num_boost_round = 5, 
                    metrics = "auc", as_pandas = True, seed = 123)
# Print cv_results
print(cv_results)
# Print the AUC
print((cv_results["test-auc-mean"]).iloc[-1])

# ============================== #
# =====    Ejercicio 5     ===== #
# ============================== #

# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
# Instantiate the XGBRegressor: xg_reg
xg_reg = xgb.XGBRegressor(objective = 'reg:linear', n_estimators = 10,
                          booster = "gbtree", seed = 123)
# Fit the regressor to the training set
xg_reg.fit(X_train, y_train)
# Predict the labels of the test set: preds
preds = xg_reg.predict(X_test)
# Compute the rmse: rmse
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

# ============================== #
# =====    Ejercicio 6     ===== #
# ============================== #

# Convert the training and testing sets into DMatrixes: DM_train, DM_test
DM_train = xgb.DMatrix(X_train, y_train)
DM_test =  xgb.DMatrix(X_test, y_test)
# Create the parameter dictionary: params
params = {"objective": "reg:linear", "booster": "gblinear"}
# Train the model: xg_reg
xg_reg = xgb.train(dtrain = DM_train, params = params, num_boost_round = 5)
# Predict the labels of the test set: preds
preds = xg_reg.predict(DM_test)
# Compute and print the RMSE
rmse = np.sqrt(mean_squared_error(y_test,preds))
print("RMSE: %f" % (rmse))

# ============================== #
# =====    Ejercicio 7     ===== #
# ============================== #

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)
# Create the parameter dictionary: params
params = {"objective":"reg:linear", "max_depth":4}
# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain = housing_dmatrix, params = params, 
                    nfold = 4, num_boost_round = 5, metrics = "rmse", 
                    as_pandas = True, seed = 123)
# Print cv_results
print(cv_results)
# Extract and print final boosting round metric
print((cv_results["test-rmse-mean"]).tail(1))

# ============================== #
# =====    Ejercicio 8     ===== #
# ============================== #

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data = X, label = y)
# Create the parameter dictionary: params
params = {"objective":"reg:linear", "max_depth":4}
# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain = housing_dmatrix, params = params, 
                    nfold = 4, num_boost_round = 5, metrics = "mae", 
                    as_pandas = True, seed = 123)
# Print cv_results
print(cv_results)
# Extract and print final boosting round metric
print((cv_results["test-mae-mean"]).tail(1))

# ============================== #
# =====    Ejercicio 9     ===== #
# ============================== #

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)
reg_params = [1, 10, 100]
# Create the initial parameter dictionary for varying l2 strength: params
params = {"objective": "reg:linear", "max_depth": 3}
# Create an empty list for storing rmses as a function of l2 complexity
rmses_l2 = []
# Iterate over reg_params
for reg in reg_params:
    # Update l2 strength
    params["lambda"] = reg
    # Pass this updated param dictionary into cv
    cv_results_rmse = xgb.cv(dtrain = housing_dmatrix, params = params, nfold = 2, 
                             num_boost_round = 5, metrics = "rmse", as_pandas = True, seed = 123)
    # Append best rmse (final round) to rmses_l2
    rmses_l2.append(cv_results_rmse["test-rmse-mean"].tail(1).values[0])
# Look at best rmse per l2 param
print("Best rmse as a function of l2:")
print(pd.DataFrame(list(zip(reg_params, rmses_l2)), columns=["l2", "rmse"]))

# ============================== #
# =====    Ejercicio 10    ===== #
# ============================== #

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)
# Create the parameter dictionary: params
params = {"objective": "reg:linear", "max_depth": 2}
# Train the model: xg_reg
xg_reg = xgb.train(params = params, dtrain = housing_dmatrix, num_boost_round = 10)
# Plot the first tree
xgb.plot_tree(xg_reg, num_trees = 0)
plt.show()
# Plot the fifth tree
xgb.plot_tree(xg_reg, num_trees = 5)
plt.show()
# Plot the last tree sideways
xgb.plot_tree(xg_reg, num_trees = 10, rankdir = 'LR')
plt.show()

# ============================== #
# =====    Ejercicio 11    ===== #
# ============================== #

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data = X, label = y)
# Create the parameter dictionary: params
params = {"objective": "reg:linear", "max_depth": 4}
# Train the model: xg_reg
xg_reg = xgb.train(params = params, dtrain = housing_dmatrix, num_boost_round = 10)
# Plot the feature importances
xgb.plot_importance(xg_reg)
plt.show()
