from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV

plsda = PLSRegression()

n_components_values = list(np.arange(1, 101))

param_grid = {'n_components': n_components_values}

grid_search = GridSearchCV(plsda, param_grid, scoring='neg_mean_squared_error', cv=10)
grid_search.fit(X_t, Y_t)

best_n_components = grid_search.best_params_['n_components']

print(f'最佳主成分数量: {best_n_components}')

best_plsda = PLSRegression(n_components=best_n_components)
best_plsda.fit(X_t, Y_t)

y_pred = best_plsda.predict(X_testing)

mae = mean_absolute_error(Y_testing, y_pred)
mse = mean_squared_error(Y_testing, y_pred)

print(f'线性回归的MAE: {mae}')
print(f'线性回归的MSE: {mse}')
