from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

model = LinearRegression()  

obtain_model=model.fit(X_t,Y_t)
test_predict_label = obtain_model.predict(X_testing)
test_predict_label = np.array(test_predict_label)

mae = mean_absolute_error(Y_testing, test_predict_label)
mse = mean_squared_error(Y_testing, test_predict_label)

print(f'线性回归的MAE: {mae}')
print(f'线性回归的MSE: {mse}')
