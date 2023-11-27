import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

# 读取数据
data = pd.read_excel(r'D:\矫正完成夜间灯光数据连续\西南\Excle\JS重庆.xlsx')

# 提取x和y的数据列
x_data = data.iloc[:, 5].values.reshape(-1, 1)  # 第四列作为x，需要转换为二维数组
y_data = data.iloc[:, 4].values

# 将x_data和对应的y_data按照x_data的值进行升序排序
sort_indices = np.argsort(x_data[:, 0])
x_data_sorted = x_data[sort_indices]
y_data_sorted = y_data[sort_indices]

# 不含截距的三次函数拟合
poly = PolynomialFeatures(degree=3, include_bias=False)
poly_features = poly.fit_transform(x_data_sorted)
linear_model = LinearRegression(fit_intercept=False)  # fit_intercept=False表示不添加截距项
linear_model.fit(poly_features, y_data_sorted)

# 输出不含截距的三次函数拟合方程
coefs = linear_model.coef_
equation = f'y = {coefs[2]:.2e}x^3 + {coefs[1]:.2e}x^2 + {coefs[0]:.2e}x'

# 输出R2值
y_pred = linear_model.predict(poly_features)
r2_value = r2_score(y_data_sorted, y_pred)

print("不含截距的三次函数拟合方程：", equation)
print("R2值：", r2_value)

# 使用交叉验证选择最佳的alpha值
alphas = [0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000, 10000, 100000, 1000000, 1000000]  # 可以根据需要调整alpha值的候选列表
ridge_model_cv = RidgeCV(alphas=alphas, fit_intercept=False)
ridge_model_cv.fit(poly_features, y_data_sorted)

best_alpha = ridge_model_cv.alpha_
print("Best alpha:", best_alpha)

# 使用最佳alpha值的模型进行拟合和输出
ridge_model_best = Ridge(alpha=best_alpha, fit_intercept=False)
#ridge_model_best = Ridge(alpha=10, fit_intercept=False)
ridge_model_best.fit(poly_features, y_data_sorted)

# 输出加入L2正则化的不含截距的三次函数拟合方程和R2值
coefs_ridge_best = ridge_model_best.coef_
equation_ridge_best = f'y = {coefs_ridge_best[2]:.2e}x^3 + {coefs_ridge_best[1]:.2e}x^2 + {coefs_ridge_best[0]:.2e}x'
y_pred_ridge_best = ridge_model_best.predict(poly_features)
r2_value_ridge_best = r2_score(y_data_sorted, y_pred_ridge_best)

print("加入L2正则化的不含截距的三次函数拟合方程（最佳alpha）：", equation_ridge_best)
print("R2值（加入L2正则化，最佳alpha）：", r2_value_ridge_best)

# 获取Excel表格的名字
excel_filename = r'D:\矫正完成夜间灯光数据连续\西南\Excle\JS重庆.xlsx'
excel_name = excel_filename.split('\\')[-1].split('.')[0]

# 指定输出文件路径和名称
output_path = r'D:\矫正完成夜间灯光数据连续\西南\Excle\图像'
output_file = f'{excel_name}_polynomial_regression.png'

# 将系数和R2值格式化为指定的小数位数
equation = f'y = {coefs[2]:.6f}x^3 + {coefs[1]:.6f}x^2 + {coefs[0]:.6f}x'
equation_ridge_best = f'y = {coefs_ridge_best[2]:.6f}x^3 + {coefs_ridge_best[1]:.6f}x^2 + {coefs_ridge_best[0]:.6f}x'
r2_value = round(r2_value, 4)
r2_value_ridge_best = round(r2_value_ridge_best, 4)

# 绘制拟合曲线和显示方程和R2值
plt.figure(figsize=(10, 6))
plt.scatter(x_data_sorted, y_data_sorted, color='blue', label='数据')
plt.plot(x_data_sorted, y_pred, color='red', label='三次多项式拟合 (无L2正则化)')
plt.plot(x_data_sorted, y_pred_ridge_best, color='orange', label='三次多项式拟合 (有L2正则化)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('多项式回归')
plt.legend(loc='lower right')
plt.text(0.05, 0.9, equation, transform=plt.gca().transAxes, fontsize=12)
plt.text(0.05, 0.85, f'R2值: {r2_value}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.05, 0.1, equation_ridge_best, transform=plt.gca().transAxes, fontsize=12)
plt.text(0.05, 0.05, f'R2值 (加入L2正则化): {r2_value_ridge_best}', transform=plt.gca().transAxes, fontsize=12)
plt.savefig(output_path + '\\' + output_file)
plt.show()