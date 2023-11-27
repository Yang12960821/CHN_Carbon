import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 读取xlsx表格
df = pd.read_excel(r'D:/矫正完成夜间灯光数据连续/东北/Excle/JS辽宁.xlsx')

# 提取第二列和第四列数据
y = df.iloc[:, 1].values
x = df.iloc[:, 3].values

# 定义不含截距的三次方程模型
def cubic_func(x, a, b, c):
    return a * x**3 + b * x**2 + c * x

# 进行不含截距的三次方程拟合
params, _ = curve_fit(cubic_func, x, y)

# 获取拟合参数
a, b, c = params

# 计算拟合曲线
y_fit = cubic_func(x, a, b, c)

# 绘制散点图和拟合曲线
plt.scatter(x, y, label='Data')
plt.plot(x, y_fit, color='red', label='Fit')
plt.xlabel('第四列')
plt.ylabel('第二列')
plt.title('散点图和拟合曲线')
plt.legend()

# 显示拟合方程
equation = f'拟合方程: y = {a:.4f} * x^3 + {b:.4f} * x^2 + {c:.4f} * x'
plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

# 显示图像
plt.show()
