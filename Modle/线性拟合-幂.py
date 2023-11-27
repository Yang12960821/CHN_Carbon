import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit
import os

# 读取xls文件
data = pd.read_excel('G:/数据/新方法预处理/Albers投影/重采样1KM/中国裁剪/饱和矫正之后/连续性矫正/最终/重新裁剪/重采样50km/数据.xlsx')

# 提取第三列和第三列之后的其他列
y = data.iloc[:, 2]

# 定义幂函数模型
def power_function(y, a, b):
    return a * np.power(y, b)

# 遍历第三列之后的每一列进行幂函数拟合和绘图
for col in data.columns[3:]:
    x = data[col]

    # 进行幂函数拟合
    params, _ = curve_fit(power_function, x, y)

    # 计算拟合曲线上的点
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = power_function(x_fit, params[0], params[1])

    # 绘制散点图和拟合曲线
    plt.scatter(x, y, s=1, label='Data')
    plt.plot(x_fit, y_fit, 'r', label='Power Function Fit')

    # 添加图例
    #plt.legend()

    # 计算残差
    residuals = y - power_function(x, params[0], params[1])

    # 计算总平方和和回归平方和
    total_sum_of_squares = np.sum((y - np.mean(y))**2)
    regression_sum_of_squares = np.sum((power_function(x, params[0], params[1]) - np.mean(y))**2)

    # 计算R平方
    r_squared = regression_sum_of_squares / total_sum_of_squares

    # 在图像右下角显示幂函数方程和R2
    equation = f'Y = {params[0]:.5f}X^{params[1]:.4f}'
    r_squared_text = f'R2 = {r_squared:.4f}'
    plt.text(0.05, 0.8, equation, transform=plt.gca().transAxes)
    plt.text(0.05, 0.75, r_squared_text, transform=plt.gca().transAxes)

    # 保存图像到指定目录下，保存名称为列名.png
    save_path = 'G:/数据/DM-NPP/幂函数/'
    os.makedirs(save_path, exist_ok=True)
    save_name = f'{col}.png'
    save_file = os.path.join(save_path, save_name)
    plt.savefig(save_file, dpi=150)

    # 清除当前图像，准备绘制下一个拟合结果图
    plt.clf()
