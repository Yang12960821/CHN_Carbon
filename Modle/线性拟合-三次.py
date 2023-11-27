import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit
import os

# 读取xls文件
data = pd.read_excel('G:/数据/新方法预处理/Albers投影/重采样1KM/中国裁剪/鹤岗/excle/Nhegang.xls')

# 提取第三列和第三列之后的其他列
y = data.iloc[:, 2]

# 定义三次多项式模型
def cubic_function(y, a, b, c, d):
    return a * y**3 + b * y**2 + c * y + d

# 遍历第三列之后的每一列进行三次多项式拟合和绘图
for col in data.columns[3:]:
    x = data[col]

    # 进行三次多项式拟合
    params = np.polyfit(x, y, 3)

    # 计算拟合曲线上的点
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = cubic_function(x_fit, params[0], params[1], params[2], params[3])

    # 绘制散点图和拟合曲线
    plt.scatter(x, y, s=10, label='Data')
    plt.plot(x_fit, y_fit, 'r', label='Cubic Function Fit')

    # 添加图例
    #plt.legend()

    # 计算残差
    residuals = y - cubic_function(x, params[0], params[1], params[2], params[3])

    # 计算总平方和和回归平方和
    total_sum_of_squares = np.sum((y - np.mean(y))**2)
    regression_sum_of_squares = np.sum((cubic_function(x, params[0], params[1], params[2], params[3]) - np.mean(y))**2)

    # 计算R平方
    r_squared = regression_sum_of_squares / total_sum_of_squares

    # 在图像右下角显示三次多项式方程和R2
    equation = f'Y = {params[0]:.5f} * X^3 + {params[1]:.4f} * X^2 + {params[2]:.4f} * X + {params[3]:.4f}'
    r_squared_text = f'R2 = {r_squared:.4f}'
    plt.text(0.05, 0.8, equation, fontsize=7, transform=plt.gca().transAxes)
    plt.text(0.05, 0.75, r_squared_text, fontsize=7, transform=plt.gca().transAxes)

    # 保存图像到指定目录下，保存名称为列名.png
    save_path = 'G:/数据/新方法预处理/Albers投影/重采样1KM/中国裁剪/鹤岗/拟合趋势图/三次多项式/'
    os.makedirs(save_path, exist_ok=True)
    save_name = f'{col}.png'
    save_file = os.path.join(save_path, save_name)
    plt.savefig(save_file, dpi=150)

    # 清除当前图像，准备绘制下一个拟合结果图
    plt.clf()
