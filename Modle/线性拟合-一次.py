import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# 读取xls文件
data = pd.read_excel('G:/数据/DM-NPP/5KMChina.xlsx')

# 提取第三列和第三列之后的其他列
y = data.iloc[:, 2]

# 遍历第三列之后的每一列进行线性拟合和绘图
for col in data.columns[3:]:
    x = data[col]

    # 进行线性拟合
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # 计算拟合直线上的点
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept

    # 绘制散点图和拟合直线
    plt.scatter(x, y, s=1, label='Data')
    plt.plot(x_fit, y_fit, 'r', label='Linear Fit')

    # 添加图例
    #plt.legend()

    # 在图像右下角显示趋势线方程和R2
    equation = f'Y = {slope:.4f}X + {intercept:.4f}'
    r_squared = f'R2 = {r_value**2:.4f}'
    plt.text(0.05, 0.8, equation, transform=plt.gca().transAxes)
    plt.text(0.05, 0.75, r_squared, transform=plt.gca().transAxes)

    # 保存图像到指定目录下，保存名称为列名.png
    save_path = 'G:/数据/DM-NPP/线性/'
    os.makedirs(save_path, exist_ok=True)
    save_name = f'{col}.png'
    save_file = os.path.join(save_path, save_name)
    plt.savefig(save_file, dpi=150)

    # 清除当前图像，准备绘制下一个拟合结果图
    plt.clf()
