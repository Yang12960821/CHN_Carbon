import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 指定数据文件夹路径和图像保存路径
data_dir = r'D:\矫正完成夜间灯光数据连续\华北\Excle'  # 数据文件夹路径
output_dir = r'D:\矫正完成夜间灯光数据连续\华北\Excle\图像'  # 图像保存路径
os.makedirs(output_dir, exist_ok=True)  # 创建图像保存路径（如果不存在）

# 遍历数据文件夹下的所有xlsx文件
for filename in os.listdir(data_dir):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(data_dir, filename)
        file_name = os.path.splitext(filename)[0]  # 提取文件名（不包含扩展名）

        # 读取xlsx表格
        df = pd.read_excel(file_path)

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

        # 计算 R-squared 值
        residuals = y - y_fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - (ss_res / ss_tot)

        # 绘制散点图和拟合曲线
        plt.scatter(x, y, label='Data')
        plt.plot(x, y_fit, color='red', label='Fit')
        plt.xlabel('有效像素总和')
        plt.ylabel('统计碳排放(万吨)')
        plt.title('散点图和拟合曲线')

        # 设置横坐标刻度为原始数值
        # plt.xticks(x)

        # 显示拟合方程和R-squared值
        equation = f'拟合方程: y = {a:.4e} * x^3 + {b:.4e} * x^2 + {c:.4e} * x'
        r2_text = f'R-squared = {r2:.4f}'
        plt.text(0.02, 0.95, equation, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top')
        plt.text(0.02, 0.90, r2_text, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top')

        # 不显示图例
        plt.legend().remove()

        # 保存图像文件
        image_path = os.path.join(output_dir, f'{file_name}.png')
        plt.savefig(image_path, dpi=300)  # 设置dpi参数为300，保存为高清图像

        # 清除当前图形
        plt.clf()

        # 打印处理完成的文件名
        print(f'处理完成：{filename}')

print('所有文件处理完成。')