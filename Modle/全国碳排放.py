import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'

# 读取Excel数据
data = pd.read_excel(r'G:\数据\全国JS碳排放.xlsx')

# 获取列名
columns = data.columns

# 设置x轴和y轴的数据
x = data[columns[0]]
y = data[columns[2]]

# 计算趋势线的斜率和截距
slope, intercept = np.polyfit(x, y, 1)

# 创建折线图
plt.plot(x, y, ls='-', marker='s', linewidth=2.5)  # 将折线颜色设置为红色
plt.plot(x, slope*x + intercept, linestyle='--', color='red', linewidth=2.5)

# 设置标题
title = columns[2]
#plt.title(title, fontproperties=font, fontsize=16)  # 设置标题字体大小为16

# 设置x轴和y轴标签
plt.xlabel(columns[0], fontproperties=font, fontsize=20)  # 设置x轴标签字体大小为14
plt.ylabel(columns[2], fontproperties=font, fontsize=20)  # 设置y轴标签字体大小为14

# 设置x轴和y轴刻度字体大小
plt.xticks(fontsize=20)  # 设置x轴刻度字体大小为12
plt.yticks(fontsize=20)  # 设置y轴刻度字体大小为12

# 调整图像分辨率和dpi
fig = plt.gcf()
fig.set_size_inches(12, 8)  # 设置图像大小，单位为英寸
fig.set_dpi(300)  # 设置图像dpi

# 保存图片
plt.tight_layout()
save_path = r'D:\矫正完成夜间灯光数据连续\出图\\' + title + '.png'
plt.savefig(save_path, dpi=fig.get_dpi())

# 显示图形
plt.show()