import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import rcParams
from statistics import mean
from sklearn.metrics import explained_variance_score, r2_score, median_absolute_error, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os


plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取数据
df = pd.read_excel(r'D:\矫正完成夜间灯光数据连续\验证\地级市\2005验证.xlsx')
x = df['文献碳排放']
y = df['反演碳排放']

# 计算散点密度
xy = np.vstack([x, y])
z = stats.gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x.iloc[idx], y.iloc[idx], z[idx]

# 拟合线性回归
def calculate_regression_line(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) / ((mean(xs) * mean(xs)) - mean(xs * xs)))
    b = mean(ys) - m * mean(xs)
    return m, b

k, b = calculate_regression_line(x, y)
regression_line = [(k * a) + b for a in x]

# 计算统计指标
BIAS = mean(x - y)
MSE = mean_squared_error(x, y)
RMSE = np.power(MSE, 0.5)
R2 = pearsonr(x, y)[0] ** 2  # 使用pearsonr函数计算R2
adjR2 = 1 - ((1 - r2_score(x, y)) * (len(x) - 1)) / (len(x) - 5 - 1)
MAE = mean_absolute_error(x, y)
EV = explained_variance_score(x, y)
NSE = 1 - (RMSE ** 2 / np.var(x))

# 设置字体
rcParams.update({"font.family": "Times New Roman", "font.size": 16, "mathtext.fontset": "stix"})

# 创建图
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# 绘制散点图
scatter = ax.scatter(x, y, marker='o', c=z * 100, edgecolors=None, s=15, cmap='RdBu_r',  alpha=0.8)
cbar = plt.colorbar(scatter, shrink=1, orientation='vertical', extend='both', pad=0.015, aspect=30, label='Frequency')

# 绘制回归线
plt.plot(x, regression_line, 'black', lw=1.5, label='Regression Line')

# 添加标签和标题
plt.xlabel('Chen et al carbon emissions')
plt.ylabel('Simulated carbon emissions')
#plt.title('散点图与线性回归\n$R^2=%.3f$, $NSE=%.3f$, $MAE=%.3f$, $RMSE=%.3f$' % (R2, NSE, MAE, RMSE))

# 绘制1:1线
scale = max(max(x), max(y))
plt.plot([-scale, scale], [-scale, scale], 'red', lw=1.5, linestyle='--', label='1:1 Line')
# 设置x轴和y轴的范围从0开始
plt.xlim(0, scale)
plt.ylim(0, scale)

# 添加统计指标文本
plt.text(scale * 0.95, scale * 0.06, f'$R^2 = {R2:.3f}$', family='Times New Roman', horizontalalignment='right')
#plt.text(scale * 0.95, scale * 0.01, f'$NSE = {NSE:.3f}$', family='Times New Roman', horizontalalignment='right')
plt.text(scale * 0.95, scale * 0.11, f'$MAE = {MAE:.3f}$', family='Times New Roman', horizontalalignment='right')
plt.text(scale * 0.95, scale * 0.16, f'$RMSE = {RMSE:.3f}$', family='Times New Roman', horizontalalignment='right')

# 显示图例
ax.legend(loc='upper left', frameon=False)

# 创建输出图片文件夹
output_image_folder = r'D:\矫正完成夜间灯光数据连续\PSO最终结果\出图'
os.makedirs(output_image_folder, exist_ok=True)

# 保存验证结果的图片
output_image_path = os.path.join(output_image_folder, '2005验证1.png')
plt.savefig(output_image_path)

# 显示图形
plt.show()