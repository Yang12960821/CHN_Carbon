import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# 设置中文字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# 获取当前字体信息
print(fm.findfont('DejaVu Sans'))

# 实际值和预测值
df = pd.read_excel(r"G:/数据/中国/STIRPAT岭回归模型/实际预测对比1.xlsx")
x = df.iloc[:, 0].values
y1 = df.iloc[:, 1].values
y2 = df.iloc[:, 2].values

# 计算R2和P值
r2 = r2_score(y1, y2)
p = pearsonr(y1, y2)[0]

# 绘制对比图
plt.plot(x, y1,'o-', label='Actual')
plt.plot(x, y2,'s-',label='Predicted')

# 添加标题、标签和图例
plt.title('Actual vs Predicted')
plt.xlabel('Year')
plt.ylabel('CEADs (million tons of CO2)')
plt.legend()

# 设置横坐标间隔为2，保留为整数
plt.xticks(np.arange(min(x), max(x)+1, 2.0).astype(int))

# 将R2和P值添加到图像中
plt.text(0.8, 0.15, f"R2 = {r2:.3f}", transform=plt.gca().transAxes, fontsize=10)
plt.text(0.8, 0.1, f"P = {p:.3f}", transform=plt.gca().transAxes, fontsize=10)
plt.savefig('G:/数据/中国/STIRPAT岭回归模型/实际预测对比1.png', dpi=300)

plt.show()
