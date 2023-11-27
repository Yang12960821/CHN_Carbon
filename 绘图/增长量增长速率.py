import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取XLSX文件
df = pd.read_excel(r'D:\矫正完成夜间灯光数据连续\EXCLE\西南增长速率.xlsx')

# 创建图表，并指定figsize来调整尺寸
fig, ax1 = plt.subplots(figsize=(8, 6))  # 调整宽度和高度

# 绘制柱状图
bar = ax1.bar(df['年份'], df['碳排放量'], color='b', alpha=0.7, label='碳排放量')
ax1.set_xlabel('年份', fontsize=13)
ax1.set_ylabel('碳排放量(Gt)', fontsize=13)
ax1.tick_params(axis='y', labelsize=13)  # 更改左边y轴刻度字体大小

# 创建第二个y轴，绘制折线图
ax2 = ax1.twinx()
line, = ax2.plot(df['年份'], df['增长速率'], 'r-o', label='增长速率')
ax2.set_ylabel('增长率(%)', fontsize=13)
ax2.tick_params(axis='y', labelsize=13)  # 更改右边y轴刻度字体大小

# 合并两个图例
handlers, labels = [], []
for ax in [ax1, ax2]:
    h, l = ax.get_legend_handles_labels()
    handlers.extend(h)
    labels.extend(l)

# 添加图例
ax1.legend(handles=handlers, labels=labels, loc='upper left', fontsize=14)

# 设置标题
#plt.title('碳排放量和增长速率')

# 自定义坐标轴刻度字体大小
plt.xticks(fontsize=15)  # 设置x轴刻度字体大小为15
plt.yticks(fontsize=13)  # 设置y轴刻度字体大小为13

# 保存图表到指定位置，设置dpi参数为300
plt.savefig('D:\矫正完成夜间灯光数据连续\PSO最终结果\出图\西南增长速率.png', dpi=300)  # 修改路径和文件名

# 显示图表
plt.show()
