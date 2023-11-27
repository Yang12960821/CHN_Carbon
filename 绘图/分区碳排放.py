import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
data = pd.read_excel('D:\矫正完成夜间灯光数据连续\EXCLE\分区域碳排放.xlsx')

# 提取x轴和y轴数据
x = data.iloc[:, 0]  # 第一列为x轴
y = data.iloc[:, 1:]  # 从第二列开始的所有列为y轴

# 设置不同形状的标记
markers = ['o', 's', 'D', '^', 'v', 'p', 'h']  # 可根据需要添加更多形状

# 创建更大的图像
plt.figure(figsize=(10, 6))  # 调整图像大小，单位为英寸

# 绘制折线图并标注形状
#for i in range(len(y.columns)):
#    lines = plt.plot(x, y.iloc[:, i], marker=markers[i % len(markers)])

# 绘制折线图
lines = plt.plot(x, y)

# 设置y轴范围为0到1
#plt.ylim(0.2, 1)

# 添加标题和轴标签
save_path = 'D:\矫正完成夜间灯光数据连续\出图\分区碳排放图.png'
plt.title('分区碳排放图', fontproperties='SimHei')
#plt.xlabel('x')
plt.ylabel('碳排放量(Gt)', fontproperties='SimHei')

# 旋转x轴刻度标签
#plt.xticks(rotation=45)  # 设置角度为45度


# 设置图例标签为列名
labels = y.columns
plt.legend(lines, labels, prop={'family': 'SimHei'})

plt.savefig(save_path, dpi=300)

# 显示图表
plt.show()
