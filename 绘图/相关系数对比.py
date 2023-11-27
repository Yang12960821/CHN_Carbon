import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
data = pd.read_excel('G:/数据/新方法预处理/Albers投影/重采样1KM/中国裁剪/鹤岗/excle/r2.xlsx')

# 提取x轴和y轴数据
x = data.iloc[:, 0]  # 第一列为x轴
y = data.iloc[:, 1:]  # 从第二列开始的所有列为y轴

# 设置不同形状的标记
markers = ['o', 's', 'D', '^', 'v']  # 可根据需要添加更多形状

# 创建更大的图像
plt.figure(figsize=(10, 6))  # 调整图像大小，单位为英寸

# 绘制折线图并标注形状
for i in range(len(y.columns)):
    plt.plot(x, y.iloc[:, i], marker=markers[i % len(markers)])

# 设置y轴范围为0.2到1
plt.ylim(0.2, 1)

# 添加标题和轴标签
save_path = 'G:/数据/新方法预处理/Albers投影/重采样1KM/中国裁剪/鹤岗/拟合趋势图/相关系数对比图.png'
plt.title('相关系数对比图', fontproperties='SimHei')
#plt.xlabel('x')
plt.ylabel('相关系数', fontproperties='SimHei')

# 设置图例标签为列名
labels = y.columns
plt.legend(labels, prop={'family': 'SimHei'})

# 旋转x轴刻度标签
plt.xticks(rotation=45)  # 设置角度为45度

plt.savefig(save_path, dpi=300)

# 显示图表
plt.show()

