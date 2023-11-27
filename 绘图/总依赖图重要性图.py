import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import os
import shap
from PIL import Image

# 读取Excel表格数据
input_file = r'D:\LIA\TEST4.xlsx'
df = pd.read_excel(input_file)

# 创建输出图片文件夹
output_image_folder = r'D:\LIA\Output'
os.makedirs(output_image_folder, exist_ok=True)

# 提取输入特征和输出
X = df[['NBI', 'Chl', 'Flav', 'Anth', 'PWC', 'F']].values
y = df['D'].values

# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 构建随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练随机森林模型
model.fit(X_train, y_train)

# 创建SHAP解释器
explainer = shap.TreeExplainer(model)

# 计算SHAP值
shap_values = explainer.shap_values(X_test)

# 定义特征名称列表
feature_names = ['NBI', 'Chl', 'Flav', 'Anth', 'PWC', 'F']

# 将 SHAP 值四舍五入到3位小数
#shap_values_rounded = [shap_values[i].round(3) for i in range(shap_values.shape[0])]

shap_values_all = explainer.shap_values(X_test)

# 创建摘要图，显示特征名称
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# 添加特征重要性的条形图
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=feature_names)
