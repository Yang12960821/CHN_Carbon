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

shap_values_all = explainer.shap_values(X_test)

# 可视化单个预测的SHAP值，不使用JavaScript

shap.force_plot(explainer.expected_value, shap_values[1, :], X_test[1, :], matplotlib=True, feature_names=feature_names, show = False)
output_image_path = os.path.join(output_image_folder, '第1个样本力图.png')
plt.savefig(output_image_path)

shap.force_plot(explainer.expected_value, shap_values[6, :], X_test[6, :], matplotlib=True, feature_names=feature_names, show = False)
output_image_path = os.path.join(output_image_folder, '第7个样本力图.png')
plt.savefig(output_image_path)

shap.dependence_plot("NBI", shap_values, X_test, feature_names=feature_names, interaction_index=None, show = False)
output_image_path = os.path.join(output_image_folder, 'NBI依赖图.png')
plt.savefig(output_image_path)

shap.dependence_plot("Chl", shap_values, X_test, feature_names=feature_names, interaction_index=None, show = False)
output_image_path = os.path.join(output_image_folder, 'Chl依赖图.png')
plt.savefig(output_image_path)

# 计算R2值
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# 绘制验证结果的图表
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(y_test, label='actual value', color='blue', marker='o')
plt.plot(y_pred, label='predictive value', color='red', marker='x')
plt.xlabel('Sample number')
plt.ylabel('I-value')
plt.title('Random Forest model validation results\nR2 = {:.4f}'.format(r2))
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('actual value')
plt.ylabel('predictive value')
plt.title('Random Forest model validation scatter plot\nR2 = {:.4f}'.format(r2))

# 保存验证结果的图片
output_image_path = os.path.join(output_image_folder, 'Random_Forest_Model_Validation.png')
plt.savefig(output_image_path)

plt.tight_layout()
plt.show()

# 创建输出xlsx表格
output_xlsx_path = os.path.join(output_image_folder, 'Random_Forest_Model_Validation.xlsx')
df_validation = pd.DataFrame({'实际值': y_test.flatten(), '预测值': y_pred.flatten()})
df_validation.to_excel(output_xlsx_path, index=False)

print("验证结果图片已保存至:", output_image_path)
print("验证结果表格已保存至:", output_xlsx_path)