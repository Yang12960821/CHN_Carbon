import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
import numpy as np
import rasterio

# 读取Excel表格数据
input_file = r'D:\矫正完成夜间灯光数据连续\西北\Excle\PSOBP青海.xlsx'
df = pd.read_excel(input_file)

# 分割输入和输出数据
X = df[['年份', 'DN值']].values
y = df['碳排放量'].values

# 数据归一化
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y.reshape(-1, 1)
y = scaler.fit_transform(y).flatten()

# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义IPSO-BP模型
def pso_optimizer(params, X_train, y_train, X_test, y_test, scaler):
    learning_rate_init, alpha, inertia_weight = params
    mlp = MLPRegressor(hidden_layer_sizes=(30,), activation='relu', max_iter=50,
                       random_state=42, learning_rate_init=learning_rate_init, alpha=alpha)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    r2 = r2_score(y_test, y_pred)
    return -r2


# 使用IPSO算法优化参数
from pyswarm import pso

lb = [0.001, 0.00001, 0.1]  # 每个参数的下界
ub = [1, 1, 1]  # 每个参数的上界
x_opt, _ = pso(pso_optimizer, lb, ub, args=(X_train, y_train, X_test, y_test, scaler), swarmsize=10, maxiter=10)


# 使用优化后的参数构建最终的IPSO-BP模型
final_mlp = MLPRegressor(hidden_layer_sizes=(30,), activation='relu', max_iter=50,
                         random_state=42, learning_rate_init=x_opt[0], alpha=x_opt[1])

final_mlp.fit(X_train, y_train)

# 验证模型并绘制验证结果
y_pred = final_mlp.predict(X_test)
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))  # 将y_pred转换为2D数组
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))  # 将y_test转换为2D数组

# 计算R2值
r2 = r2_score(y_test, y_pred)

plt.figure(figsize=(12, 5))

# 绘制验证结果的折线图
plt.subplot(1, 2, 1)
plt.plot(y_test, label='实际值', color='blue', marker='o')
plt.plot(y_pred, label='预测值', color='red', marker='x')
plt.xlabel('样本序号')
plt.ylabel('碳排放量')
plt.title('IPSO-BP模型验证结果\nR2 = {:.4f}'.format(r2))
plt.legend()

# 绘制验证结果的散点图
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('IPSO-BP模型验证散点图\nR2 = {:.4f}'.format(r2))

# 输出图片到指定路径
output_image_folder = r'D:\矫正完成夜间灯光数据连续\西北\Excle\PSOBP青海\output'
os.makedirs(output_image_folder, exist_ok=True)
output_image_path = os.path.join(output_image_folder, 'IPSO_BP_Model_Validation.png')
plt.savefig(output_image_path)

plt.tight_layout()
plt.show()

# 创建输出xlsx表格
output_xlsx_path = os.path.join(output_image_folder, 'IPSO_BP_Model_Validation.xlsx')
df_validation = pd.DataFrame({'实际值': y_test.flatten(), '预测值': y_pred.flatten()})
df_validation.to_excel(output_xlsx_path, index=False)

print("验证结果图片已保存至:", output_image_path)
print("验证结果表格已保存至:", output_xlsx_path)

# 指定文件夹路径
folder_path = r'D:\矫正完成夜间灯光数据连续\西南\西藏自治区'

# 定义关键字列表
keywords = [str(year) for year in range(2000, 2021)]

# 获取文件夹下所有tif文件的路径
tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif') and any(keyword in f for keyword in keywords)]

# 循环处理每个年份的tif文件
for year in range(2000, 2021):
    # 查找对应年份的tif文件
    tif_file = next((f for f in tif_files if str(year) in f), None)

    if tif_file is not None:
        # 读取TIFF文件
        input_tif_file = os.path.join(folder_path, tif_file)
        with rasterio.open(input_tif_file) as dataset:
            band = dataset.read(1, masked=True)  # 读取第一个波段的数据，并将NoData值标记为Masked
            metadata = dataset.profile  # 获取原始TIFF文件的metadata信息

        # 获取TIFF文件的像元值作为输入的DN值
        dn_values = band.data.flatten()

        # 获取Nodata值所在的像元索引
        nodata_indexes = np.where(band.mask.flatten())

        # 构建输入数据，排除Nodata值对应的像元点
        input_data = np.vstack([np.ones(len(dn_values)) * year, dn_values]).T
        input_data = np.delete(input_data, nodata_indexes, axis=0)

        # 使用IPSO-BP模型预测每个像元点的碳排放量
        carbon_emissions = final_mlp.predict(input_data)

        # 计算有效像元的最小值
        min_value = np.nanmin(carbon_emissions)

        # 将预测结果中的最小值减去所有像元值（排除Nodata值）
        carbon_emissions = carbon_emissions - min_value

        # 创建输出栅格图像并将预测结果写入其中
        predicted_array = np.ones(band.shape, dtype=np.float32) * band.fill_value
        predicted_array[~band.mask] = carbon_emissions

        # 创建输出栅格图像并将预测结果写入其中
        output_folder = r'D:\矫正完成夜间灯光数据连续\西南\西藏自治区\碳排放\IPSO'
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, '校正前' + tif_file)
        with rasterio.open(output_file, 'w', **metadata) as dst:
            dst.write(predicted_array, 1)

        print("完成", year)

print("所有栅格图像预测结果已保存至:", output_folder)
