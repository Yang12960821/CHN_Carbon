import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
import rasterio
import numpy as np

# 读取Excel表格数据
input_file = r'D:\矫正完成夜间灯光数据连续\西北\Excle\PSOBP新疆.xlsx'
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

# 定义SVR模型
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)

# 训练SVR模型
svr.fit(X_train, y_train)

# 验证模型并绘制验证结果
y_pred = svr.predict(X_test)
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
plt.title('SVR模型验证结果\nR2 = {:.4f}'.format(r2))
plt.legend()

# 绘制验证结果的散点图
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('SVR模型验证散点图\nR2 = {:.4f}'.format(r2))

# 输出图片到指定路径
output_image_folder = r'D:\矫正完成夜间灯光数据连续\西北\新疆维吾尔自治区\碳排放\SVR\xlsx'
os.makedirs(output_image_folder, exist_ok=True)
output_image_path = os.path.join(output_image_folder, 'SVR_Model_Validation.png')
plt.savefig(output_image_path)

plt.tight_layout()
plt.show()

# 创建输出xlsx表格
output_xlsx_path = os.path.join(output_image_folder, 'SVR_Model_Validation.xlsx')
df_validation = pd.DataFrame({'实际值': y_test.flatten(), '预测值': y_pred.flatten()})
df_validation.to_excel(output_xlsx_path, index=False)

print("验证结果图片已保存至:", output_image_path)
print("验证结果表格已保存至:", output_xlsx_path)

# 以下部分保持不变，仍然使用SVR模型进行栅格图像预测

# 指定文件夹路径
folder_path = r'D:\矫正完成夜间灯光数据连续\西北\新疆维吾尔自治区'

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

        # 使用SVR模型预测每个像元点的碳排放量
        carbon_emissions = svr.predict(input_data)

        # 计算有效像元的最小值
        min_value = np.nanmin(carbon_emissions)

        # 将预测结果中的最小值减去所有像元值（排除Nodata值）
        #carbon_emissions = carbon_emissions - min_value

        # 创建输出栅格图像并将预测结果写入其中
        predicted_array = np.ones(band.shape, dtype=np.float32) * band.fill_value
        predicted_array[~band.mask] = carbon_emissions

        # 创建输出栅格图像并将预测结果写入其中
        output_folder = r'D:\矫正完成夜间灯光数据连续\西北\新疆维吾尔自治区\碳排放\SVR'
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, '校正前' + tif_file)
        with rasterio.open(output_file, 'w', **metadata) as dst:
            dst.write(predicted_array, 1)

        print("完成", year)

print("所有栅格图像预测结果已保存至:", output_folder)
