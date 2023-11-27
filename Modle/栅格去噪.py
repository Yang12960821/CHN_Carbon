import numpy as np
import rasterio

# 输入和输出文件路径
input_file = "G:/数据/NPPVIIRS_中国范围逐月夜光影像_2012-2021.7/月度/去除背景噪声/YNPP2020.tif"
output_file = "G:/数据/NPPVIIRS_中国范围逐月夜光影像_2012-2021.7/月度/去除背景噪声/去除异常值（邻域法）/NPPChina2020.tif"

# 定义阈值和邻域大小
threshold = 313.97
neighborhood_size = 1

# 读取栅格数据
with rasterio.open(input_file) as src:
    # 获取栅格数据的元数据
    profile = src.profile
    data = src.read(1)

    # 复制栅格数据用于处理
    processed_data = data.copy()

    # 像元计数器
    total_pixels = processed_data.size
    processed_pixels = 0

    # 迭代处理，直到没有大于阈值的像元
    while np.any(processed_data > threshold):
        for i in range(neighborhood_size, data.shape[0] - neighborhood_size):
            for j in range(neighborhood_size, data.shape[1] - neighborhood_size):
                if processed_data[i, j] > threshold:
                    neighborhood = data[i-neighborhood_size:i+neighborhood_size+1, j-neighborhood_size:j+neighborhood_size+1]
                    min_value = np.min(neighborhood)
                    processed_data[i, j] = min_value
                    processed_pixels += 1

    # 保存处理结果
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(processed_data, 1)

    # 打印像元统计信息
    print("总像元个数:", total_pixels)
    print("处理后的像元个数:", processed_pixels)
    print("处理完成")
