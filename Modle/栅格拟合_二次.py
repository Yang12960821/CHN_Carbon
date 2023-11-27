import numpy as np
from osgeo import gdal
from scipy.optimize import curve_fit

# 读取栅格图像数据
def read_raster(raster_path):
    dataset = gdal.Open(raster_path)
    band = dataset.GetRasterBand(1)  # 假设只有一个波段
    raster_array = band.ReadAsArray()
    return dataset, band, raster_array

# 定义一元二次方程模型
def quadratic_func(x, a, b, c):
    return a * x**2 + b * x + c

# 输入图像路径
image1_path = r'G:/数据/新方法预处理/Albers投影/重采样1KM/中国裁剪/饱和矫正之后/连续性矫正/最终/重新裁剪/D/DChina_2013.tif'
image2_path = r'G:/数据/NPPVIIRS_中国范围逐月夜光影像_2012-2021.7/月度/去除背景噪声/去除异常值（邻域法）/NPPChina2013.tif'

# 读取图像数据
dataset1, band1, image1_data = read_raster(image1_path)
dataset2, band2, image2_data = read_raster(image2_path)

# 找到 NoData 值
nodata_value1 = band1.GetNoDataValue()
nodata_value2 = band2.GetNoDataValue()

# 提取有效像素值
valid_pixels = (image1_data != nodata_value1) & (image2_data != nodata_value2)
x_data = image1_data[valid_pixels].flatten()
y_data = image2_data[valid_pixels].flatten()

# 拟合曲线
params, _ = curve_fit(quadratic_func, x_data, y_data)

# 提取回归系数
a, b, c = params

# 计算预测值
y_pred = quadratic_func(x_data, a, b, c)

# 计算R2值
ss_total = np.sum((y_data - np.mean(y_data))**2)
ss_residual = np.sum((y_data - y_pred)**2)
r_squared = 1 - (ss_residual / ss_total)

# 输出回归方程和R2值
print("回归方程：y = {:.2f}x^2 + {:.2f}x + {:.2f}".format(a, b, c))
print("R2值：{:.2f}".format(r_squared))

# 关闭数据集
band1 = None
band2 = None
dataset1 = None
dataset2 = None
