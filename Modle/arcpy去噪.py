import arcpy

# 设置工作空间和输入输出路径
arcpy.env.workspace = r"G:\数据\NPPVIIRS_中国范围逐月夜光影像_2012-2021.7\月度\去除背景噪声"
input_raster = "FYNPPChina2013.tif"
output_raster = "NPPChina2013.tif"

# 定义阈值和邻域大小
threshold = 400
neighborhood_size = 3

# 提取大于阈值的像元
extract_expression = "VALUE > {}".format(threshold)
extracted_raster = arcpy.sa.ExtractByAttributes(input_raster, extract_expression)

# 迭代处理，直到没有大于阈值的像元
while arcpy.GetRasterProperties_management(extracted_raster, "MAXIMUM").getOutput(0) > threshold:
    max_raster = arcpy.sa.FocalStatistics(extracted_raster, arcpy.sa.NbrRectangle(neighborhood_size, neighborhood_size), "MAXIMUM")
    extracted_raster = arcpy.sa.Con(extracted_raster > threshold, max_raster, extracted_raster)

# 保存处理结果
extracted_raster.save(output_raster)

print("处理完成！")