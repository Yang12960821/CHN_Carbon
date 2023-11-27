import arcpy
import numpy as np
import os
import csv

# 设置工作环境
arcpy.env.workspace = r"D:\L2正则化三次函数拟合\东北\黑龙江省\碳排放"

# 获取指定目录下的所有TIF文件
tif_files = arcpy.ListRasters("*.tif")

# 删除已经存在的临时表格
if arcpy.Exists("in_memory\\temp_table"):
    arcpy.Delete_management("in_memory\\temp_table")

# 创建一个临时表格
temp_table = r"in_memory\temp_table"
arcpy.CreateTable_management("in_memory", "temp_table")
arcpy.AddField_management(temp_table, "文件名", "TEXT")
arcpy.AddField_management(temp_table, "有效像素总和", "DOUBLE")

# 插入数据到临时表格
with arcpy.da.InsertCursor(temp_table, ["文件名", "有效像素总和"]) as cursor:
    for tif_file in tif_files:
        # 将栅格图像转换为 NumPy 数组
        raster = arcpy.Raster(tif_file)
        raster_array = arcpy.RasterToNumPyArray(raster)
        
        # 获取栅格图像的 NoData 值
        no_data_value = raster.noDataValue
        
        # 计算栅格图像的总和（排除 NoData 值）
        total_sum = np.sum(raster_array[raster_array != no_data_value])
        
        # 插入数据到临时表格
        cursor.insertRow([os.path.basename(tif_file), total_sum])

# 保存临时表格为CSV文件
output_csv = r"D:\EXCLE\jilin.csv"
with open(output_csv, 'w') as csvfile:
    writer = csv.writer(csvfile)
    
    # 写入标题行
    writer.writerow(["文件名", "有效像素总和"])
    
    # 遍历临时表格的行并写入CSV文件
    with arcpy.da.SearchCursor(temp_table, ["文件名", "有效像素总和"]) as search_cursor:
        for row in search_cursor:
            writer.writerow(row)

print("计算完成并已保存为", output_csv)
