import pandas as pd
import os


excel_path = r"data2/test/Exp002/Test_Angle2_real.xlsx"


print("文件存在吗:")
print(os.path.exists(excel_path))


df = pd.read_excel(
    excel_path
)


print("读取成功")


print("行数:")
print(len(df))


print("前5行:")
print(df.head())


print("列名:")
print(df.columns.tolist())
