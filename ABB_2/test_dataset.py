from dataset import AngleDataset


# 创建数据集对象

dataset = AngleDataset(

    csv_file="data/train/Exp001/train_labels.csv",

    image_dir="data/train/Exp001/frames"

)


# 查看数据数量

print("数据数量:")

print(len(dataset))


# 获取第0个数据

image, label = dataset[0]


print("\n第一张图片:")

print(image)


print("\n对应六轴角度:")

print(label)