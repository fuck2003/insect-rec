import os
import shutil
import random

def create_dataset_split(source_dir, train_dir, val_dir, test_dir, train_ratio=0.8, val_ratio=0.1):
    # 创建训练集、验证集和测试集目录
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    classes = os.listdir(source_dir)

    # 创建标签文件
    train_label_file = open(os.path.join(source_dir, 'train.txt'), 'w')
    val_label_file = open(os.path.join(source_dir, 'val.txt'), 'w')
    test_label_file = open(os.path.join(source_dir, 'test.txt'), 'w')

    for class_name in classes:
        class_dir = os.path.join(source_dir, class_name)
        if os.path.isdir(class_dir):
            # 列出该类别下的所有文件
            files = os.listdir(class_dir)
            random.shuffle(files)  # 随机打乱文件顺序
            
            # 计算分割点
            total_files = len(files)
            train_count = int(total_files * train_ratio)
            val_count = int(total_files * val_ratio)

            # 分割文件
            train_files = files[:train_count]
            val_files = files[train_count:train_count + val_count]
            test_files = files[train_count + val_count:]

            # 创建类别文件夹
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

            # 移动文件到对应的文件夹并记录文件
            for file_name in train_files:
                shutil.copy(os.path.join(class_dir, file_name),
                            os.path.join(train_dir, class_name, file_name))
                train_label_file.write(f"{class_name}/{file_name}\n")  # 记录训练文件

            for file_name in val_files:
                shutil.copy(os.path.join(class_dir, file_name),
                            os.path.join(val_dir, class_name, file_name))
                val_label_file.write(f"{class_name}/{file_name}\n")  # 记录验证文件

            for file_name in test_files:
                shutil.copy(os.path.join(class_dir, file_name),
                            os.path.join(test_dir, class_name, file_name))
                test_label_file.write(f"{class_name}/{file_name}\n")  # 记录测试文件

    # 关闭标签文件
    train_label_file.close()
    val_label_file.close()
    test_label_file.close()

source_directory = r"C:\Users\GUOXI\Desktop\kunchong\dataset\farm_insects"
train_directory = r"C:\Users\GUOXI\Desktop\kunchong\dataset\train"
val_directory = r"C:\Users\GUOXI\Desktop\kunchong\dataset\val"
test_directory = r"C:\Users\GUOXI\Desktop\kunchong\dataset\test"

create_dataset_split(source_directory, train_directory, val_directory, test_directory)
