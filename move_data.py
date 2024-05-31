# import os
# import shutil
# import random

# def split_dataset(source_dirs, dest_dir, ratio):
#     # 创建目标文件夹
#     for split in ['train', 'train_GT', 'val', 'val_GT', 'test', 'test_GT']:
#         split_dir = os.path.join(dest_dir, split)
#         if not os.path.exists(split_dir):
#             os.makedirs(split_dir)

#     for category in source_dirs:
#         high_quality_dir = os.path.join(category, 'high_quality')
#         low_quality_dir = os.path.join(category, 'low_quality')
        
#         # 获取所有文件名
#         files = os.listdir(high_quality_dir)
        
#         # 打乱文件顺序
#         random.shuffle(files)
        
#         # 计算每个数据集的数量
#         total_files = len(files)
#         train_size = int(total_files * ratio['train'])
#         val_size = int(total_files * ratio['val'])
        
#         # 分配文件到训练集
#         for file in files[:train_size]:
#             shutil.move(os.path.join(high_quality_dir, file), os.path.join(dest_dir, 'train_GT', file))
#             shutil.move(os.path.join(low_quality_dir, file), os.path.join(dest_dir, 'train', file))
        
#         # 分配文件到验证集
#         for file in files[train_size:train_size + val_size]:
#             shutil.move(os.path.join(high_quality_dir, file), os.path.join(dest_dir, 'val_GT', file))
#             shutil.move(os.path.join(low_quality_dir, file), os.path.join(dest_dir, 'val', file))
        
#         # 分配文件到测试集
#         for file in files[train_size + val_size:]:
#             shutil.move(os.path.join(high_quality_dir, file), os.path.join(dest_dir, 'test_GT', file))
#             shutil.move(os.path.join(low_quality_dir, file), os.path.join(dest_dir, 'test', file))

# # 使用示例
# source_dirs = ['train_datasets/breast', 'train_datasets/carotid', 
#                'train_datasets/kidney', 'train_datasets/liver', 'train_datasets/thyroid']
# dest_dir = 'dataset'
# ratio = {'train': 0.7, 'val': 0.2, 'test': 0.1}

# split_dataset(source_dirs, dest_dir, ratio)

import os

def generate_meta_info(dataset_dir, output_file):
    with open(output_file, 'w') as f:
        for root, _, files in os.walk(dataset_dir):
            for file in sorted(files):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    relative_path = os.path.relpath(os.path.join(root, file), dataset_dir)
                    f.write(f"{relative_path}\n")
    print("out")
# 生成 train 和 val 的 meta_info 文件
# generate_meta_info('dataset/train_GT', 'dataset/train/meta_info.txt')
# generate_meta_info('dataset/val_GT', 'dataset/val/meta_info.txt')
# generate_meta_info('dataset/test_GT', 'dataset/test_meta_info.txt')
generate_meta_info('test/breast/gt', 'test/breast_meta_info.txt')
generate_meta_info('test/carotid/gt', 'test/carotid_meta_info.txt')
generate_meta_info('test/kidney/gt', 'test/kidney_meta_info.txt')
generate_meta_info('test/liver/gt', 'test/liver_meta_info.txt')
generate_meta_info('test/thyroid/gt', 'test/thyroid_meta_info.txt')