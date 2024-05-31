import os
import shutil

# 定义路径
base_dir = 'Simple-Align-main'
original_test_dir = os.path.join(base_dir, 'dataset', 'test')
original_test_gt_dir = os.path.join(base_dir, 'dataset', 'test_GT')
new_test_dir = os.path.join(base_dir, 'test')
train_datasets_dir = os.path.join(base_dir, 'train_datasets')

# 创建新目录结构
categories = ['breast', 'carotid', 'kidney', 'liver', 'thyroid']
for category in categories:
    os.makedirs(os.path.join(new_test_dir, category, 'lq'), exist_ok=True)
    os.makedirs(os.path.join(new_test_dir, category, 'gt'), exist_ok=True)

# 复制文件
for category in categories:
    lq_path = os.path.join(train_datasets_dir, category, 'low_quality')
    gt_path = os.path.join(train_datasets_dir, category, 'high_quality')
    
    for filename in os.listdir(original_test_dir):
        if filename in os.listdir(lq_path):
            shutil.copy(os.path.join(original_test_dir, filename), os.path.join(new_test_dir, category, 'lq', filename))
    
    for filename in os.listdir(original_test_gt_dir):
        if filename in os.listdir(gt_path):
            shutil.copy(os.path.join(original_test_gt_dir, filename), os.path.join(new_test_dir, category, 'gt', filename))

print("文件已成功复制和分类。")
