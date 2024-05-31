import os

def rename_files_in_directory(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if '_test_23' in file:
                new_name = file.replace('_test_23', '')
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, new_name)
                os.rename(old_path, new_path)
                print(f'Renamed: {old_path} -> {new_path}')

# 主文件夹路径
base_dir = '/home/user3/project/other/cv/Simple-Align-main/results/test_23/visualization/mix'

# 调用函数重命名文件
rename_files_in_directory(base_dir)
