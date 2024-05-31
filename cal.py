import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import correlate
import cv2
import torch
import torch.nn.functional as F

def calculate_ssim(img1, img2):
    return ssim(img1, img2, multichannel=True, win_size=3)

def compute_LNCC(img1, img2, kernel_size=9):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size should be odd.")
    
    padding = kernel_size // 2
    img1 = F.pad(img1, [padding]*4, 'reflect')
    img2 = F.pad(img2, [padding]*4, 'reflect')

    # Compute local sums for img1, img2, img1^2, img2^2, and img1*img2
    sum_img1 = F.avg_pool2d(img1, kernel_size, stride=1)
    sum_img2 = F.avg_pool2d(img2, kernel_size, stride=1)
    sum_img1_sq = F.avg_pool2d(img1 * img1, kernel_size, stride=1)
    sum_img2_sq = F.avg_pool2d(img2 * img2, kernel_size, stride=1)
    sum_img1_img2 = F.avg_pool2d(img1 * img2, kernel_size, stride=1)
    
    # Compute mean and variance in local window for img1 and img2
    mean_img1 = sum_img1 / (kernel_size ** 2)
    mean_img2 = sum_img2 / (kernel_size ** 2)
    var_img1 = sum_img1_sq - mean_img1 ** 2
    var_img2 = sum_img2_sq - mean_img2 ** 2

    # Compute the local normalized cross-correlation
    ncc = (sum_img1_img2 - mean_img1 * mean_img2) / (torch.sqrt(var_img1 * var_img2) + 1e-5)
    
    return torch.mean(ncc).item()

def calculate_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def read_image(filepath):
    return cv2.imread(filepath, cv2.IMREAD_COLOR)

def calculate_metrics(generated_images_dir, gt_images_dir):
    ssim_scores = []
    lncc_scores = []
    mse_scores = []

    gen_files = os.listdir(generated_images_dir)
    gt_files = os.listdir(gt_images_dir)

    if not gen_files:
        print(f"No files found in generated_images_dir: {generated_images_dir}")
    if not gt_files:
        print(f"No files found in gt_images_dir: {gt_images_dir}")

    for filename in gen_files:
        gen_img_path = os.path.join(generated_images_dir, filename)
        gt_img_path = os.path.join(gt_images_dir, filename)
        
        if os.path.exists(gt_img_path):
            gen_img = read_image(gen_img_path)
            gt_img = read_image(gt_img_path)
            
            ssim_score = calculate_ssim(gen_img, gt_img)
            mse_score = calculate_mse(gen_img, gt_img)
            
            # 将图像转换为张量
            gen_img_tensor = torch.tensor(gen_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            gt_img_tensor = torch.tensor(gt_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            
            lncc_score = compute_LNCC(gen_img_tensor, gt_img_tensor)
            
            ssim_scores.append(ssim_score)
            lncc_scores.append(lncc_score)
            mse_scores.append(mse_score)
            
            print(f"{filename} - SSIM: {ssim_score}, LNCC: {lncc_score}, MSE: {mse_score}")
        else:
            print(f"GT image not found for: {filename}")

    # 计算均值和方差
    if ssim_scores:
        average_ssim = np.mean(ssim_scores)
        std_ssim = np.std(ssim_scores)
        average_lncc = np.mean(lncc_scores)
        std_lncc = np.std(lncc_scores)
        average_mse = np.mean(mse_scores)
        std_mse = np.std(mse_scores)
    else:
        average_ssim = std_ssim = average_lncc = std_lncc = average_mse = std_mse = float('nan')

    print(f"Average SSIM: {average_ssim}, Std SSIM: {std_ssim}")
    print(f"Average LNCC: {average_lncc}, Std LNCC: {std_lncc}")
    print(f"Average MSE: {average_mse}, Std MSE: {std_mse}")
list_nam = ['carotid', 'thyroid', 'kidney', 'liver', 'breast']
# 调用函数
name = "liver"
generated_images_dir = "/home/user3/project/other/cv/Simple-Align-main/results/test_23/visualization/mix" 
gt_images_dir = "/home/user3/project/other/cv/Simple-Align-main/dataset/test_GT"

print(name)
calculate_metrics(generated_images_dir, gt_images_dir)
# print(list_nam)