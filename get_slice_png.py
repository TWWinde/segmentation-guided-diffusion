import numpy as np
from PIL import Image
import os
import nibabel as nib

def save_slices_as_png(array_3d, output_dir, name):
    # 1. 加载 .npy 文件
    print(array_3d.shape)
    # 检查 array_3d 的形状，确保它是一个三维矩阵
    if len(array_3d.shape) != 3:
        raise ValueError("输入的 NumPy 数组必须是三维的。")

    # 获取三维矩阵的维度 (depth, height, width)
    depth, height, width = array_3d.shape

    # 创建输出目录，如果不存在的话
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 2. 遍历 depth 维度，逐层切片
    for i in range(depth):
        # 获取第 i 个切片 (二维切片)
        data = array_3d[i, :, :]
        """
        unique_values = np.unique(data)
        print("Unique pixel values in the NIfTI image:")
        print(unique_values)
        print(data.shape)
        print(data.dtype)
        print(data.min())
        print(data.max())
        """
        img = Image.fromarray(data, mode='L')

        img.save(os.path.join(output_dir, f"ts_{i+name*32}.png"))

    print(f"保存了 {depth} 张切片图像到 {output_dir}.")


if __name__ == "__main__":
    root_path = "/data/private/autoPET/autopet_3d/test/label"
    label_path = os.listdir(root_path)
    output_root = "/data/private/autoPET/medicaldiffusion_results/test_results/ddpm/AutoPET/output_with_segconv_64out/video_results/mask/all/test"
    os.makedirs(output_root, exist_ok=True)
    for item in sorted(label_path):
        name = int(item.split("_")[0])
        path = os.path.join(root_path, item)
        img = nib.load(path)
        data = img.get_fdata()
        #arrray = np.squeeze(arrray, axis=0)
        #arrray = np.squeeze(arrray, axis=0)
        save_slices_as_png(data, output_root, name)
