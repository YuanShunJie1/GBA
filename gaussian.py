import torch
import numpy as np
import matplotlib.pyplot as plt

# 定义类别数、标准差和均值间隔
num_classes = 10
std_dev = 1.0  # 所有类别相同的标准差
mean_interval = 6 * std_dev  # 均值之间的最小间隔

# 生成10个类别的均值
means = torch.arange(0, num_classes * mean_interval, mean_interval)

# 为每个类别生成高斯分布的曲线
x = torch.linspace(-10, 10 + means[-1].item(), 1000)
plt.figure(figsize=(10, 6))

# 绘制每个类别的高斯分布曲线
for i, mean in enumerate(means):
    gaussian = (1 / (std_dev * torch.sqrt(torch.tensor(2 * np.pi)))) * torch.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    plt.plot(x.numpy(), gaussian.numpy(), label=f'Class {i+1}')

# 图形美化
plt.title('Gaussian Distributions for 10 Classes (PyTorch)')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
# plt.grid(True)
# plt.show()
plt.savefig('gaussian_clustering.png')  # 可以保存为其他格式，如 'gaussian_clustering.pdf'

