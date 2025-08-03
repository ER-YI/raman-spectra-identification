# -*- coding: utf-8 -*-
import numpy as np
from scipy import interpolate
from sklearn.utils import resample
import matplotlib.pyplot as plt
import random

def add_noise(spectrum, noise_level=0.01):
    """
    向光谱中添加随机噪声
    
    参数:
    spectrum : array-like
        输入光谱强度数据
    noise_level : float
        噪声水平（相对于信号最大值的比例）
    
    返回:
    添加噪声后的光谱
    """
    noise = np.random.normal(0, noise_level * np.max(spectrum), len(spectrum))
    return spectrum + noise

def spectral_shift(spectrum, shift_range=(-5, 5)):
    """
    对光谱进行波长偏移
    
    参数:
    spectrum : array-like
        输入光谱强度数据
    shift_range : tuple
        偏移范围（最小值，最大值）
    
    返回:
    偏移后的光谱
    """
    shift = np.random.uniform(shift_range[0], shift_range[1])
    # 使用插值实现偏移
    indices = np.arange(len(spectrum))
    shifted_indices = indices + shift
    
    # 确保索引在有效范围内
    shifted_indices = np.clip(shifted_indices, 0, len(spectrum)-1)
    
    # 线性插值
    f = interpolate.interp1d(indices, spectrum, kind='linear', fill_value='extrapolate')
    try:
        shifted_spectrum = f(shifted_indices)
    except:
        # 如果插值失败，使用简单方法
        shifted_spectrum = spectrum
    
    return shifted_spectrum

def peak_perturbation(spectrum, perturbation_factor=(0.8, 1.2)):
    """
    对光谱中的峰值进行增强或减弱
    
    参数:
    spectrum : array-like
        输入光谱强度数据
    perturbation_factor : tuple
        扰动因子范围（最小值，最大值）
    
    返回:
    扰动后的光谱
    """
    # 确定哪些点是峰值点（局部最大值）
    peaks = []
    for i in range(1, len(spectrum)-1):
        if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
            peaks.append(i)
    
    # 对峰值应用扰动
    perturbed_spectrum = np.copy(spectrum)
    factor_min, factor_max = perturbation_factor
    
    for peak in peaks:
        factor = np.random.uniform(factor_min, factor_max)
        perturbed_spectrum[peak] = spectrum[peak] * factor
    
    return perturbed_spectrum

def simulate_instrument_effects(spectrum, resolution_factor=1.0):
    """
    模拟不同仪器条件对光谱的影响
    
    参数:
    spectrum : array-like
        输入光谱强度数据
    resolution_factor : float
        分辨率因子（<1表示分辨率降低）
    
    返回:
    模拟仪器效应后的光谱
    """
    if resolution_factor >= 1.0:
        return spectrum
    
    # 降低分辨率的简单方法：移动平均
    window_size = int(1 / resolution_factor)
    if window_size % 2 == 0:
        window_size += 1
    if window_size < 3:
        window_size = 3
    
    # 移动平均
    padded_spectrum = np.pad(spectrum, (window_size//2, window_size//2), mode='edge')
    smoothed_spectrum = np.convolve(padded_spectrum, np.ones(window_size)/window_size, mode='valid')
    
    return smoothed_spectrum

def augment_spectrum(wavenumber, intensity, 
                     add_noise_flag=True,
                     spectral_shift_flag=True,
                     peak_perturbation_flag=True,
                     instrument_effects_flag=True,
                     num_augmentations=5):
    """
    对单个光谱进行多种增强操作
    
    参数:
    wavenumber : array-like
        波数数据
    intensity : array-like
        强度数据
    add_noise_flag : bool
        是否添加噪声
    spectral_shift_flag : bool
        是否进行光谱偏移
    peak_perturbation_flag : bool
        是否进行峰值扰动
    instrument_effects_flag : bool
        是否模拟仪器效应
    num_augmentations : int
        增强样本数量
    
    返回:
    增强后的光谱列表
    """
    augmented_spectra = []
    
    for _ in range(num_augmentations):
        augmented_intensity = np.copy(intensity)
        
        # 添加噪声
        if add_noise_flag:
            augmented_intensity = add_noise(augmented_intensity, noise_level=np.random.uniform(0.005, 0.02))
        
        # 光谱偏移
        if spectral_shift_flag:
            augmented_intensity = spectral_shift(augmented_intensity, shift_range=(-3, 3))
        
        # 峰值扰动
        if peak_perturbation_flag:
            augmented_intensity = peak_perturbation(augmented_intensity, perturbation_factor=(0.9, 1.1))
        
        # 模拟仪器效应
        if instrument_effects_flag:
            resolution = np.random.uniform(0.7, 1.0)
            augmented_intensity = simulate_instrument_effects(augmented_intensity, resolution_factor=resolution)
        
        augmented_spectra.append((np.copy(wavenumber), augmented_intensity))
    
    return augmented_spectra

def generate_augmented_dataset(spectra_data, labels=None, augmentation_factor=5):
    """
    为整个数据集生成增强样本
    
    参数:
    spectra_data : list of tuples
        光谱数据列表，每个元素为(波数, 强度)元组
    labels : list
        标签列表，如果提供则会为增强样本复制标签
    augmentation_factor : int
        每个原始样本的增强倍数
    
    返回:
    增强后的数据集和对应的标签
    """
    augmented_data = []
    augmented_labels = [] if labels is not None else None
    
    for i, (wavenumber, intensity) in enumerate(spectra_data):
        # 添加原始数据
        augmented_data.append((wavenumber, intensity))
        if labels is not None:
            augmented_labels.append(labels[i])
        
        # 生成增强样本
        augmented_spectra = augment_spectrum(wavenumber, intensity, 
                                           num_augmentations=augmentation_factor)
        augmented_data.extend(augmented_spectra)
        
        if labels is not None:
            # 为增强样本复制标签
            augmented_labels.extend([labels[i]] * augmentation_factor)
    
    return augmented_data, augmented_labels

def visualize_augmentation(wavenumber, original_intensity, augmented_spectra, title="数据增强效果"):
    """
    可视化数据增强效果
    
    参数:
    wavenumber : array-like
        波数数据
    original_intensity : array-like
        原始强度数据
    augmented_spectra : list of tuples
        增强后的光谱数据列表
    title : str
        图表标题
    """
    plt.figure(figsize=(15, 10))
    
    # 绘制原始光谱
    plt.subplot(2, 3, 1)
    plt.plot(wavenumber, original_intensity, linewidth=1, color='black')
    plt.title('原始光谱')
    plt.xlabel('波数 (cm⁻¹)')
    plt.ylabel('强度')
    plt.grid(True, alpha=0.3)
    
    # 绘制增强后的光谱
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (aug_wavenumber, aug_intensity) in enumerate(augmented_spectra[:5]):  # 最多显示5个增强样本
        plt.subplot(2, 3, i+2)
        plt.plot(wavenumber, original_intensity, linewidth=1, color='black', alpha=0.5, label='原始')
        plt.plot(aug_wavenumber, aug_intensity, linewidth=1, color=colors[i % len(colors)], label='增强')
        plt.title(f'增强样本 {i+1}')
        plt.xlabel('波数 (cm⁻¹)')
        plt.ylabel('强度')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 所有增强样本的对比
    plt.subplot(2, 3, 6)
    plt.plot(wavenumber, original_intensity, linewidth=2, color='black', label='原始')
    for i, (aug_wavenumber, aug_intensity) in enumerate(augmented_spectra[:5]):
        plt.plot(aug_wavenumber, aug_intensity, linewidth=1, 
                color=colors[i % len(colors)], alpha=0.7, label=f'增强{i+1}')
    plt.title('所有增强样本对比')
    plt.xlabel('波数 (cm⁻¹)')
    plt.ylabel('强度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def visualize_augmentation_techniques(wavenumber, intensity):
    """
    可视化各种数据增强技术的效果
    
    参数:
    wavenumber : array-like
        波数数据
    intensity : array-like
        原始强度数据
    """
    plt.figure(figsize=(15, 12))
    
    # 原始光谱
    plt.subplot(3, 2, 1)
    plt.plot(wavenumber, intensity, linewidth=1, color='black')
    plt.title('原始光谱')
    plt.xlabel('波数 (cm⁻¹)')
    plt.ylabel('强度')
    plt.grid(True, alpha=0.3)
    
    # 添加噪声
    noisy_spectrum = add_noise(intensity, noise_level=0.02)
    plt.subplot(3, 2, 2)
    plt.plot(wavenumber, intensity, linewidth=1, color='black', alpha=0.5, label='原始')
    plt.plot(wavenumber, noisy_spectrum, linewidth=1, color='red', label='添加噪声')
    plt.title('添加噪声效果')
    plt.xlabel('波数 (cm⁻¹)')
    plt.ylabel('强度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 光谱偏移
    shifted_spectrum = spectral_shift(intensity, shift_range=(-5, 5))
    plt.subplot(3, 2, 3)
    plt.plot(wavenumber, intensity, linewidth=1, color='black', alpha=0.5, label='原始')
    plt.plot(wavenumber, shifted_spectrum, linewidth=1, color='blue', label='光谱偏移')
    plt.title('光谱偏移效果')
    plt.xlabel('波数 (cm⁻¹)')
    plt.ylabel('强度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 峰值扰动
    perturbed_spectrum = peak_perturbation(intensity, perturbation_factor=(0.7, 1.3))
    plt.subplot(3, 2, 4)
    plt.plot(wavenumber, intensity, linewidth=1, color='black', alpha=0.5, label='原始')
    plt.plot(wavenumber, perturbed_spectrum, linewidth=1, color='green', label='峰值扰动')
    plt.title('峰值扰动效果')
    plt.xlabel('波数 (cm⁻¹)')
    plt.ylabel('强度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 模拟仪器效应
    instrument_spectrum = simulate_instrument_effects(intensity, resolution_factor=0.5)
    plt.subplot(3, 2, 5)
    plt.plot(wavenumber, intensity, linewidth=1, color='black', alpha=0.5, label='原始')
    plt.plot(wavenumber, instrument_spectrum, linewidth=1, color='orange', label='仪器效应')
    plt.title('模拟仪器效应效果')
    plt.xlabel('波数 (cm⁻¹)')
    plt.ylabel('强度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 综合增强效果
    augmented_spectra = augment_spectrum(wavenumber, intensity, num_augmentations=1)
    plt.subplot(3, 2, 6)
    plt.plot(wavenumber, intensity, linewidth=2, color='black', label='原始')
    plt.plot(wavenumber, augmented_spectra[0][1], linewidth=1, color='purple', label='综合增强')
    plt.title('综合增强效果')
    plt.xlabel('波数 (cm⁻¹)')
    plt.ylabel('强度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('拉曼光谱数据增强技术可视化')
    plt.tight_layout()
    plt.show()

# 示例和测试代码
def test_augmentation():
    """
    测试数据增强功能
    """
    # 生成模拟拉曼光谱数据
    wavenumber = np.linspace(200, 2000, 1000)
    # 模拟几个拉曼峰
    intensity = (100 * np.exp(-0.5*((wavenumber-500)/20)**2) + 
                80 * np.exp(-0.5*((wavenumber-1000)/15)**2) + 
                120 * np.exp(-0.5*((wavenumber-1600)/25)**2) +
                np.random.normal(0, 2, len(wavenumber)))
    
    # 可视化各种增强技术
    visualize_augmentation_techniques(wavenumber, intensity)
    
    # 生成增强样本并可视化
    augmented_spectra = augment_spectrum(wavenumber, intensity, num_augmentations=5)
    visualize_augmentation(wavenumber, intensity, augmented_spectra)

if __name__ == "__main__":
    test_augmentation()