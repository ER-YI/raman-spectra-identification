# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def baseline_correction_als(y, lam=10**6, p=0.001, niter=10):
    """
    使用非对称最小二乘法(Asymmetric Least Squares)进行基线校正
    
    参数:
    y : array-like
        原始光谱信号
    lam : float
        平滑参数，较大值产生更平滑的基线
    p : float
        非对称参数，介于0-1之间
    niter : int
        迭代次数
    
    返回:
    校正后的光谱信号
    """
    L = len(y)
    D = csr_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = csr_matrix((w, (range(L), range(L))), shape=(L, L))
        Z = W + lam * D.dot(D.transpose())
        z = np.linalg.solve(Z.toarray(), w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return y - z

def remove_noise(y, window_length=11, polyorder=3):
    """
    使用Savitzky-Golay滤波器去除噪声
    
    参数:
    y : array-like
        输入信号
    window_length : int
        滤波器窗口长度，必须为正奇数
    polyorder : int
        用于拟合的多项式阶数，必须小于window_length
    
    返回:
    去噪后的信号
    """
    if window_length % 2 == 0:
        window_length += 1  # 确保窗口长度为奇数
    if window_length <= polyorder:
        window_length = polyorder + 3  # 确保窗口长度大于多项式阶数
    
    return signal.savgol_filter(y, window_length, polyorder)

def normalize_spectrum(y, method='minmax'):
    """
    光谱归一化
    
    参数:
    y : array-like
        输入信号
    method : str
        归一化方法 ('minmax', 'area', 'max')
    
    返回:
    归一化后的信号
    """
    if method == 'minmax':
        scaler = MinMaxScaler()
        return scaler.fit_transform(y.reshape(-1, 1)).flatten()
    elif method == 'area':
        area = np.trapz(y)
        return y / area if area != 0 else y
    elif method == 'max':
        max_val = np.max(y)
        return y / max_val if max_val != 0 else y
    else:
        raise ValueError("不支持的归一化方法")

def align_spectra(spectra, reference_index=0):
    """
    光谱对齐
    
    参数:
    spectra : list of array-like
        光谱数据列表
    reference_index : int
        参考光谱的索引
    
    返回:
    对齐后的光谱数据列表
    """
    # 简单实现：基于互相关进行对齐
    reference = spectra[reference_index]
    aligned_spectra = [reference]
    
    for i, spectrum in enumerate(spectra):
        if i == reference_index:
            continue
            
        # 计算互相关
        correlation = signal.correlate(reference, spectrum, mode='full')
        lag = np.argmax(correlation) - (len(spectrum) - 1)
        
        # 根据滞后量进行对齐
        if lag > 0:
            aligned_spectrum = np.pad(spectrum, (lag, 0), mode='constant')[:-lag if lag != 0 else None]
        elif lag < 0:
            aligned_spectrum = np.pad(spectrum, (0, -lag), mode='constant')[-lag:]
        else:
            aligned_spectrum = spectrum
            
        aligned_spectra.append(aligned_spectrum)
    
    return aligned_spectra

def preprocess_spectrum(wavenumber, intensity, 
                       baseline_correct=True, 
                       remove_noise_flag=True, 
                       normalize=True,
                       baseline_params=None,
                       noise_params=None):
    """
    对拉曼光谱进行完整的预处理
    
    参数:
    wavenumber : array-like
        波数数据
    intensity : array-like
        强度数据
    baseline_correct : bool
        是否进行基线校正
    remove_noise_flag : bool
        是否去除噪声
    normalize : bool
        是否归一化
    baseline_params : dict
        基线校正参数
    noise_params : dict
        去噪参数
    
    返回:
    处理后的波数和强度数据
    """
    processed_intensity = np.array(intensity)
    
    # 基线校正
    if baseline_correct:
        params = baseline_params if baseline_params else {}
        processed_intensity = baseline_correction_als(processed_intensity, **params)
    
    # 去噪
    if remove_noise_flag:
        params = noise_params if noise_params else {}
        processed_intensity = remove_noise(processed_intensity, **params)
    
    # 归一化
    if normalize:
        processed_intensity = normalize_spectrum(processed_intensity, method='minmax')
    
    return np.array(wavenumber), processed_intensity

def visualize_preprocessing(wavenumber, original_intensity, processed_intensity, title="光谱预处理效果"):
    """
    可视化预处理前后的光谱对比
    
    参数:
    wavenumber : array-like
        波数数据
    original_intensity : array-like
        原始强度数据
    processed_intensity : array-like
        处理后的强度数据
    title : str
        图表标题
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(wavenumber, original_intensity, linewidth=1)
    plt.title('原始光谱')
    plt.xlabel('波数 (cm⁻¹)')
    plt.ylabel('强度')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(wavenumber, processed_intensity, linewidth=1, color='red')
    plt.title('预处理后光谱')
    plt.xlabel('波数 (cm⁻¹)')
    plt.ylabel('强度')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def visualize_preprocessing_steps(wavenumber, intensity):
    """
    可视化每一步预处理的效果
    
    参数:
    wavenumber : array-like
        波数数据
    intensity : array-like
        原始强度数据
    """
    plt.figure(figsize=(15, 10))
    
    # 原始光谱
    plt.subplot(2, 3, 1)
    plt.plot(wavenumber, intensity, linewidth=1)
    plt.title('原始光谱')
    plt.xlabel('波数 (cm⁻¹)')
    plt.ylabel('强度')
    plt.grid(True, alpha=0.3)
    
    # 基线校正
    baseline_corrected = baseline_correction_als(intensity)
    plt.subplot(2, 3, 2)
    plt.plot(wavenumber, baseline_corrected, linewidth=1, color='green')
    plt.title('基线校正后')
    plt.xlabel('波数 (cm⁻¹)')
    plt.ylabel('强度')
    plt.grid(True, alpha=0.3)
    
    # 去噪
    denoised = remove_noise(baseline_corrected)
    plt.subplot(2, 3, 3)
    plt.plot(wavenumber, denoised, linewidth=1, color='orange')
    plt.title('去噪后')
    plt.xlabel('波数 (cm⁻¹)')
    plt.ylabel('强度')
    plt.grid(True, alpha=0.3)
    
    # 归一化
    normalized = normalize_spectrum(denoised)
    plt.subplot(2, 3, 4)
    plt.plot(wavenumber, normalized, linewidth=1, color='purple')
    plt.title('归一化后')
    plt.xlabel('波数 (cm⁻¹)')
    plt.ylabel('强度')
    plt.grid(True, alpha=0.3)
    
    # 对比原始和最终结果
    plt.subplot(2, 3, (5, 6))
    plt.plot(wavenumber, intensity, linewidth=1, label='原始', alpha=0.7)
    plt.plot(wavenumber, normalized, linewidth=1, label='预处理后', color='red')
    plt.title('预处理效果对比')
    plt.xlabel('波数 (cm⁻¹)')
    plt.ylabel('强度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('拉曼光谱预处理步骤可视化')
    plt.tight_layout()
    plt.show()

# 测试函数
def test_preprocessing():
    """
    测试预处理功能
    """
    # 生成模拟拉曼光谱数据
    wavenumber = np.linspace(200, 2000, 1000)
    # 模拟几个拉曼峰
    intensity = (100 * np.exp(-0.5*((wavenumber-500)/20)**2) + 
                80 * np.exp(-0.5*((wavenumber-1000)/15)**2) + 
                120 * np.exp(-0.5*((wavenumber-1600)/25)**2) +
                np.random.normal(0, 5, len(wavenumber)))  # 添加噪声
    
    # 添加基线
    baseline = 0.01 * wavenumber + 5
    intensity_with_baseline = intensity + baseline
    
    # 显示原始数据
    visualize_preprocessing_steps(wavenumber, intensity_with_baseline)

if __name__ == "__main__":
    test_preprocessing()