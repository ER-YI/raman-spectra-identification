# -*- coding: utf-8 -*-
"""
模型训练和导出脚本
用于训练各种拉曼光谱分类模型并导出为文件
"""

import numpy as np
import os
import argparse
import sys
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def generate_sample_data(sample_size=1000):
    """
    生成示例拉曼光谱数据
    
    参数:
    sample_size : int
        样本总数
        
    返回:
    spectra_data : list of tuples
        光谱数据列表
    labels : list
        标签列表
    """
    np.random.seed(42)
    
    # 计算每类样本数
    samples_per_class = sample_size // 3
    
    # 创建3类拉曼光谱数据
    spectra_data = []
    labels = []
    
    # 类别1 - 模拟石英矿物
    for i in range(samples_per_class):
        wavenumber = np.linspace(200, 2000, 1000)
        # 添加一些随机性到峰值位置和强度
        peak1_center = np.random.normal(450, 5)
        peak2_center = np.random.normal(800, 10)
        peak3_center = np.random.normal(1600, 15)
        
        intensity = (np.random.normal(100, 15) * np.exp(-0.5*((wavenumber-peak1_center)/20)**2) + 
                    np.random.normal(80, 12) * np.exp(-0.5*((wavenumber-peak2_center)/15)**2) + 
                    np.random.normal(60, 10) * np.exp(-0.5*((wavenumber-peak3_center)/25)**2) +
                    np.random.normal(0, np.random.uniform(3, 8), len(wavenumber)))  # 随机噪声
        
        # 添加基线
        baseline = np.random.uniform(2, 10) + np.random.uniform(0.005, 0.02) * wavenumber
        intensity += baseline
        
        spectra_data.append((wavenumber, intensity))
        labels.append("石英")
    
    # 类别2 - 模拟方解石
    for i in range(samples_per_class):
        wavenumber = np.linspace(200, 2000, 1000)
        # 添加一些随机性到峰值位置和强度
        peak1_center = np.random.normal(300, 8)
        peak2_center = np.random.normal(720, 12)
        peak3_center = np.random.normal(1080, 10)
        
        intensity = (np.random.normal(70, 12) * np.exp(-0.5*((wavenumber-peak1_center)/20)**2) + 
                    np.random.normal(100, 15) * np.exp(-0.5*((wavenumber-peak2_center)/15)**2) + 
                    np.random.normal(90, 12) * np.exp(-0.5*((wavenumber-peak3_center)/25)**2) +
                    np.random.normal(0, np.random.uniform(3, 8), len(wavenumber)))  # 随机噪声
        
        # 添加基线
        baseline = np.random.uniform(3, 12) + np.random.uniform(0.003, 0.015) * wavenumber
        intensity += baseline
        
        spectra_data.append((wavenumber, intensity))
        labels.append("方解石")
    
    # 类别3 - 模拟白云石
    for i in range(samples_per_class):
        wavenumber = np.linspace(200, 2000, 1000)
        # 添加一些随机性到峰值位置和强度
        peak1_center = np.random.normal(350, 7)
        peak2_center = np.random.normal(780, 10)
        peak3_center = np.random.normal(1100, 12)
        
        intensity = (np.random.normal(90, 13) * np.exp(-0.5*((wavenumber-peak1_center)/20)**2) + 
                    np.random.normal(60, 10) * np.exp(-0.5*((wavenumber-peak2_center)/15)**2) + 
                    np.random.normal(110, 14) * np.exp(-0.5*((wavenumber-peak3_center)/25)**2) +
                    np.random.normal(0, np.random.uniform(3, 8), len(wavenumber)))  # 随机噪声
        
        # 添加基线
        baseline = np.random.uniform(2, 8) + np.random.uniform(0.004, 0.018) * wavenumber
        intensity += baseline
        
        spectra_data.append((wavenumber, intensity))
        labels.append("白云石")
    
    return spectra_data, labels

def train_and_export_models(spectra_data, labels, output_dir="./models", epochs=100):
    """
    训练所有模型并导出
    
    参数:
    spectra_data : list of tuples
        光谱数据列表
    labels : list
        标签列表
    output_dir : str
        模型输出目录
    epochs : int
        神经网络训练轮数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始训练模型，数据集大小: {len(spectra_data)} 个样本")
    print(f"类别: {list(set(labels))}")
    
    # 数据预处理
    try:
        from preprocessing import preprocess_spectrum
        print("\n正在进行数据预处理...")
        processed_spectra = []
        for wavenumber, intensity in spectra_data:
            _, processed_intensity = preprocess_spectrum(
                wavenumber, intensity,
                baseline_correct=True,
                remove_noise_flag=True,
                normalize=True
            )
            processed_spectra.append((wavenumber, processed_intensity))
    except Exception as e:
        print(f"数据预处理时出错: {e}")
        print("使用原始数据继续...")
        processed_spectra = spectra_data
    
    # 数据增强
    try:
        from data_augmentation import generate_augmented_dataset
        print("正在进行数据增强...")
        augmented_data, augmented_labels = generate_augmented_dataset(
            processed_spectra, labels, augmentation_factor=3
        )
        print(f"增强后数据量: {len(augmented_data)}")
    except Exception as e:
        print(f"数据增强时出错: {e}")
        print("使用预处理数据继续...")
        augmented_data, augmented_labels = processed_spectra, labels
    
    # 训练SVM模型
    print("\n训练SVM模型...")
    try:
        from models import RamanSpectrumClassifier
        svm_classifier = RamanSpectrumClassifier(model_type='svm')
        svm_classifier.train(augmented_data, augmented_labels)
        svm_classifier.evaluate()
        
        # 保存SVM模型
        svm_model_path = os.path.join(output_dir, "svm_model.pkl")
        svm_classifier.save_model(svm_model_path)
        print(f"SVM模型已保存到: {svm_model_path}")
    except Exception as e:
        print(f"训练或保存SVM模型时出错: {e}")
    
    # 训练随机森林模型
    print("\n训练随机森林模型...")
    try:
        from models import RamanSpectrumClassifier
        rf_classifier = RamanSpectrumClassifier(model_type='rf')
        rf_classifier.train(augmented_data, augmented_labels)
        rf_classifier.evaluate()
        
        # 保存随机森林模型
        rf_model_path = os.path.join(output_dir, "rf_model.pkl")
        rf_classifier.save_model(rf_model_path)
        print(f"随机森林模型已保存到: {rf_model_path}")
    except Exception as e:
        print(f"训练或保存随机森林模型时出错: {e}")
    
    # 训练普通神经网络模型
    print("\n训练普通神经网络模型...")
    try:
        from neural_network import RamanSpectrumNNClassifier
        X = np.array([intensity for _, intensity in augmented_data])
        num_classes = len(np.unique(augmented_labels))
        nn_classifier = RamanSpectrumNNClassifier(X.shape[1], num_classes, list(np.unique(augmented_labels)))
        nn_classifier.train(augmented_data, augmented_labels, epochs=epochs, verbose=0)
        nn_classifier.evaluate()
        
        # 保存普通神经网络模型
        nn_model_path = os.path.join(output_dir, "nn_model.pth")
        nn_classifier.save_model(nn_model_path)
        print(f"普通神经网络模型已保存到: {nn_model_path}")
    except Exception as e:
        print(f"训练或保存普通神经网络模型时出错: {e}")
    
    # 训练CNN模型
    print("\n训练CNN模型...")
    try:
        from cnn_model import RamanSpectrumCNNClassifier
        X = np.array([intensity for _, intensity in augmented_data])
        input_length = X.shape[1]
        num_classes = len(np.unique(augmented_labels))
        cnn_classifier = RamanSpectrumCNNClassifier(input_length, num_classes, list(np.unique(augmented_labels)))
        cnn_classifier.train(augmented_data, augmented_labels, epochs=epochs, verbose=0)
        cnn_classifier.evaluate()
        
        # 保存CNN模型
        cnn_model_path = os.path.join(output_dir, "cnn_model.pth")
        cnn_classifier.save_model(cnn_model_path)
        print(f"CNN模型已保存到: {cnn_model_path}")
    except Exception as e:
        print(f"训练或保存CNN模型时出错: {e}")
    
    # 创建模型信息文件
    try:
        model_info = f"""
模型训练完成报告
================
训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
原始数据量: {len(spectra_data)}
增强后数据量: {len(augmented_data)}
类别数量: {num_classes}
类别名称: {list(np.unique(augmented_labels))}
模型文件:
  1. SVM模型: {svm_model_path if 'svm_model_path' in locals() else 'N/A'}
  2. 随机森林模型: {rf_model_path if 'rf_model_path' in locals() else 'N/A'}
  3. 普通神经网络模型: {nn_model_path if 'nn_model_path' in locals() else 'N/A'}
  4. CNN模型: {cnn_model_path if 'cnn_model_path' in locals() else 'N/A'}
"""
        
        info_file_path = os.path.join(output_dir, "model_info.txt")
        with open(info_file_path, 'w', encoding='utf-8') as f:
            f.write(model_info)
        
        print(f"\n模型信息已保存到: {info_file_path}")
    except Exception as e:
        print(f"保存模型信息时出错: {e}")
    
    print(f"\n所有模型处理完成，输出目录: {output_dir}")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='训练拉曼光谱分类模型并导出')
    parser.add_argument('--size', type=int, default=1000, help='训练数据集大小 (默认: 1000)')
    parser.add_argument('--epochs', type=int, default=100, help='神经网络训练轮数 (默认: 100)')
    parser.add_argument('--output', type=str, default='./exported_models', help='模型输出目录 (默认: ./exported_models)')
    
    args = parser.parse_args()
    
    # 生成数据
    print("正在生成示例数据...")
    spectra_data, labels = generate_sample_data(args.size)
    
    # 训练和导出模型
    train_and_export_models(spectra_data, labels, args.output, args.epochs)

if __name__ == "__main__":
    main()