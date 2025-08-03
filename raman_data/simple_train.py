# -*- coding: utf-8 -*-
"""
简化版模型训练脚本
用于快速训练和导出一个CNN模型作为示例
"""

import numpy as np
import os
import sys
import torch

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def generate_sample_data(sample_size=300):
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
    
    # 类别1
    for i in range(samples_per_class):
        wavenumber = np.linspace(200, 2000, 1000)
        intensity = (np.random.normal(100, 10) * np.exp(-0.5*((wavenumber-500)/20)**2) + 
                    np.random.normal(80, 10) * np.exp(-0.5*((wavenumber-1000)/15)**2) + 
                    np.random.normal(20, 5) * np.exp(-0.5*((wavenumber-1600)/25)**2) +
                    np.random.normal(0, 5, len(wavenumber)))
        spectra_data.append((wavenumber, intensity))
        labels.append(0)  # 类别0
    
    # 类别2
    for i in range(samples_per_class):
        wavenumber = np.linspace(200, 2000, 1000)
        intensity = (np.random.normal(20, 5) * np.exp(-0.5*((wavenumber-500)/20)**2) + 
                    np.random.normal(100, 10) * np.exp(-0.5*((wavenumber-1000)/15)**2) + 
                    np.random.normal(80, 10) * np.exp(-0.5*((wavenumber-1600)/25)**2) +
                    np.random.normal(0, 5, len(wavenumber)))
        spectra_data.append((wavenumber, intensity))
        labels.append(1)  # 类别1
    
    # 类别3
    for i in range(samples_per_class):
        wavenumber = np.linspace(200, 2000, 1000)
        intensity = (np.random.normal(80, 10) * np.exp(-0.5*((wavenumber-500)/20)**2) + 
                    np.random.normal(20, 5) * np.exp(-0.5*((wavenumber-1000)/15)**2) + 
                    np.random.normal(100, 10) * np.exp(-0.5*((wavenumber-1600)/25)**2) +
                    np.random.normal(0, 5, len(wavenumber)))
        spectra_data.append((wavenumber, intensity))
        labels.append(2)  # 类别2
    
    return spectra_data, labels

def simple_cnn_train_and_export():
    """
    简单的CNN训练和导出示例
    """
    print("生成示例数据...")
    spectra_data, labels = generate_sample_data(300)
    
    print("加载CNN模型模块...")
    try:
        from cnn_model import RamanSpectrumCNNClassifier
    except ImportError as e:
        print(f"导入CNN模型时出错: {e}")
        print("请确保所有依赖模块都存在")
        return
    
    # 准备数据
    X = np.array([intensity for _, intensity in spectra_data])
    input_length = X.shape[1]
    num_classes = len(np.unique(labels))
    
    print(f"数据准备完成:")
    print(f"  样本数: {X.shape[0]}")
    print(f"  特征数: {X.shape[1]}")
    print(f"  类别数: {num_classes}")
    
    # 创建CNN分类器
    print("创建CNN分类器...")
    cnn_classifier = RamanSpectrumCNNClassifier(input_length, num_classes)
    
    # 训练模型
    print("开始训练CNN模型...")
    try:
        train_score, test_score = cnn_classifier.train(spectra_data, labels, epochs=20, verbose=0)
        print(f"训练完成!")
        print(f"  训练集准确率: {train_score:.4f}")
        print(f"  测试集准确率: {test_score:.4f}")
    except Exception as e:
        print(f"训练过程中出错: {e}")
        return
    
    # 评估模型
    print("评估模型...")
    try:
        cnn_classifier.evaluate()
    except Exception as e:
        print(f"评估过程中出错: {e}")
    
    # 保存模型
    print("保存模型...")
    try:
        # 创建输出目录
        output_dir = "./exported_models"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(output_dir, "simple_cnn_model.pth")
        cnn_classifier.save_model(model_path)
        print(f"模型已保存到: {model_path}")
        
        # 验证模型可以加载
        print("验证模型加载...")
        new_classifier = RamanSpectrumCNNClassifier(input_length, num_classes)
        new_classifier.load_model(model_path)
        print("模型加载验证成功!")
        
    except Exception as e:
        print(f"保存模型时出错: {e}")
        return
    
    print("CNN模型训练和导出示例完成!")

def test_model_loading():
    """
    测试模型加载功能
    """
    print("测试模型加载功能...")
    try:
        from cnn_model import RamanSpectrumCNNClassifier
        
        # 创建测试数据
        test_data = []
        wavenumber = np.linspace(200, 2000, 1000)
        intensity = (90 * np.exp(-0.5*((wavenumber-500)/20)**2) + 
                    70 * np.exp(-0.5*((wavenumber-1000)/15)**2) + 
                    30 * np.exp(-0.5*((wavenumber-1600)/25)**2) +
                    np.random.normal(0, 5, len(wavenumber)))
        test_data.append((wavenumber, intensity))
        
        # 加载模型
        model_path = "./exported_models/simple_cnn_model.pth"
        if os.path.exists(model_path):
            classifier = RamanSpectrumCNNClassifier(1000, 3)
            classifier.load_model(model_path)
            
            # 进行预测
            predictions, probabilities = classifier.predict(test_data)
            print(f"预测结果: {predictions}")
            print(f"预测概率: {probabilities}")
            print("模型加载和预测测试成功!")
        else:
            print(f"模型文件不存在: {model_path}")
            
    except Exception as e:
        print(f"测试模型加载时出错: {e}")

if __name__ == "__main__":
    print("=== 简化版拉曼光谱CNN模型训练和导出示例 ===")
    
    # 训练和导出模型
    simple_cnn_train_and_export()
    
    # 测试模型加载
    test_model_loading()
    
    print("\n所有步骤完成!")