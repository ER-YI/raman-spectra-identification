# -*- coding: utf-8 -*-
"""
模型加载和预测示例脚本
演示如何加载已训练的模型并进行预测
"""

import numpy as np
import argparse
import os

def generate_test_data(num_samples=5):
    """
    生成测试数据
    
    参数:
    num_samples : int
        测试样本数量
        
    返回:
    test_data : list of tuples
        测试光谱数据
    """
    np.random.seed(123)  # 使用不同的随机种子
    
    test_data = []
    for i in range(num_samples):
        wavenumber = np.linspace(200, 2000, 1000)
        # 随机选择一个类别
        class_type = np.random.choice([0, 1, 2])
        
        if class_type == 0:  # 类别1 - 类似石英
            intensity = (np.random.normal(95, 10) * np.exp(-0.5*((wavenumber-440)/18)**2) + 
                        np.random.normal(75, 8) * np.exp(-0.5*((wavenumber-810)/14)**2) + 
                        np.random.normal(65, 8) * np.exp(-0.5*((wavenumber-1590)/22)**2) +
                        np.random.normal(0, 6, len(wavenumber)))
            baseline = np.random.uniform(3, 9) + np.random.uniform(0.004, 0.018) * wavenumber
            intensity += baseline
            label = "石英"
        elif class_type == 1:  # 类别2 - 类似方解石
            intensity = (np.random.normal(75, 10) * np.exp(-0.5*((wavenumber-290)/16)**2) + 
                        np.random.normal(95, 12) * np.exp(-0.5*((wavenumber-730)/13)**2) + 
                        np.random.normal(85, 10) * np.exp(-0.5*((wavenumber-1090)/20)**2) +
                        np.random.normal(0, 6, len(wavenumber)))
            baseline = np.random.uniform(4, 10) + np.random.uniform(0.003, 0.014) * wavenumber
            intensity += baseline
            label = "方解石"
        else:  # 类别3 - 类似白云石
            intensity = (np.random.normal(85, 11) * np.exp(-0.5*((wavenumber-360)/15)**2) + 
                        np.random.normal(65, 9) * np.exp(-0.5*((wavenumber-770)/12)**2) + 
                        np.random.normal(105, 12) * np.exp(-0.5*((wavenumber-1110)/21)**2) +
                        np.random.normal(0, 6, len(wavenumber)))
            baseline = np.random.uniform(3, 7) + np.random.uniform(0.005, 0.016) * wavenumber
            intensity += baseline
            label = "白云石"
        
        test_data.append(((wavenumber, intensity), label))
    
    return test_data

def load_and_predict(model_type, model_path, test_data):
    """
    加载模型并进行预测
    
    参数:
    model_type : str
        模型类型 ('svm', 'rf', 'nn', 'cnn')
    model_path : str
        模型文件路径
    test_data : list
        测试数据
    """
    print(f"正在加载 {model_type.upper()} 模型...")
    
    # 分离光谱数据和真实标签
    spectra_data = [data[0] for data in test_data]
    true_labels = [data[1] for data in test_data]
    
    if model_type in ['svm', 'rf']:
        from models import RamanSpectrumClassifier
        classifier = RamanSpectrumClassifier()
        classifier.load_model(model_path)
    elif model_type == 'nn':
        from neural_network import RamanSpectrumNNClassifier
        # 先创建一个临时分类器来加载模型信息
        temp_classifier = RamanSpectrumNNClassifier(1000, 3)  # 参数会在加载时被替换
        temp_classifier.load_model(model_path)
        
        # 使用加载的信息重新创建分类器
        classifier = RamanSpectrumNNClassifier(
            temp_classifier.input_dim, 
            temp_classifier.num_classes, 
            temp_classifier.class_names
        )
        classifier.load_model(model_path)
    elif model_type == 'cnn':
        from cnn_model import RamanSpectrumCNNClassifier
        # 先创建一个临时分类器来加载模型信息
        temp_classifier = RamanSpectrumCNNClassifier(1000, 3)  # 参数会在加载时被替换
        temp_classifier.load_model(model_path)
        
        # 使用加载的信息重新创建分类器
        classifier = RamanSpectrumCNNClassifier(
            temp_classifier.input_length, 
            temp_classifier.num_classes, 
            temp_classifier.class_names
        )
        classifier.load_model(model_path)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 进行预测
    print("正在进行预测...")
    predictions, probabilities = classifier.predict(spectra_data)
    
    # 解码预测结果（对于神经网络模型）
    if hasattr(classifier, 'label_encoder'):
        decoded_predictions = classifier.label_encoder.inverse_transform(predictions)
    else:
        decoded_predictions = predictions
    
    # 显示结果
    print(f"\n{model_type.upper()} 模型预测结果:")
    print("-" * 50)
    for i, (true_label, pred, prob) in enumerate(zip(true_labels, decoded_predictions, probabilities)):
        print(f"样本 {i+1}:")
        print(f"  真实标签: {true_label}")
        print(f"  预测标签: {pred}")
        print(f"  预测概率: {[f'{p:.3f}' for p in prob]}")
        
        # 计算置信度
        confidence = np.max(prob)
        print(f"  预测置信度: {confidence:.3f}")
        print(f"  预测正确: {'是' if str(true_label) == str(pred) else '否'}")
        print()

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='加载已训练的模型并进行预测')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['svm', 'rf', 'nn', 'cnn'], 
                       help='模型类型 (svm, rf, nn, cnn)')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='模型文件路径')
    parser.add_argument('--num_samples', type=int, default=5, 
                       help='测试样本数量 (默认: 5)')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件 {args.model_path} 不存在")
        return
    
    # 生成测试数据
    print("正在生成测试数据...")
    test_data = generate_test_data(args.num_samples)
    
    # 加载模型并预测
    load_and_predict(args.model, args.model_path, test_data)

if __name__ == "__main__":
    main()