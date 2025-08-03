# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from urllib.parse import urljoin
import numpy as np
import matplotlib.pyplot as plt

# 导入我们创建的模块
from preprocessing import preprocess_spectrum, visualize_preprocessing_steps
from data_augmentation import generate_augmented_dataset, visualize_augmentation_techniques, augment_spectrum
from models import RamanSpectrumClassifier, visualize_multiple_spectra

def get_rruff_statistics():
    """
    获取RRUFF数据库的统计信息
    """
    base_url = "https://rruff.info"
    
    try:
        # 设置请求头，模拟浏览器访问
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # 首先访问主页
        print("正在访问RRUFF主页...")
        response = requests.get(base_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        print("成功连接到RRUFF数据库")
        
        # 查找统计信息
        stats_info = []
        
        # 查找包含数字和数据相关信息的文本
        text_elements = soup.find_all(string=re.compile(r'\d+\s*(specimen|mineral|file|dataset|sample)', re.I))
        for text in text_elements:
            stats_info.append(text.strip())
        
        # 查找特定的统计信息区域
        # RRUFF网站可能有专门的统计信息页面
        stat_links = soup.find_all('a', href=re.compile(r'(stat|about|info)', re.I))
        
        # 查找拉曼光谱相关链接
        raman_links = soup.find_all('a', href=re.compile(r'raman', re.I))
        
        return {
            'general_stats': stats_info,
            'stat_links': len(stat_links),
            'raman_links': len(raman_links)
        }
        
    except requests.exceptions.RequestException as e:
        print(f"网络请求错误: {e}")
        return None
    except Exception as e:
        print(f"解析网页时出错: {e}")
        return None

def explore_rruff_search():
    """
    探索RRUFF搜索功能
    """
    search_url = "https://rruff.info/search.php"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 查找搜索表单
        search_form = soup.find('form')
        if search_form:
            print("找到搜索表单")
            
            # 查找表单中的输入字段
            inputs = search_form.find_all('input')
            print(f"表单包含 {len(inputs)} 个输入字段")
            
            # 查找选择框
            selects = search_form.find_all('select')
            print(f"表单包含 {len(selects)} 个选择框")
            
            # 查找提交按钮
            submit_buttons = search_form.find_all('input', type='submit')
            print(f"表单包含 {len(submit_buttons)} 个提交按钮")
        
        # 查找可能的矿物列表或分类
        mineral_categories = soup.find_all('option')
        if mineral_categories:
            print(f"找到 {len(mineral_categories)} 个选项（可能是矿物分类）")
            categories = [opt.text.strip() for opt in mineral_categories if opt.text.strip()]
            if categories:
                print("部分分类示例:")
                for cat in categories[:10]:
                    print(f"  - {cat}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"访问搜索页面时出错: {e}")
        return False
    except Exception as e:
        print(f"解析搜索页面时出错: {e}")
        return False

def get_sample_raman_data():
    """
    尝试获取示例拉曼光谱数据
    """
    # RRUFF数据库中拉曼光谱的典型URL格式
    sample_urls = [
        "https://rruff.info/raman_sample.php",
        "https://rruff.info/spectrum.php",
        "https://rruff.info/download_sample.php"
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for url in sample_urls:
        try:
            print(f"尝试访问: {url}")
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                print(f"成功访问 {url}")
                # 简单分析页面内容
                soup = BeautifulSoup(response.content, 'html.parser')
                # 查找数据表格或下载链接
                tables = soup.find_all('table')
                links = soup.find_all('a', href=re.compile(r'\.(csv|txt|dat|zip)'))
                print(f"  找到 {len(tables)} 个表格, {len(links)} 个下载链接")
                return True
            else:
                print(f"访问 {url} 失败，状态码: {response.status_code}")
        except Exception as e:
            print(f"访问 {url} 时出错: {e}")
    
    return False

def explore_rruff_minerals():
    """
    探索RRUFF数据库中的矿物种类
    """
    minerals_url = "https://rruff.info/minerals.php"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(minerals_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 查找矿物列表
        mineral_links = soup.find_all('a', href=re.compile(r'mineral_name'))
        
        print(f"在RRUFF数据库中找到 {len(mineral_links)} 个矿物链接")
        
        # 提取前20个矿物名称和链接
        minerals = []
        for link in mineral_links[:20]:
            mineral_name = link.text.strip()
            mineral_url = urljoin("https://rruff.info/", link.get('href', ''))
            minerals.append((mineral_name, mineral_url))
        
        print("前20个矿物示例:")
        for i, (name, url) in enumerate(minerals):
            print(f"  {i+1}. {name}: {url}")
        
        return minerals
        
    except requests.exceptions.RequestException as e:
        print(f"访问矿物页面时出错: {e}")
        return []
    except Exception as e:
        print(f"解析矿物页面时出错: {e}")
        return []

def get_raman_spectra_for_mineral(mineral_name):
    """
    获取特定矿物的拉曼光谱数据
    
    参数:
    mineral_name : str
        矿物名称
        
    返回:
    spectra_data : list
        光谱数据列表
    """
    # 构造搜索URL
    search_url = f"https://rruff.info/search.php?mineral={mineral_name}&property=Raman"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 查找拉曼光谱链接
        raman_links = soup.find_all('a', href=re.compile(r'raman\.php|spectrum\.php'))
        
        print(f"找到 {len(raman_links)} 个拉曼光谱链接 for {mineral_name}")
        
        spectra_data = []
        # 获取前5个光谱作为示例
        for link in raman_links[:5]:
            spectrum_url = urljoin("https://rruff.info/", link.get('href', ''))
            spectrum_data = download_raman_spectrum(spectrum_url)
            if spectrum_data is not None:
                spectra_data.append(spectrum_data)
        
        return spectra_data
        
    except requests.exceptions.RequestException as e:
        print(f"获取 {mineral_name} 的光谱数据时出错: {e}")
        return []
    except Exception as e:
        print(f"解析 {mineral_name} 的光谱数据时出错: {e}")
        return []

def download_raman_spectrum(spectrum_url):
    """
    从URL下载拉曼光谱数据
    
    参数:
    spectrum_url : str
        光谱数据URL
        
    返回:
    tuple : (wavenumber, intensity) 或 None
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(spectrum_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # 尝试解析数据
        # RRUFF数据库通常提供CSV或TXT格式的下载链接
        if spectrum_url.endswith('.txt') or spectrum_url.endswith('.csv'):
            # 直接下载数据文件
            content = response.content.decode('utf-8')
            lines = content.strip().split('\n')
            
            wavenumber = []
            intensity = []
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        w = float(parts[0])
                        i = float(parts[1])
                        wavenumber.append(w)
                        intensity.append(i)
                    except ValueError:
                        continue
            
            if len(wavenumber) > 0 and len(intensity) > 0:
                return (np.array(wavenumber), np.array(intensity))
        
        return None
        
    except Exception as e:
        print(f"下载光谱数据时出错 {spectrum_url}: {e}")
        return None

def load_real_rruff_data(max_samples=1000):
    """
    从RRUFF数据库加载真实拉曼光谱数据
    
    参数:
    max_samples : int
        最大样本数量
        
    返回:
    spectra_data : list of tuples
        光谱数据列表，每个元素为(波数, 强度)元组
    labels : list
        标签列表
    """
    print("正在从RRUFF数据库获取真实拉曼光谱数据...")
    
    # 探索矿物种类
    minerals = explore_rruff_minerals()
    
    if not minerals:
        print("无法获取矿物信息，使用模拟数据")
        return load_sample_spectra(1000)  # 返回较大的模拟数据集
    
    spectra_data = []
    labels = []
    
    # 为每个矿物获取光谱数据
    for mineral_name, mineral_url in minerals[:min(10, len(minerals))]:  # 限制前10个矿物
        print(f"正在获取 {mineral_name} 的拉曼光谱数据...")
        mineral_spectra = get_raman_spectra_for_mineral(mineral_name)
        
        for spectrum in mineral_spectra:
            if spectrum is not None and len(spectrum[0]) > 0:
                spectra_data.append(spectrum)
                labels.append(mineral_name)
                
                if len(spectra_data) >= max_samples:
                    break
        
        if len(spectra_data) >= max_samples:
            break
    
    if len(spectra_data) == 0:
        print("无法从RRUFF数据库获取真实数据，使用模拟数据")
        return load_sample_spectra(1000)
    
    print(f"成功获取 {len(spectra_data)} 个真实拉曼光谱样本")
    return spectra_data, labels

def load_sample_spectra(sample_size=300):
    """
    加载示例光谱数据用于演示
    
    参数:
    sample_size : int
        每个类别的样本数量
        
    返回:
    spectra_data : list of tuples
        光谱数据列表，每个元素为(波数, 强度)元组
    labels : list
        标签列表
    """
    # 生成模拟数据
    np.random.seed(42)
    
    # 创建3类拉曼光谱数据
    spectra_data = []
    labels = []
    
    # 类别1 - 模拟石英矿物
    for i in range(sample_size):
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
    for i in range(sample_size):
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
    for i in range(sample_size):
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

def preprocess_spectra(spectra_data, visualize=False):
    """
    对光谱数据进行预处理
    
    参数:
    spectra_data : list of tuples
        光谱数据列表，每个元素为(波数, 强度)元组
    visualize : bool
        是否可视化预处理效果
        
    返回:
    processed_spectra : list of tuples
        预处理后的光谱数据列表
    """
    processed_spectra = []
    
    for i, (wavenumber, intensity) in enumerate(spectra_data):
        # 对每个光谱进行预处理
        processed_wavenumber, processed_intensity = preprocess_spectrum(
            wavenumber, intensity,
            baseline_correct=True,
            remove_noise_flag=True,
            normalize=True
        )
        processed_spectra.append((processed_wavenumber, processed_intensity))
        
        # 可视化第一个样本的预处理效果
        if visualize and i == 0:
            visualize_preprocessing_steps(wavenumber, intensity)
    
    return processed_spectra

def demonstrate_augmentation(spectra_data, labels):
    """
    演示数据增强效果
    
    参数:
    spectra_data : list of tuples
        光谱数据列表
    labels : list
        标签列表
    """
    # 选择一个样本进行增强演示
    wavenumber, intensity = spectra_data[0]
    label = labels[0]
    
    print(f"正在演示 '{label}' 类别的数据增强效果...")
    
    # 可视化各种增强技术
    visualize_augmentation_techniques(wavenumber, intensity)
    
    # 生成增强样本
    augmented_spectra = augment_spectrum(wavenumber, intensity, num_augmentations=5)
    
    # 可视化增强效果
    from data_augmentation import visualize_augmentation
    visualize_augmentation(wavenumber, intensity, augmented_spectra, f"'{label}' 数据增强效果")

def train_and_evaluate_models(spectra_data, labels, visualize=False):
    """
    训练和评估模型
    
    参数:
    spectra_data : list of tuples
        光谱数据列表
    labels : list
        标签列表
    visualize : bool
        是否可视化训练结果
    """
    print("开始训练和评估模型...")
    
    # 数据增强
    print("正在进行数据增强...")
    # 为大数据集调整增强因子
    augmentation_factor = 5 if len(spectra_data) < 500 else 2
    augmented_data, augmented_labels = generate_augmented_dataset(
        spectra_data, labels, augmentation_factor=augmentation_factor
    )
    print(f"原始数据量: {len(spectra_data)}")
    print(f"增强后数据量: {len(augmented_data)}")
    
    if visualize:
        # 可视化增强后的数据集 (只显示一部分以提高效率)
        print("正在可视化增强后的数据集...")
        display_size = min(100, len(augmented_data))
        visualize_multiple_spectra(augmented_data[:display_size], 
                                 augmented_labels[:display_size] if augmented_labels else None, 
                                 "增强后的光谱数据示例")
    
    # 训练SVM模型
    print("\n训练SVM模型...")
    svm_classifier = RamanSpectrumClassifier(model_type='svm')
    svm_train_score, svm_test_score = svm_classifier.train(augmented_data, augmented_labels)
    svm_predictions = svm_classifier.evaluate()
    
    if visualize:
        # 绘制SVM模型的ROC曲线
        try:
            svm_classifier.plot_roc_curves(list(set(augmented_labels)))
        except:
            print("无法绘制SVM的ROC曲线")
    
    # 训练随机森林模型
    print("\n训练随机森林模型...")
    rf_classifier = RamanSpectrumClassifier(model_type='rf')
    rf_train_score, rf_test_score = rf_classifier.train(augmented_data, augmented_labels)
    rf_predictions = rf_classifier.evaluate()
    
    if visualize:
        # 绘制随机森林模型的ROC曲线
        try:
            rf_classifier.plot_roc_curves(list(set(augmented_labels)))
        except:
            print("无法绘制随机森林的ROC曲线")
    
    # 训练神经网络模型
    print("\n训练神经网络模型...")
    from neural_network import RamanSpectrumNNClassifier
    X = np.array([intensity for _, intensity in augmented_data])
    num_classes = len(np.unique(augmented_labels))
    nn_classifier = RamanSpectrumNNClassifier(X.shape[1], num_classes, list(np.unique(augmented_labels)))
    nn_train_score, nn_test_score = nn_classifier.train(augmented_data, augmented_labels, epochs=50, verbose=0)
    nn_predictions = nn_classifier.evaluate()
    
    if visualize:
        # 绘制神经网络训练历史
        try:
            nn_classifier.plot_training_history()
        except:
            print("无法绘制神经网络训练历史")
    
    # 训练CNN模型
    print("\n训练CNN模型...")
    from cnn_model import RamanSpectrumCNNClassifier
    input_length = X.shape[1]
    cnn_classifier = RamanSpectrumCNNClassifier(input_length, num_classes, list(np.unique(augmented_labels)))
    cnn_train_score, cnn_test_score = cnn_classifier.train(augmented_data, augmented_labels, epochs=50, verbose=0)
    cnn_predictions = cnn_classifier.evaluate()
    
    if visualize:
        # 绘制CNN训练历史
        try:
            cnn_classifier.plot_training_history()
        except:
            print("无法绘制CNN训练历史")
    
    # 比较模型性能
    print("\n模型性能比较:")
    print(f"SVM 训练集准确率: {svm_train_score:.4f}, 测试集准确率: {svm_test_score:.4f}")
    print(f"随机森林 训练集准确率: {rf_train_score:.4f}, 测试集准确率: {rf_test_score:.4f}")
    print(f"神经网络 训练集准确率: {nn_train_score:.4f}, 测试集准确率: {nn_test_score:.4f}")
    print(f"CNN 训练集准确率: {cnn_train_score:.4f}, 测试集准确率: {cnn_test_score:.4f}")
    
    if visualize:
        # 可视化模型比较
        models_dict = {
            'SVM': svm_classifier.model,
            'Random Forest': rf_classifier.model
        }
        try:
            svm_classifier.compare_models(models_dict, augmented_data, augmented_labels)
        except:
            print("无法比较传统模型性能")
        
        # 比较CNN与其它方法
        try:
            from cnn_model import compare_cnn_with_other_models
            compare_cnn_with_other_models(augmented_data[:300], augmented_labels[:300])  # 使用部分数据以提高效率
        except:
            print("无法比较CNN与其它方法性能")
    
    return svm_classifier, rf_classifier, nn_classifier, cnn_classifier

def visualize_dataset_overview(spectra_data, labels):
    """
    可视化数据集概览
    
    参数:
    spectra_data : list of tuples
        光谱数据列表
    labels : list
        标签列表
    """
    print("正在可视化数据集概览...")
    
    # 可视化原始光谱数据 (限制显示数量以提高效率)
    display_size = min(50, len(spectra_data))
    visualize_multiple_spectra(spectra_data[:display_size], labels[:display_size], 
                              f"原始光谱数据集概览 (显示前{display_size}个样本)")
    
    # 显示标签分布
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(unique_labels, counts, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title('数据集标签分布')
    plt.xlabel('矿物类别')
    plt.ylabel('样本数量')
    plt.xticks(rotation=45)
    
    # 在柱状图上添加数值标签
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print("数据集统计信息:")
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count} 个样本")

def main(sample_size=300, use_real_data=False):
    """
    主函数：分析RRUFF数据库中的拉曼光谱数据
    
    参数:
    sample_size : int
        每个类别的样本数量
    use_real_data : bool
        是否尝试从RRUFF数据库获取真实数据
    """
    print("=== RRUFF数据库拉曼光谱数据分析 ===")
    
    # 1. 获取数据库统计信息
    print("\n1. 获取数据库统计信息...")
    stats = get_rruff_statistics()
    if stats:
        print("统计信息:")
        if stats['general_stats']:
            for stat in stats['general_stats'][:5]:  # 只显示前5条
                print(f"  - {stat}")
        print(f"  统计链接数量: {stats['stat_links']}")
        print(f"  拉曼相关链接数量: {stats['raman_links']}")
    
    # 2. 探索搜索功能
    print("\n2. 探索搜索功能...")
    explore_rruff_search()
    
    # 3. 探索矿物种类
    print("\n3. 探索RRUFF数据库中的矿物种类...")
    explore_rruff_minerals()
    
    # 4. 尝试获取示例数据
    print("\n4. 尝试获取示例拉曼光谱数据...")
    get_sample_raman_data()
    
    # 5. 加载光谱数据
    if use_real_data:
        print(f"\n5. 从RRUFF数据库加载真实拉曼光谱数据...")
        spectra_data, labels = load_real_rruff_data(max_samples=sample_size*3)  # 总样本数
    else:
        print(f"\n5. 加载示例光谱数据 (每类{sample_size}个样本)...")
        spectra_data, labels = load_sample_spectra(sample_size)
    
    total_samples = len(spectra_data)
    print(f"成功加载 {total_samples} 个光谱样本")
    unique_labels = list(set(labels))
    print(f"标签类别: {unique_labels}")
    
    # 为了演示目的，如果数据量太大，可视化时只显示一部分
    visualization_sample_size = min(100, total_samples // len(unique_labels))
    
    # 6. 数据集概览可视化
    print("\n6. 数据集概览可视化...")
    visualize_dataset_overview(spectra_data[:min(50, len(spectra_data))], 
                              labels[:min(50, len(spectra_data))])
    
    # 7. 数据预处理 (只对一部分数据进行可视化以提高效率)
    print("\n7. 数据预处理...")
    processed_spectra = preprocess_spectra(spectra_data[:50], visualize=True)  # 只可视化前50个
    # 对所有数据进行预处理
    all_processed_spectra = preprocess_spectra(spectra_data)
    print("数据预处理完成")
    
    # 8. 数据增强演示 (只使用原始数据的一部分)
    print("\n8. 数据增强演示...")
    demonstrate_augmentation(spectra_data[:10], labels[:10])
    
    # 9. 训练和评估模型 (使用所有数据)
    print("\n9. 模型训练和评估...")
    svm_model, rf_model, nn_model, cnn_model = train_and_evaluate_models(all_processed_spectra, labels, visualize=True)
    
    print("\n分析完成。")

if __name__ == "__main__":
    # 可以通过命令行参数调整样本大小
    import sys
    sample_size = 1000  # 默认总样本数1000个
    use_real_data = False
    
    for arg in sys.argv[1:]:
        if arg.startswith('--size='):
            try:
                sample_size = int(arg.split('=')[1])
            except ValueError:
                print("无效的样本大小参数，使用默认值1000")
        elif arg == '--real':
            use_real_data = True
        else:
            try:
                sample_size = int(arg)
            except ValueError:
                print(f"未知参数: {arg}")
    
    main(sample_size, use_real_data)
