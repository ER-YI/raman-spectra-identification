# -*- coding: utf-8 -*-
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

class RamanSpectrumClassifier:
    """
    拉曼光谱分类器
    """
    def __init__(self, model_type='svm'):
        """
        初始化分类器
        
        参数:
        model_type : str
            模型类型 ('svm' 或 'rf')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # 根据模型类型初始化模型
        if model_type == 'svm':
            self.model = SVC(probability=True)
        elif model_type == 'rf':
            self.model = RandomForestClassifier()
        else:
            raise ValueError("不支持的模型类型，请选择 'svm' 或 'rf'")
    
    def prepare_data(self, spectra_data, labels):
        """
        准备训练数据
        
        参数:
        spectra_data : list of tuples
            光谱数据列表，每个元素为(波数, 强度)元组
        labels : list
            标签列表
            
        返回:
        X : array-like
            特征矩阵
        y : array-like
            标签向量
        """
        # 提取强度作为特征
        X = np.array([intensity for _, intensity in spectra_data])
        y = np.array(labels)
        
        return X, y
    
    def train(self, spectra_data, labels, test_size=0.2, random_state=42):
        """
        训练模型
        
        参数:
        spectra_data : list of tuples
            光谱数据列表
        labels : list
            标签列表
        test_size : float
            测试集比例
        random_state : int
            随机种子
        """
        # 准备数据
        X, y = self.prepare_data(spectra_data, labels)
        
        # 分割训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 标准化特征
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # 训练模型
        self.model.fit(self.X_train_scaled, self.y_train)
        self.is_trained = True
        
        # 输出训练结果
        train_score = self.model.score(self.X_train_scaled, self.y_train)
        test_score = self.model.score(self.X_test_scaled, self.y_test)
        
        print(f"{self.model_type.upper()} 模型训练完成:")
        print(f"  训练集准确率: {train_score:.4f}")
        print(f"  测试集准确率: {test_score:.4f}")
        
        return train_score, test_score
    
    def cross_validate(self, spectra_data, labels, cv=5):
        """
        交叉验证
        
        参数:
        spectra_data : list of tuples
            光谱数据列表
        labels : list
            标签列表
        cv : int
            交叉验证折数
            
        返回:
        scores : array
            交叉验证得分
        """
        X, y = self.prepare_data(spectra_data, labels)
        X_scaled = self.scaler.fit_transform(X)
        
        scores = cross_val_score(self.model, X_scaled, y, cv=cv)
        
        print(f"{self.model_type.upper()} 模型 {cv} 折交叉验证结果:")
        print(f"  平均准确率: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def hyperparameter_tuning(self, spectra_data, labels, param_grid=None, cv=5):
        """
        超参数调优
        
        参数:
        spectra_data : list of tuples
            光谱数据列表
        labels : list
            标签列表
        param_grid : dict
            参数网格
        cv : int
            交叉验证折数
        """
        X, y = self.prepare_data(spectra_data, labels)
        X_scaled = self.scaler.fit_transform(X)
        
        # 默认参数网格
        if param_grid is None:
            if self.model_type == 'svm':
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'kernel': ['rbf', 'poly', 'sigmoid']
                }
            elif self.model_type == 'rf':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
        
        # 网格搜索
        grid_search = GridSearchCV(self.model, param_grid, cv=cv, n_jobs=-1, verbose=1)
        grid_search.fit(X_scaled, y)
        
        # 更新模型为最佳模型
        self.model = grid_search.best_estimator_
        
        print(f"{self.model_type.upper()} 模型超参数调优完成:")
        print(f"  最佳参数: {grid_search.best_params_}")
        print(f"  最佳得分: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_, grid_search.best_score_
    
    def evaluate(self):
        """
        评估模型性能
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用 train() 方法")
        
        # 预测
        y_pred = self.model.predict(self.X_test_scaled)
        
        # 分类报告
        print(f"\n{self.model_type.upper()} 模型详细评估结果:")
        print(classification_report(self.y_test, y_pred))
        
        # 混淆矩阵
        cm = confusion_matrix(self.y_test, y_pred)
        
        # 绘制混淆矩阵
        self.plot_confusion_matrix(cm)
        
        return y_pred
    
    def plot_confusion_matrix(self, cm, title=None):
        """
        绘制混淆矩阵
        
        参数:
        cm : array-like
            混淆矩阵
        title : str
            图表标题
        """
        if title is None:
            title = f'{self.model_type.upper()} 模型混淆矩阵'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.show()
    
    def plot_training_history(self, train_scores, val_scores=None, title=None):
        """
        绘制训练历史（准确率变化）
        
        参数:
        train_scores : list
            训练集得分列表
        val_scores : list
            验证集得分列表
        title : str
            图表标题
        """
        if title is None:
            title = f'{self.model_type.upper()} 模型训练历史'
        
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_scores) + 1)
        
        plt.plot(epochs, train_scores, 'bo-', label='训练集准确率')
        if val_scores is not None:
            plt.plot(epochs, val_scores, 'ro-', label='验证集准确率')
        
        plt.title(title)
        plt.xlabel('训练轮次')
        plt.ylabel('准确率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_roc_curves(self, classes=None):
        """
        绘制ROC曲线
        
        参数:
        classes : list
            类别标签列表
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用 train() 方法")
        
        # 获取预测概率
        y_pred_proba = self.model.predict_proba(self.X_test_scaled)
        
        # 如果没有提供类别标签，则从训练数据中获取
        if classes is None:
            classes = np.unique(self.y_train)
        
        # 将标签二值化
        y_test_bin = label_binarize(self.y_test, classes=classes)
        n_classes = y_test_bin.shape[1]
        
        # 计算每个类别的ROC曲线和AUC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # 处理二分类和多分类情况
        if n_classes == 2:
            fpr[0], tpr[0], _ = roc_curve(y_test_bin, y_pred_proba[:, 1])
            roc_auc[0] = auc(fpr[0], tpr[0])
        else:
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
        
        # 绘制ROC曲线
        plt.figure(figsize=(10, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple'])
        
        if n_classes == 2:
            plt.plot(fpr[0], tpr[0], color='darkorange', lw=2,
                    label=f'ROC曲线 (AUC = {roc_auc[0]:.2f})')
        else:
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率')
        plt.ylabel('真正率')
        plt.title(f'{self.model_type.upper()} 模型ROC曲线')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def compare_models(self, models_dict, spectra_data, labels):
        """
        比较多个模型的性能
        
        参数:
        models_dict : dict
            模型字典，格式为 {'model_name': model_instance}
        spectra_data : list of tuples
            光谱数据列表
        labels : list
            标签列表
        """
        # 准备数据
        X, y = self.prepare_data(spectra_data, labels)
        X_scaled = self.scaler.fit_transform(X)
        
        # 交叉验证比较
        model_scores = {}
        for name, model in models_dict.items():
            scores = cross_val_score(model, X_scaled, y, cv=5)
            model_scores[name] = scores
        
        # 绘制模型比较图
        plt.figure(figsize=(10, 6))
        model_names = list(model_scores.keys())
        scores_list = [model_scores[name] for name in model_names]
        
        plt.boxplot(scores_list, labels=model_names)
        plt.title('模型性能比较')
        plt.ylabel('交叉验证准确率')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # 打印详细结果
        print("模型性能比较结果:")
        for name, scores in model_scores.items():
            print(f"  {name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    def predict(self, spectra_data):
        """
        预测新样本
        
        参数:
        spectra_data : list of tuples
            待预测的光谱数据列表
            
        返回:
        predictions : array
            预测结果
        probabilities : array
            预测概率（如果模型支持）
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用 train() 方法")
        
        # 提取强度特征
        X = np.array([intensity for _, intensity in spectra_data])
        X_scaled = self.scaler.transform(X)
        
        # 预测
        predictions = self.model.predict(X_scaled)
        probabilities = None
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities

    def save_model(self, filepath):
        """
        保存模型到文件
        
        参数:
        filepath : str
            模型保存路径
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存")
        
        import pickle
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'model_type': self.model_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """
        从文件加载模型
        
        参数:
        filepath : str
            模型文件路径
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        self.model_type = model_data['model_type']
        
        print(f"模型已从 {filepath} 加载")

def visualize_multiple_spectra(spectra_data, labels=None, title="多光谱可视化"):
    """
    可视化多个光谱
    
    参数:
    spectra_data : list of tuples
        光谱数据列表，每个元素为(波数, 强度)元组
    labels : list
        标签列表
    title : str
        图表标题
    """
    plt.figure(figsize=(12, 8))
    
    # 为不同类别使用不同颜色
    unique_labels = list(set(labels)) if labels else [None]
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    if labels:
        # 按类别绘制光谱
        for i, (label, color) in enumerate(zip(unique_labels, colors)):
            indices = [j for j, l in enumerate(labels) if l == label]
            for idx in indices[:10]:  # 每个类别最多显示10个样本
                wavenumber, intensity = spectra_data[idx]
                plt.plot(wavenumber, intensity, linewidth=0.8, 
                        color=color, alpha=0.7, label=label if idx == indices[0] else "")
    else:
        # 不分类别直接绘制
        for wavenumber, intensity in spectra_data[:50]:  # 最多显示50个样本
            plt.plot(wavenumber, intensity, linewidth=0.8, alpha=0.7)
    
    plt.title(title)
    plt.xlabel('波数 (cm⁻¹)')
    plt.ylabel('强度')
    plt.grid(True, alpha=0.3)
    
    if labels:
        plt.legend()
    
    plt.tight_layout()
    plt.show()

# 测试代码
def test_models():
    """
    测试分类器
    """
    # 生成模拟数据
    np.random.seed(42)
    
    # 创建3类拉曼光谱数据
    spectra_data = []
    labels = []
    
    # 类别1
    for _ in range(50):
        wavenumber = np.linspace(200, 2000, 1000)
        intensity = (np.random.normal(100, 10) * np.exp(-0.5*((wavenumber-500)/20)**2) + 
                    np.random.normal(80, 10) * np.exp(-0.5*((wavenumber-1000)/15)**2) + 
                    np.random.normal(20, 5) * np.exp(-0.5*((wavenumber-1600)/25)**2) +
                    np.random.normal(0, 5, len(wavenumber)))
        spectra_data.append((wavenumber, intensity))
        labels.append(0)
    
    # 类别2
    for _ in range(50):
        wavenumber = np.linspace(200, 2000, 1000)
        intensity = (np.random.normal(20, 5) * np.exp(-0.5*((wavenumber-500)/20)**2) + 
                    np.random.normal(100, 10) * np.exp(-0.5*((wavenumber-1000)/15)**2) + 
                    np.random.normal(80, 10) * np.exp(-0.5*((wavenumber-1600)/25)**2) +
                    np.random.normal(0, 5, len(wavenumber)))
        spectra_data.append((wavenumber, intensity))
        labels.append(1)
    
    # 类别3
    for _ in range(50):
        wavenumber = np.linspace(200, 2000, 1000)
        intensity = (np.random.normal(80, 10) * np.exp(-0.5*((wavenumber-500)/20)**2) + 
                    np.random.normal(20, 5) * np.exp(-0.5*((wavenumber-1000)/15)**2) + 
                    np.random.normal(100, 10) * np.exp(-0.5*((wavenumber-1600)/25)**2) +
                    np.random.normal(0, 5, len(wavenumber)))
        spectra_data.append((wavenumber, intensity))
        labels.append(2)
    
    # 可视化原始光谱
    visualize_multiple_spectra(spectra_data, labels, "原始光谱数据")
    
    # 测试SVM分类器
    print("测试SVM分类器:")
    svm_classifier = RamanSpectrumClassifier(model_type='svm')
    svm_classifier.train(spectra_data, labels)
    svm_classifier.evaluate()
    
    # 测试随机森林分类器
    print("\n测试随机森林分类器:")
    rf_classifier = RamanSpectrumClassifier(model_type='rf')
    rf_classifier.train(spectra_data, labels)
    rf_classifier.evaluate()

if __name__ == "__main__":
    test_models()