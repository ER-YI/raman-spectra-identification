# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns

class RamanSpectrumNNClassifier:
    """
    基于神经网络的拉曼光谱分类器
    """
    def __init__(self, input_dim, num_classes, class_names=None):
        """
        初始化神经网络分类器
        
        参数:
        input_dim : int
            输入特征维度（光谱点数）
        num_classes : int
            类别数量
        class_names : list
            类别名称列表
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.class_names = class_names
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.history = None
        self.is_trained = False
        self.device = 'cuda' if tf.config.list_physical_devices('GPU') else 'cpu'
        
    def build_model(self, hidden_layers=[512, 256, 128], dropout_rate=0.3):
        """
        构建神经网络模型
        
        参数:
        hidden_layers : list
            隐藏层神经元数量列表
        dropout_rate : float
            Dropout比率
        """
        model = keras.Sequential()
        
        # 输入层
        model.add(layers.Dense(hidden_layers[0], 
                              activation='relu', 
                              input_shape=(self.input_dim,)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        
        # 隐藏层
        for units in hidden_layers[1:]:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        # 输出层
        if self.num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(self.num_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
        
        # 编译模型
        model.compile(optimizer='adam',
                     loss=loss,
                     metrics=['accuracy'])
        
        self.model = model
        return model
    
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
    
    def train(self, spectra_data, labels, 
              test_size=0.2, 
              validation_split=0.2,
              epochs=100, 
              batch_size=32,
              verbose=1):
        """
        训练神经网络模型
        
        参数:
        spectra_data : list of tuples
            光谱数据列表
        labels : list
            标签列表
        test_size : float
            测试集比例
        validation_split : float
            验证集比例
        epochs : int
            训练轮数
        batch_size : int
            批次大小
        verbose : int
            训练过程显示详细程度
            
        返回:
        train_score : float
            训练集准确率
        test_score : float
            测试集准确率
        """
        # 准备数据
        X, y = self.prepare_data(spectra_data, labels)
        
        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 如果模型未构建，则构建默认模型
        if self.model is None:
            self.build_model()
        
        # 训练模型
        self.history = self.model.fit(
            X_train_scaled, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            verbose=verbose,
            shuffle=True
        )
        
        # 评估模型
        train_loss, train_score = self.model.evaluate(X_train_scaled, y_train, verbose=0)
        test_loss, test_score = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        
        self.X_test_scaled = X_test_scaled
        self.y_test = y_test
        self.is_trained = True
        
        print("神经网络模型训练完成:")
        print(f"  训练集准确率: {train_score:.4f}")
        print(f"  测试集准确率: {test_score:.4f}")
        
        return train_score, test_score
    
    def evaluate(self):
        """
        评估模型性能
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用 train() 方法")
        
        # 预测
        y_pred_prob = self.model.predict(self.X_test_scaled)
        if self.num_classes == 2:
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        else:
            y_pred = np.argmax(y_pred_prob, axis=1)
        
        # 分类报告
        print("\n神经网络模型详细评估结果:")
        print(classification_report(self.y_test, y_pred, 
                                  target_names=self.class_names))
        
        # 混淆矩阵
        cm = confusion_matrix(self.y_test, y_pred)
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
            title = '神经网络模型混淆矩阵'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(title)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.show()
    
    def plot_training_history(self, title=None):
        """
        绘制训练历史
        
        参数:
        title : str
            图表标题
        """
        if self.history is None:
            raise ValueError("模型尚未训练，无训练历史可绘制")
        
        if title is None:
            title = '神经网络模型训练历史'
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 绘制准确率
        ax1.plot(self.history.history['accuracy'], label='训练集准确率')
        if 'val_accuracy' in self.history.history:
            ax1.plot(self.history.history['val_accuracy'], label='验证集准确率')
        ax1.set_title('模型准确率')
        ax1.set_xlabel('训练轮次')
        ax1.set_ylabel('准确率')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制损失
        ax2.plot(self.history.history['loss'], label='训练集损失')
        if 'val_loss' in self.history.history:
            ax2.plot(self.history.history['val_loss'], label='验证集损失')
        ax2.set_title('模型损失')
        ax2.set_xlabel('训练轮次')
        ax2.set_ylabel('损失')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
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
            预测概率
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用 train() 方法")
        
        # 提取强度特征
        X = np.array([intensity for _, intensity in spectra_data])
        X_scaled = self.scaler.transform(X)
        
        # 预测
        probabilities = self.model.predict(X_scaled)
        if self.num_classes == 2:
            predictions = (probabilities > 0.5).astype(int).flatten()
        else:
            predictions = np.argmax(probabilities, axis=1)
        
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
        
        model_data = {
            'model_config': self.model.get_config(),
            'model_weights': self.model.get_weights(),
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'history': self.history
        }
        
        tf.saved_model.save(self.model, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """
        从文件加载模型
        
        参数:
        filepath : str
            模型文件路径
        """
        # 加载模型
        self.model = tf.keras.models.load_model(filepath)
        
        # 加载模型相关信息
        model_data = tf.keras.models.load_model(filepath)
        
        # 获取模型配置和权重
        model_config = self.model.get_config()
        model_weights = self.model.get_weights()
        
        # 解析模型信息
        self.input_dim = model_config['layers'][0]['config']['batch_input_shape'][1]
        self.num_classes = model_weights[-1].shape[1] if self.model.layers[-1].get_config()['units'] > 1 else 1
        self.class_names = [name for name in model_data.__dict__ if name == 'class_names'][0]
        self.scaler = [scaler for scaler in model_data.__dict__ if scaler == 'scaler'][0]
        self.label_encoder = [le for le in model_data.__dict__ if le == 'label_encoder'][0]
        self.history = [hist for hist in model_data.__dict__ if hist == 'history'][0]
        
        self.is_trained = True
        print(f"模型已从 {filepath} 加载")

def compare_nn_with_traditional(spectra_data, labels):
    """
    比较神经网络与传统机器学习方法
    
    参数:
    spectra_data : list of tuples
        光谱数据列表
    labels : list
        标签列表
    """
    from models import RamanSpectrumClassifier
    
    # 准备数据
    X = np.array([intensity for _, intensity in spectra_data])
    num_classes = len(np.unique(labels))
    input_dim = X.shape[1]
    
    # 创建神经网络分类器
    nn_classifier = RamanSpectrumNNClassifier(input_dim, num_classes)
    
    # 训练神经网络
    print("训练神经网络模型...")
    nn_train_score, nn_test_score = nn_classifier.train(
        spectra_data, labels, epochs=50, verbose=0)
    
    # 训练传统模型进行比较
    print("训练SVM模型...")
    svm_classifier = RamanSpectrumClassifier(model_type='svm')
    svm_train_score, svm_test_score = svm_classifier.train(spectra_data, labels)
    
    print("训练随机森林模型...")
    rf_classifier = RamanSpectrumClassifier(model_type='rf')
    rf_train_score, rf_test_score = rf_classifier.train(spectra_data, labels)
    
    # 绘制性能比较图
    plt.figure(figsize=(12, 6))
    
    models = ['神经网络', 'SVM', '随机森林']
    train_scores = [nn_train_score, svm_train_score, rf_train_score]
    test_scores = [nn_test_score, svm_test_score, rf_test_score]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, train_scores, width, label='训练集准确率', alpha=0.8)
    plt.bar(x + width/2, test_scores, width, label='测试集准确率', alpha=0.8)
    
    plt.xlabel('模型类型')
    plt.ylabel('准确率')
    plt.title('不同模型性能比较')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (train_score, test_score) in enumerate(zip(train_scores, test_scores)):
        plt.text(i - width/2, train_score + 0.01, f'{train_score:.3f}', 
                ha='center', va='bottom')
        plt.text(i + width/2, test_score + 0.01, f'{test_score:.3f}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # 打印详细结果
    print("\n模型性能比较结果:")
    print(f"  神经网络 - 训练集准确率: {nn_train_score:.4f}, 测试集准确率: {nn_test_score:.4f}")
    print(f"  SVM - 训练集准确率: {svm_train_score:.4f}, 测试集准确率: {svm_test_score:.4f}")
    print(f"  随机森林 - 训练集准确率: {rf_train_score:.4f}, 测试集准确率: {rf_test_score:.4f}")

# 测试代码
def test_neural_network():
    """
    测试神经网络分类器
    """
    # 生成模拟数据
    np.random.seed(42)
    
    # 创建3类拉曼光谱数据
    spectra_data = []
    labels = []
    
    # 类别1
    for _ in range(100):
        wavenumber = np.linspace(200, 2000, 1000)
        intensity = (np.random.normal(100, 10) * np.exp(-0.5*((wavenumber-500)/20)**2) + 
                    np.random.normal(80, 10) * np.exp(-0.5*((wavenumber-1000)/15)**2) + 
                    np.random.normal(20, 5) * np.exp(-0.5*((wavenumber-1600)/25)**2) +
                    np.random.normal(0, 5, len(wavenumber)))
        spectra_data.append((wavenumber, intensity))
        labels.append('类别1')
    
    # 类别2
    for _ in range(100):
        wavenumber = np.linspace(200, 2000, 1000)
        intensity = (np.random.normal(20, 5) * np.exp(-0.5*((wavenumber-500)/20)**2) + 
                    np.random.normal(100, 10) * np.exp(-0.5*((wavenumber-1000)/15)**2) + 
                    np.random.normal(80, 10) * np.exp(-0.5*((wavenumber-1600)/25)**2) +
                    np.random.normal(0, 5, len(wavenumber)))
        spectra_data.append((wavenumber, intensity))
        labels.append('类别2')
    
    # 类别3
    for _ in range(100):
        wavenumber = np.linspace(200, 2000, 1000)
        intensity = (np.random.normal(80, 10) * np.exp(-0.5*((wavenumber-500)/20)**2) + 
                    np.random.normal(20, 5) * np.exp(-0.5*((wavenumber-1000)/15)**2) + 
                    np.random.normal(100, 10) * np.exp(-0.5*((wavenumber-1600)/25)**2) +
                    np.random.normal(0, 5, len(wavenumber)))
        spectra_data.append((wavenumber, intensity))
        labels.append('类别3')
    
    # 创建神经网络分类器
    X = np.array([intensity for _, intensity in spectra_data])
    nn_classifier = RamanSpectrumNNClassifier(input_dim=X.shape[1], num_classes=3)
    
    # 训练模型
    train_score, test_score = nn_classifier.train(spectra_data, labels, epochs=30, verbose=0)
    
    # 评估模型
    nn_classifier.evaluate()
    
    # 绘制训练历史
    nn_classifier.plot_training_history()
    
    print(f"\n神经网络模型训练完成:")
    print(f"  训练集准确率: {train_score:.4f}")
    print(f"  测试集准确率: {test_score:.4f}")

if __name__ == "__main__":
    test_neural_network()