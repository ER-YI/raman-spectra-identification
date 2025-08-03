# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns

class RamanSpectrumCNN(nn.Module):
    """
    用于拉曼光谱分类的CNN模型
    将一维光谱数据视为图像进行处理
    """
    def __init__(self, input_length, num_classes):
        """
        初始化CNN模型
        
        参数:
        input_length : int
            输入光谱长度
        num_classes : int
            类别数量
        """
        super(RamanSpectrumCNN, self).__init__()
        
        # 将1D光谱数据调整为适合CNN的格式
        # 使用1D卷积处理光谱序列
        self.conv1 = nn.Conv1d(1, 32, kernel_size=21, padding=10)  # 第一个卷积层
        self.conv2 = nn.Conv1d(32, 64, kernel_size=11, padding=5)  # 第二个卷积层
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, padding=3)  # 第三个卷积层
        
        # 批归一化
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        
        # 激活函数和池化
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        
        # 计算全连接层输入维度
        # 经过3次卷积和池化后的输出长度
        conv_output_length = input_length // (2 ** 3)  # 3次池化
        self.fc_input_dim = 128 * conv_output_length
        
        # 全连接层
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
        x : torch.Tensor
            输入张量，形状为(batch_size, input_length)
            
        返回:
        torch.Tensor
            输出张量，形状为(batch_size, num_classes)
        """
        # 调整输入形状为(batch_size, 1, input_length)
        x = x.unsqueeze(1)
        
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # 第三个卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x

class RamanSpectrumCNNClassifier:
    """
    基于CNN的拉曼光谱分类器
    """
    def __init__(self, input_length, num_classes, class_names=None):
        """
        初始化CNN分类器
        
        参数:
        input_length : int
            输入光谱长度
        num_classes : int
            类别数量
        class_names : list
            类别名称列表
        """
        self.input_length = input_length
        self.num_classes = num_classes
        self.class_names = class_names
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def build_model(self):
        """
        构建CNN模型
        """
        self.model = RamanSpectrumCNN(self.input_length, self.num_classes)
        self.model.to(self.device)
        return self.model
    
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
        # 提取强度作为特征，确保所有光谱长度一致
        intensities = []
        for wavenumber, intensity in spectra_data:
            if len(intensity) != self.input_length:
                # 如果长度不匹配，进行插值
                from scipy.interpolate import interp1d
                f = interp1d(np.linspace(0, 1, len(intensity)), intensity, kind='linear')
                new_intensity = f(np.linspace(0, 1, self.input_length))
                intensities.append(new_intensity)
            else:
                intensities.append(intensity)
        
        X = np.array(intensities)
        y = np.array(labels)
        
        return X, y
    
    def train(self, spectra_data, labels, 
              test_size=0.2, 
              validation_split=0.2,
              epochs=100, 
              batch_size=32,
              learning_rate=0.001,
              verbose=1):
        """
        训练CNN模型
        
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
        learning_rate : float
            学习率
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
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.LongTensor(y_test)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 如果模型未构建，则构建默认模型
        if self.model is None:
            self.build_model()
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        
        # 训练模型
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 统计训练信息
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # 更新学习率
            scheduler.step()
            
            # 计算平均损失和准确率
            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            
            # 验证模型
            val_loss, val_acc = self._validate(X_test_tensor, y_test_tensor, criterion)
            
            # 记录历史
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # 打印训练进度
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], '
                      f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # 最终评估
        train_loss, train_score = self._validate(X_train_tensor, y_train_tensor, criterion)
        test_loss, test_score = self._validate(X_test_tensor, y_test_tensor, criterion)
        
        self.X_test_tensor = X_test_tensor
        self.y_test_tensor = y_test_tensor
        self.is_trained = True
        
        print("CNN模型训练完成:")
        print(f"  训练集准确率: {train_score:.4f}")
        print(f"  测试集准确率: {test_score:.4f}")
        
        return train_score, test_score
    
    def _validate(self, X_tensor, y_tensor, criterion):
        """
        验证模型性能
        
        参数:
        X_tensor : torch.Tensor
            输入特征张量
        y_tensor : torch.Tensor
            标签张量
        criterion : torch.nn._Loss
            损失函数
            
        返回:
        loss : float
            平均损失
        accuracy : float
            准确率
        """
        self.model.eval()
        X_tensor, y_tensor = X_tensor.to(self.device), y_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_tensor).sum().item() / y_tensor.size(0)
        
        return loss.item(), accuracy
    
    def evaluate(self):
        """
        评估模型性能
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用 train() 方法")
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = self.X_test_tensor.to(self.device)
            outputs = self.model(X_test_tensor)
            _, y_pred = torch.max(outputs.data, 1)
            y_pred = y_pred.cpu().numpy()
        
        y_test = self.y_test_tensor.numpy()
        
        # 解码标签
        y_test_labels = self.label_encoder.inverse_transform(y_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        
        # 分类报告
        print("\nCNN模型详细评估结果:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.class_names))
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
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
            title = 'CNN模型混淆矩阵'
        
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
        if not self.history['train_loss']:
            raise ValueError("模型尚未训练，无训练历史可绘制")
        
        if title is None:
            title = 'CNN模型训练历史'
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 绘制准确率
        ax1.plot(self.history['train_acc'], label='训练集准确率')
        ax1.plot(self.history['val_acc'], label='验证集准确率')
        ax1.set_title('模型准确率')
        ax1.set_xlabel('训练轮次')
        ax1.set_ylabel('准确率')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制损失
        ax2.plot(self.history['train_loss'], label='训练集损失')
        ax2.plot(self.history['val_loss'], label='验证集损失')
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
        intensities = []
        for wavenumber, intensity in spectra_data:
            if len(intensity) != self.input_length:
                # 如果长度不匹配，进行插值
                from scipy.interpolate import interp1d
                f = interp1d(np.linspace(0, 1, len(intensity)), intensity, kind='linear')
                new_intensity = f(np.linspace(0, 1, self.input_length))
                intensities.append(new_intensity)
            else:
                intensities.append(intensity)
        
        X = np.array(intensities)
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs.data, 1)
            predictions = predictions.cpu().numpy()
            probabilities = probabilities.cpu().numpy()
        
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
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'input_length': self.input_length,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'history': self.history
        }
        
        torch.save(model_data, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """
        从文件加载模型
        
        参数:
        filepath : str
            模型文件路径
        """
        model_data = torch.load(filepath, map_location=self.device)
        
        self.input_length = model_data['input_length']
        self.num_classes = model_data['num_classes']
        self.class_names = model_data['class_names']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.history = model_data['history']
        
        # 重新构建模型
        self.build_model()
        self.model.load_state_dict(model_data['model_state_dict'])
        
        self.is_trained = True
        print(f"模型已从 {filepath} 加载")

def compare_cnn_with_other_models(spectra_data, labels):
    """
    比较CNN与其他模型方法
    
    参数:
    spectra_data : list of tuples
        光谱数据列表
    labels : list
        标签列表
    """
    from models import RamanSpectrumClassifier
    from neural_network import RamanSpectrumNNClassifier
    
    # 准备数据
    X = np.array([intensity for _, intensity in spectra_data])
    num_classes = len(np.unique(labels))
    input_length = X.shape[1]
    
    # 创建CNN分类器
    cnn_classifier = RamanSpectrumCNNClassifier(input_length, num_classes)
    
    # 训练CNN
    print("训练CNN模型...")
    cnn_train_score, cnn_test_score = cnn_classifier.train(
        spectra_data, labels, epochs=50, verbose=0)
    
    # 训练传统模型进行比较
    print("训练SVM模型...")
    svm_classifier = RamanSpectrumClassifier(model_type='svm')
    svm_train_score, svm_test_score = svm_classifier.train(spectra_data, labels)
    
    print("训练随机森林模型...")
    rf_classifier = RamanSpectrumClassifier(model_type='rf')
    rf_train_score, rf_test_score = rf_classifier.train(spectra_data, labels)
    
    # 训练普通神经网络
    print("训练普通神经网络模型...")
    nn_classifier = RamanSpectrumNNClassifier(input_length, num_classes)
    nn_train_score, nn_test_score = nn_classifier.train(spectra_data, labels, epochs=50, verbose=0)
    
    # 绘制性能比较图
    plt.figure(figsize=(12, 6))
    
    models = ['CNN', '普通神经网络', 'SVM', '随机森林']
    train_scores = [cnn_train_score, nn_train_score, svm_train_score, rf_train_score]
    test_scores = [cnn_test_score, nn_test_score, svm_test_score, rf_test_score]
    
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
    print(f"  CNN - 训练集准确率: {cnn_train_score:.4f}, 测试集准确率: {cnn_test_score:.4f}")
    print(f"  普通神经网络 - 训练集准确率: {nn_train_score:.4f}, 测试集准确率: {nn_test_score:.4f}")
    print(f"  SVM - 训练集准确率: {svm_train_score:.4f}, 测试集准确率: {svm_test_score:.4f}")
    print(f"  随机森林 - 训练集准确率: {rf_train_score:.4f}, 测试集准确率: {rf_test_score:.4f}")

# 测试代码
def test_cnn():
    """
    测试CNN分类器
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
    
    # 创建CNN分类器
    cnn_classifier = RamanSpectrumCNNClassifier(input_length=1000, num_classes=3)
    
    # 训练模型
    train_score, test_score = cnn_classifier.train(spectra_data, labels, epochs=30, verbose=0)
    
    # 评估模型
    cnn_classifier.evaluate()
    
    # 绘制训练历史
    cnn_classifier.plot_training_history()
    
    print(f"\nCNN模型训练完成:")
    print(f"  训练集准确率: {train_score:.4f}")
    print(f"  测试集准确率: {test_score:.4f}")

if __name__ == "__main__":
    test_cnn()