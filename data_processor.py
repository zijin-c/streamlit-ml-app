"""
数据处理模块
负责加载、预处理和分割数据
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict


class DataProcessor:
    """数据处理类"""
    
    def __init__(self, data_path: str = '去噪后数据.xlsx', train_ratio: float = 0.7, n_outputs: int = 5):
        """
        初始化数据处理器
        
        Args:
            data_path: 数据文件路径
            train_ratio: 训练集比例
            n_outputs: 输出维度数量
        """
        self.data_path = data_path
        self.train_ratio = train_ratio
        self.n_outputs = n_outputs
        self.input_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.output_scalers = [MinMaxScaler(feature_range=(-1, 1)) for _ in range(n_outputs)]
        
    def load_data(self) -> pd.DataFrame:
        """加载Excel数据"""
        return pd.read_excel(self.data_path)
    
    def split_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        划分训练集和测试集
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        num_samples = len(data)
        num_train = int(num_samples * self.train_ratio)
        
        # 分离特征和标签
        X = data.iloc[:, :-self.n_outputs].values
        y = data.iloc[:, -self.n_outputs:].values
        
        # 划分训练集和测试集
        X_train = X[:num_train]
        X_test = X[num_train:]
        y_train = y[:num_train]
        y_test = y[num_train:]
        
        return X_train, X_test, y_train, y_test
    
    def normalize_data(self, X_train: np.ndarray, X_test: np.ndarray, 
                      y_train: np.ndarray, y_test: np.ndarray) -> Tuple:
        """
        归一化数据
        输入特征归一化到[-1, 1]
        每个输出维度独立归一化到[-1, 1]
        
        Returns:
            X_train_norm, X_test_norm, y_train_norm, y_test_norm
        """
        # 归一化输入特征
        X_train_norm = self.input_scaler.fit_transform(X_train)
        X_test_norm = self.input_scaler.transform(X_test)
        
        # 归一化输出（每个维度独立归一化）
        y_train_norm = np.zeros_like(y_train)
        y_test_norm = np.zeros_like(y_test)
        
        for i in range(self.n_outputs):
            y_train_norm[:, i] = self.output_scalers[i].fit_transform(
                y_train[:, i].reshape(-1, 1)
            ).flatten()
            y_test_norm[:, i] = self.output_scalers[i].transform(
                y_test[:, i].reshape(-1, 1)
            ).flatten()
        
        return X_train_norm, X_test_norm, y_train_norm, y_test_norm
    
    def inverse_transform_output(self, y_pred: np.ndarray, output_idx: int = None) -> np.ndarray:
        """
        反归一化输出
        
        Args:
            y_pred: 预测值（归一化后）
            output_idx: 输出维度索引，如果为None则对所有维度反归一化
        """
        if output_idx is not None:
            # 单个输出维度
            return self.output_scalers[output_idx].inverse_transform(
                y_pred.reshape(-1, 1)
            ).flatten()
        else:
            # 多个输出维度
            y_pred_original = np.zeros_like(y_pred)
            for i in range(self.n_outputs):
                y_pred_original[:, i] = self.output_scalers[i].inverse_transform(
                    y_pred[:, i].reshape(-1, 1)
                ).flatten()
            return y_pred_original
    
    def prepare_data(self) -> Dict:
        """
        完整的数据准备流程
        
        Returns:
            包含所有数据的字典
        """
        # 加载数据
        data = self.load_data()
        
        # 划分数据
        X_train, X_test, y_train, y_test = self.split_data(data)
        
        # 归一化
        X_train_norm, X_test_norm, y_train_norm, y_test_norm = self.normalize_data(
            X_train, X_test, y_train, y_test
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_norm': X_train_norm,
            'X_test_norm': X_test_norm,
            'y_train_norm': y_train_norm,
            'y_test_norm': y_test_norm,
            'n_features': X_train.shape[1],
            'n_train': X_train.shape[0],
            'n_test': X_test.shape[0]
        }


class TimeSeriesDataProcessor(DataProcessor):
    """
    时间序列数据处理类
    用于Transformer模型，将历史数据作为序列特征
    """
    
    def __init__(self, data_path: str = '去噪后数据.xlsx', 
                 train_ratio: float = 0.7, 
                 n_outputs: int = 5,
                 kim: int = 24,  # 历史步长
                 zim: int = 1):  # 预测步长
        super().__init__(data_path, train_ratio, n_outputs)
        self.kim = kim
        self.zim = zim
    
    def create_sequences(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        创建时间序列数据
        使用kim个历史数据点预测zim步后的值
        
        Args:
            data: 原始数据
            
        Returns:
            序列化后的数据
        """
        result = data.values
        num_samples = len(result)
        or_dim = result.shape[1]
        
        sequences = []
        for i in range(num_samples - self.kim - self.zim + 1):
            # 取kim个历史数据作为输入
            seq_input = result[i:i + self.kim, :].flatten()
            # 取预测目标
            seq_output = result[i + self.kim + self.zim - 1, :]
            # 合并
            sequences.append(np.concatenate([seq_input, seq_output]))
        
        # 转换为DataFrame
        return pd.DataFrame(sequences)
    
    def prepare_data(self) -> Dict:
        """重写准备数据方法，加入序列化处理"""
        # 加载数据
        data = self.load_data()
        
        # 创建序列
        seq_data = self.create_sequences(data)
        
        # 调用父类的方法
        return super().prepare_data()

