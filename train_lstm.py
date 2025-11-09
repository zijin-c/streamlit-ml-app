"""
LSTM模型训练脚本
使用PyTorch实现，贝叶斯优化进行超参数调优
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
import pickle
from data_processor import DataProcessor
from metrics import calculate_metrics_multioutput, print_metrics
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """LSTM回归模型"""
    
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        """
        Args:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏层单元数
            output_size: 输出维度
            dropout: Dropout比率
        """
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # 如果是2D，添加序列维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        out = self.relu(last_output)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class LSTMTrainer:
    """LSTM训练器"""
    
    def __init__(self, input_size, output_size, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.model = None
        self.best_params = None
        
        print(f"\n{'='*60}")
        print(f"使用设备: {self.device}")
        if self.device == 'cuda':
            print(f"GPU型号: {torch.cuda.get_device_name(0)}")
            print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"批次大小: 512 (已优化以充分利用GPU)")
        else:
            print("⚠️  未检测到GPU，将使用CPU训练")
            print("建议: 安装CUDA版PyTorch以使用GPU加速")
        print(f"{'='*60}\n")
    
    def train_model(self, X_train, y_train, hidden_size, learning_rate, l2_reg, 
                   batch_size=256, epochs=1000, patience=50, verbose=False):
        """
        训练单个模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            hidden_size: LSTM隐藏单元数
            learning_rate: 学习率
            l2_reg: L2正则化系数
            batch_size: 批大小
            epochs: 训练轮数
            patience: 早停耐心值
            verbose: 是否打印详细信息
            
        Returns:
            训练好的模型和最佳验证损失
        """
        # 划分训练集和验证集
        n_train = int(0.8 * len(X_train))
        X_train_split = X_train[:n_train]
        y_train_split = y_train[:n_train]
        X_val = X_train[n_train:]
        y_val = y_train[n_train:]
        
        # 确保batch_size是整数且不超过数据集大小
        batch_size = int(batch_size)
        batch_size = min(batch_size, len(X_train_split), max(1, len(X_val)))
        
        # 创建数据加载器
        train_dataset = TimeSeriesDataset(X_train_split, y_train_split)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型 (确保hidden_size是Python原生int类型)
        hidden_size = int(hidden_size)
        model = LSTMModel(self.input_size, hidden_size, self.output_size).to(self.device)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50
        )
        
        # 训练
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练模式
            model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_losses.append(loss.item())
            
            # 验证模式
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            
            # 更新学习率
            scheduler.step(val_loss)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"早停于 epoch {epoch+1}")
                    break
        
        # 加载最佳模型
        model.load_state_dict(best_model_state)
        
        return model, best_val_loss
    
    def bayesian_optimization(self, X_train, y_train, n_calls=15, verbose=True):
        """
        贝叶斯优化超参数
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            n_calls: 优化迭代次数
            verbose: 是否显示优化过程
        """
        # 定义搜索空间
        space = [
            Integer(20, 50, name='hidden_size'),
            Real(1e-3, 1e-2, prior='log-uniform', name='learning_rate'),
            Real(1e-4, 1e-3, prior='log-uniform', name='l2_reg'),
        ]
        
        @use_named_args(space)
        def objective(**params):
            """优化目标函数"""
            if verbose:
                print(f"\n尝试参数: {params}")
            
            model, val_loss = self.train_model(
                X_train, y_train,
                hidden_size=params['hidden_size'],
                learning_rate=params['learning_rate'],
                l2_reg=params['l2_reg'],
                batch_size=512,
                epochs=1000,
                patience=50,
                verbose=False
            )
            
            if verbose:
                print(f"验证损失: {val_loss:.6f}")
            
            return val_loss
        
        # 执行贝叶斯优化
        print("开始贝叶斯优化...")
        result = gp_minimize(objective, space, n_calls=n_calls, random_state=42, verbose=verbose)
        
        # 提取最优参数
        self.best_params = {
            'hidden_size': result.x[0],
            'learning_rate': result.x[1],
            'l2_reg': result.x[2],
        }
        
        print(f"\n最优参数:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        print(f"  最优验证损失: {result.fun:.6f}")
        
        return self.best_params
    
    def fit(self, X_train, y_train, n_calls=15):
        """
        完整训练流程：贝叶斯优化 + 最终训练
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            n_calls: 贝叶斯优化迭代次数
        """
        print("\n" + "="*60)
        print("第1步: 贝叶斯优化超参数")
        print("="*60)
        
        # 贝叶斯优化
        self.bayesian_optimization(X_train, y_train, n_calls=n_calls)
        
        print("\n" + "="*60)
        print("第2步: 使用最优参数训练最终模型")
        print("="*60)
        
        # 使用最优参数训练最终模型
        self.model, final_loss = self.train_model(
            X_train, y_train,
            hidden_size=self.best_params['hidden_size'],
            learning_rate=self.best_params['learning_rate'],
            l2_reg=self.best_params['l2_reg'],
            batch_size=512,
            epochs=1000,
            patience=100,
            verbose=True
        )
        
        print(f"\n最终模型训练完成，验证损失: {final_loss:.6f}")
    
    def predict(self, X):
        """
        预测
        
        Args:
            X: 特征
            
        Returns:
            预测结果
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()
    
    def save(self, filepath='models/lstm_model.pkl'):
        """保存模型"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_params': self.best_params,
            'input_size': self.input_size,
            'output_size': self.output_size,
        }, filepath)
        
        print(f"\n模型已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath='models/lstm_model.pkl', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        
        trainer = cls(checkpoint['input_size'], checkpoint['output_size'], device)
        trainer.best_params = checkpoint['best_params']
        
        # 确保所有参数都是Python原生类型
        hidden_size = int(checkpoint['best_params']['hidden_size'])
        
        # 重建模型
        trainer.model = LSTMModel(
            checkpoint['input_size'],
            hidden_size,
            checkpoint['output_size']
        ).to(device)
        
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        
        return trainer


def plot_predictions(y_true, y_pred, n_outputs, dataset_name, metrics_dict, save_path=None):
    """绘制预测结果对比图"""
    fig, axes = plt.subplots(n_outputs, 1, figsize=(12, 3*n_outputs))
    if n_outputs == 1:
        axes = [axes]
    
    for i in range(n_outputs):
        ax = axes[i]
        n_samples = len(y_true)
        x = np.arange(1, n_samples + 1)
        
        ax.plot(x, y_true[:, i], 'r-*', label='真实值', linewidth=1, markersize=4)
        ax.plot(x, y_pred[:, i], 'b-o', label='Bayes-LSTM预测值', linewidth=1, markersize=4)
        
        metrics = metrics_dict[f'output_{i+1}']
        ax.set_xlabel('预测样本')
        ax.set_ylabel('预测结果')
        ax.set_title(f'{dataset_name}预测结果对比 - 输出 {i+1}; RMSE={metrics["RMSE"]:.4f}')
        ax.legend()
        ax.grid(True)
        ax.set_xlim([1, n_samples])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_scatter(y_true, y_pred, n_outputs, dataset_name, metrics_dict, save_path=None):
    """绘制散点图和拟合线"""
    fig, axes = plt.subplots(n_outputs, 1, figsize=(8, 5*n_outputs))
    if n_outputs == 1:
        axes = [axes]
    
    for i in range(n_outputs):
        ax = axes[i]
        metrics = metrics_dict[f'output_{i+1}']
        
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=30)
        
        z = np.polyfit(y_true[:, i], y_pred[:, i], 1)
        p = np.poly1d(z)
        x_line = np.linspace(y_true[:, i].min(), y_true[:, i].max(), 100)
        ax.plot(x_line, p(x_line), 'm-', linewidth=2, label=f'拟合线: y={z[0]:.2f}x+{z[1]:.2f}')
        
        ax.plot([y_true[:, i].min(), y_true[:, i].max()], 
                [y_true[:, i].min(), y_true[:, i].max()], 
                'k--', linewidth=1, label='理想线')
        
        ax.set_xlabel('真实值')
        ax.set_ylabel('预测值')
        ax.set_title(f'{dataset_name}线性拟合图 - 输出 {i+1}; R²={metrics["R2"]:.4f}, RMSE={metrics["RMSE"]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主函数"""
    print("="*60)
    print("LSTM多输出回归模型训练")
    print("="*60)
    
    # 1. 加载和预处理数据
    print("\n[1] 加载数据...")
    processor = DataProcessor(data_path='去噪后数据.xlsx', train_ratio=0.7, n_outputs=5)
    data_dict = processor.prepare_data()
    
    print(f"训练集样本数: {data_dict['n_train']}")
    print(f"测试集样本数: {data_dict['n_test']}")
    print(f"特征维度: {data_dict['n_features']}")
    print(f"输出维度: {processor.n_outputs}")
    
    # 2. 训练模型
    print("\n[2] 开始训练LSTM模型...")
    trainer = LSTMTrainer(
        input_size=data_dict['n_features'],
        output_size=processor.n_outputs
    )
    
    trainer.fit(
        data_dict['X_train_norm'],
        data_dict['y_train_norm'],
        n_calls=15  # 贝叶斯优化迭代次数
    )
    
    # 3. 预测
    print("\n[3] 进行预测...")
    y_train_pred_norm = trainer.predict(data_dict['X_train_norm'])
    y_test_pred_norm = trainer.predict(data_dict['X_test_norm'])
    
    # 反归一化
    y_train_pred = processor.inverse_transform_output(y_train_pred_norm)
    y_test_pred = processor.inverse_transform_output(y_test_pred_norm)
    
    # 4. 评估
    print("\n[4] 评估模型性能...")
    train_metrics = calculate_metrics_multioutput(data_dict['y_train'], y_train_pred)
    test_metrics = calculate_metrics_multioutput(data_dict['y_test'], y_test_pred)
    
    print_metrics(train_metrics, "训练集评估指标")
    print_metrics(test_metrics, "测试集评估指标")
    
    # 5. 保存模型
    print("\n[5] 保存模型...")
    trainer.save('models/lstm_model.pkl')
    
    # 保存预测结果
    import pandas as pd
    import os
    os.makedirs('results', exist_ok=True)
    
    train_pred_df = pd.DataFrame(y_train_pred, columns=[f'预测维度_{i+1}' for i in range(5)])
    test_pred_df = pd.DataFrame(y_test_pred, columns=[f'预测维度_{i+1}' for i in range(5)])
    
    train_pred_df.to_excel('results/lstm_train_predictions.xlsx', index=False)
    test_pred_df.to_excel('results/lstm_test_predictions.xlsx', index=False)
    print("预测结果已保存到 results/ 目录")
    
    # 6. 可视化
    print("\n[6] 生成可视化图表...")
    os.makedirs('results/figures', exist_ok=True)
    
    plot_predictions(data_dict['y_train'], y_train_pred, 5, '训练集', 
                    train_metrics, 'results/figures/lstm_train_prediction.png')
    plot_predictions(data_dict['y_test'], y_test_pred, 5, '测试集', 
                    test_metrics, 'results/figures/lstm_test_prediction.png')
    
    plot_scatter(data_dict['y_train'], y_train_pred, 5, '训练集',
                train_metrics, 'results/figures/lstm_train_scatter.png')
    plot_scatter(data_dict['y_test'], y_test_pred, 5, '测试集',
                test_metrics, 'results/figures/lstm_test_scatter.png')
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)


if __name__ == '__main__':
    main()

