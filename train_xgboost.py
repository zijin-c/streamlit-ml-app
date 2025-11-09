"""
XGBoost模型训练脚本
使用贝叶斯优化进行超参数调优
"""
import numpy as np
import xgboost as xgb
from skopt import BayesSearchCV
from skopt.space import Integer, Real
import pickle
from data_processor import DataProcessor
from metrics import calculate_metrics, print_metrics
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class XGBoostMultiOutputRegressor:
    """多输出XGBoost回归器"""
    
    def __init__(self, n_outputs: int = 5):
        """
        初始化
        
        Args:
            n_outputs: 输出维度数
        """
        self.n_outputs = n_outputs
        self.models = []
        self.best_params = []
        
    def fit(self, X_train, y_train, n_iter=30, cv=3, verbose=True, tree_method='hist'):
        """
        训练模型（为每个输出维度分别训练）
        
        Args:
            X_train: 训练特征
            y_train: 训练标签 (n_samples, n_outputs)
            n_iter: 贝叶斯优化迭代次数
            cv: 交叉验证折数
            verbose: 是否显示训练过程
            tree_method: 树构建方法 ('hist', 'gpu_hist', 'auto')
        """
        # 定义超参数搜索空间
        search_spaces = {
            'n_estimators': Integer(10, 5000),
            'max_depth': Integer(10, 20),
            'learning_rate': Real(0.0001, 1, prior='log-uniform'),
            'subsample': Real(0.6, 1.0),
            'colsample_bytree': Real(0.6, 1.0),
            'gamma': Real(0, 5),
        }
        
        # 为每个输出维度训练模型
        for i in range(self.n_outputs):
            print(f"\n{'='*60}")
            print(f"训练输出维度 {i+1}/{self.n_outputs}")
            print(f"{'='*60}")
            
            # 创建基础模型
            base_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                tree_method=tree_method,  # 根据GPU可用性自动选择
                n_jobs=-1
            )
            
            # 贝叶斯优化
            opt = BayesSearchCV(
                base_model,
                search_spaces,
                n_iter=n_iter,
                cv=cv,
                n_jobs=-1,
                scoring='neg_mean_squared_error',
                verbose=1 if verbose else 0,
                random_state=42
            )
            
            # 训练
            opt.fit(X_train, y_train[:, i])
            
            # 保存最优模型和参数
            self.models.append(opt.best_estimator_)
            self.best_params.append(opt.best_params_)
            
            print(f"\n输出维度 {i+1} 最优参数:")
            for param, value in opt.best_params_.items():
                print(f"  {param}: {value}")
            print(f"  最优MSE: {-opt.best_score_:.6f}")
    
    def predict(self, X):
        """
        预测
        
        Args:
            X: 特征
            
        Returns:
            预测结果 (n_samples, n_outputs)
        """
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        return np.column_stack(predictions)
    
    def save(self, filepath='models/xgboost_model.pkl'):
        """保存模型"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'best_params': self.best_params,
                'n_outputs': self.n_outputs
            }, f)
        print(f"\n模型已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath='models/xgboost_model.pkl'):
        """加载模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(n_outputs=data['n_outputs'])
        model.models = data['models']
        model.best_params = data['best_params']
        return model


def plot_predictions(y_true, y_pred, n_outputs, dataset_name, metrics_dict, save_path=None):
    """
    绘制预测结果对比图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        n_outputs: 输出维度数
        dataset_name: 数据集名称（训练集/测试集）
        metrics_dict: 指标字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(n_outputs, 1, figsize=(12, 3*n_outputs))
    if n_outputs == 1:
        axes = [axes]
    
    for i in range(n_outputs):
        ax = axes[i]
        n_samples = len(y_true)
        x = np.arange(1, n_samples + 1)
        
        ax.plot(x, y_true[:, i], 'r-*', label='真实值', linewidth=1, markersize=4)
        ax.plot(x, y_pred[:, i], 'b-o', label='Bayes-XGBoost预测值', linewidth=1, markersize=4)
        
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
    """
    绘制散点图和拟合线
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        n_outputs: 输出维度数
        dataset_name: 数据集名称
        metrics_dict: 指标字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(n_outputs, 1, figsize=(8, 5*n_outputs))
    if n_outputs == 1:
        axes = [axes]
    
    for i in range(n_outputs):
        ax = axes[i]
        metrics = metrics_dict[f'output_{i+1}']
        
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=30)
        
        # 绘制拟合线
        z = np.polyfit(y_true[:, i], y_pred[:, i], 1)
        p = np.poly1d(z)
        x_line = np.linspace(y_true[:, i].min(), y_true[:, i].max(), 100)
        ax.plot(x_line, p(x_line), 'm-', linewidth=2, label=f'拟合线: y={z[0]:.2f}x+{z[1]:.2f}')
        
        # 绘制理想线
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
    print("XGBoost多输出回归模型训练")
    print("="*60)
    
    # 检查GPU支持
    print("\n检测GPU支持...")
    try:
        import numpy as np
        test_X = np.random.rand(10, 5)
        test_y = np.random.rand(10)
        test_model = xgb.XGBRegressor(tree_method='gpu_hist', n_estimators=1)
        test_model.fit(test_X, test_y)
        print("✓ 已启用GPU加速 (tree_method='gpu_hist')")
        tree_method = 'gpu_hist'
    except Exception as e:
        print(f"⚠️  GPU不可用，使用CPU优化算法 (tree_method='hist')")
        print(f"   原因: {str(e)}")
        tree_method = 'hist'
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
    print("\n[2] 开始训练模型...")
    model = XGBoostMultiOutputRegressor(n_outputs=5)
    model.fit(
        data_dict['X_train_norm'], 
        data_dict['y_train_norm'],
        n_iter=30,  # 贝叶斯优化迭代次数
        cv=3,       # 交叉验证折数
        verbose=True,
        tree_method=tree_method  # 使用检测到的tree_method
    )
    
    # 3. 预测
    print("\n[3] 进行预测...")
    y_train_pred_norm = model.predict(data_dict['X_train_norm'])
    y_test_pred_norm = model.predict(data_dict['X_test_norm'])
    
    # 反归一化
    y_train_pred = processor.inverse_transform_output(y_train_pred_norm)
    y_test_pred = processor.inverse_transform_output(y_test_pred_norm)
    
    # 4. 评估
    print("\n[4] 评估模型性能...")
    from metrics import calculate_metrics_multioutput
    
    train_metrics = calculate_metrics_multioutput(data_dict['y_train'], y_train_pred)
    test_metrics = calculate_metrics_multioutput(data_dict['y_test'], y_test_pred)
    
    print_metrics(train_metrics, "训练集评估指标")
    print_metrics(test_metrics, "测试集评估指标")
    
    # 5. 保存模型
    print("\n[5] 保存模型...")
    model.save('models/xgboost_model.pkl')
    
    # 保存预测结果
    import pandas as pd
    train_pred_df = pd.DataFrame(y_train_pred, columns=[f'预测维度_{i+1}' for i in range(5)])
    test_pred_df = pd.DataFrame(y_test_pred, columns=[f'预测维度_{i+1}' for i in range(5)])
    
    train_pred_df.to_excel('results/xgboost_train_predictions.xlsx', index=False)
    test_pred_df.to_excel('results/xgboost_test_predictions.xlsx', index=False)
    print("预测结果已保存到 results/ 目录")
    
    # 6. 可视化
    print("\n[6] 生成可视化图表...")
    import os
    os.makedirs('results/figures', exist_ok=True)
    
    plot_predictions(data_dict['y_train'], y_train_pred, 5, '训练集', 
                    train_metrics, 'results/figures/xgboost_train_prediction.png')
    plot_predictions(data_dict['y_test'], y_test_pred, 5, '测试集', 
                    test_metrics, 'results/figures/xgboost_test_prediction.png')
    
    plot_scatter(data_dict['y_train'], y_train_pred, 5, '训练集',
                train_metrics, 'results/figures/xgboost_train_scatter.png')
    plot_scatter(data_dict['y_test'], y_test_pred, 5, '测试集',
                test_metrics, 'results/figures/xgboost_test_scatter.png')
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)


if __name__ == '__main__':
    main()

