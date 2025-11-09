"""
评估指标模块
包含各种回归模型的评估指标
"""
import numpy as np
from typing import Dict


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算所有评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        包含所有指标的字典
    """
    # 确保是一维数组
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    n = len(y_true)
    
    # 均方根误差 (RMSE)
    rmse = np.sqrt(np.sum((y_true - y_pred) ** 2) / n)
    
    # 决定系数 (R²)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # 平均绝对误差 (MAE)
    mae = np.sum(np.abs(y_true - y_pred)) / n
    
    # 平均偏差误差 (MBE)
    mbe = np.sum(y_pred - y_true) / n
    
    # 平均绝对百分比误差 (MAPE)
    # 避免除零错误
    mask = y_true != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0
    
    # 均方误差 (MSE)
    mse = np.mean((y_true - y_pred) ** 2)
    
    return {
        'RMSE': rmse,
        'R2': r2,
        'MAE': mae,
        'MBE': mbe,
        'MAPE': mape,
        'MSE': mse
    }


def calculate_metrics_multioutput(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    计算多输出模型的评估指标
    
    Args:
        y_true: 真实值 (n_samples, n_outputs)
        y_pred: 预测值 (n_samples, n_outputs)
        
    Returns:
        每个输出维度的指标字典
    """
    n_outputs = y_true.shape[1]
    results = {}
    
    for i in range(n_outputs):
        results[f'output_{i+1}'] = calculate_metrics(y_true[:, i], y_pred[:, i])
    
    # 计算平均指标
    avg_metrics = {}
    metric_names = ['RMSE', 'R2', 'MAE', 'MBE', 'MAPE', 'MSE']
    for metric in metric_names:
        values = [results[f'output_{i+1}'][metric] for i in range(n_outputs)]
        avg_metrics[metric] = np.mean(values)
    
    results['average'] = avg_metrics
    
    return results


def print_metrics(metrics: Dict, title: str = "评估指标"):
    """
    打印评估指标
    
    Args:
        metrics: 指标字典
        title: 标题
    """
    print("\n" + "="*50)
    print(f"{title:^50}")
    print("="*50)
    
    if 'output_1' in metrics:
        # 多输出模型
        for output_name, output_metrics in metrics.items():
            if output_name == 'average':
                print(f"\n{'平均指标':^50}")
                print("-"*50)
            else:
                print(f"\n{output_name:^50}")
                print("-"*50)
            
            for metric_name, value in output_metrics.items():
                print(f"{metric_name:.<30}{value:.6f}")
    else:
        # 单输出模型
        for metric_name, value in metrics.items():
            print(f"{metric_name:.<30}{value:.6f}")
    
    print("="*50 + "\n")

