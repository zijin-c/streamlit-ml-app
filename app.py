"""
Streamlit可视化界面
提供交互式的模型训练和预测展示
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import pickle

from data_processor import DataProcessor
from metrics import calculate_metrics_multioutput, calculate_metrics


# 页面配置
st.set_page_config(
    page_title="机器学习模型可视化系统",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# 自定义样式 - 高级专业设计
st.markdown("""
<style>
    /* 导入Google字体 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* 全局样式 */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    
    /* 背景渐变 */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        background-attachment: fixed;
    }
    
    /* 隐藏 Streamlit 默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* 标题样式 - 更现代的渐变效果 */
    .big-font {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1.5rem;
        text-align: center;
        letter-spacing: -1px;
        animation: gradientShift 5s ease infinite;
        text-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .medium-font {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #1a202c;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 4px solid;
        border-image: linear-gradient(90deg, #667eea, #764ba2, #f093fb) 1;
        position: relative;
    }
    
    .medium-font::after {
        content: '';
        position: absolute;
        bottom: -4px;
        left: 0;
        width: 60px;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 2px;
    }
    
    /* 高级卡片样式 - 玻璃态效果 */
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%);
        backdrop-filter: blur(10px);
        padding: 2.5rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4), 
                    0 0 0 1px rgba(255, 255, 255, 0.1) inset;
        color: white;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 10s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 30px 80px rgba(102, 126, 234, 0.5),
                    0 0 0 1px rgba(255, 255, 255, 0.2) inset;
    }
    
    .info-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1),
                    0 0 0 1px rgba(255, 255, 255, 0.5) inset;
        border-left: 5px solid;
        border-image: linear-gradient(180deg, #667eea, #764ba2) 1;
        margin: 1.5rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    
    .info-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 5px;
        background: linear-gradient(180deg, #667eea, #764ba2);
        border-radius: 16px 0 0 16px;
    }
    
    .info-card:hover {
        box-shadow: 0 12px 48px rgba(102, 126, 234, 0.2),
                    0 0 0 1px rgba(255, 255, 255, 0.6) inset;
        transform: translateX(8px) translateY(-2px);
    }
    
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% 200%;
        padding: 2rem 1.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4),
                    0 5px 15px rgba(0, 0, 0, 0.1),
                    0 0 0 1px rgba(255, 255, 255, 0.1) inset;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
        animation: gradientMove 8s ease infinite;
    }
    
    @keyframes gradientMove {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .stat-card::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #667eea);
        background-size: 400% 400%;
        border-radius: 20px;
        z-index: -1;
        opacity: 0;
        transition: opacity 0.3s;
        animation: borderRotate 3s linear infinite;
    }
    
    @keyframes borderRotate {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stat-card:hover::before {
        opacity: 1;
    }
    
    .stat-card:hover {
        transform: scale(1.08) translateY(-5px);
        box-shadow: 0 25px 50px rgba(102, 126, 234, 0.5),
                    0 10px 25px rgba(0, 0, 0, 0.15);
    }
    
    .stat-card h3 {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.75rem 0;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        letter-spacing: -1px;
    }
    
    .stat-card p {
        font-size: 1rem;
        opacity: 0.95;
        margin: 0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* 侧边栏样式 - 毛玻璃效果 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, 
            rgba(102, 126, 234, 0.95) 0%, 
            rgba(118, 75, 162, 0.95) 100%);
        backdrop-filter: blur(20px);
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        color: white;
        font-weight: 500;
    }
    
    [data-testid="stSidebar"] .stSelectbox label {
        color: white;
        font-weight: 500;
    }
    
    /* 按钮样式 - 3D效果 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 2.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4),
                    0 4px 8px rgba(0, 0, 0, 0.1),
                    inset 0 1px 0 rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 24px rgba(102, 126, 234, 0.5),
                    0 6px 12px rgba(0, 0, 0, 0.15),
                    inset 0 1px 0 rgba(255, 255, 255, 0.3);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4);
    }
    
    /* 输入框样式 - 现代设计 */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        background: rgba(255, 255, 255, 0.9);
        transition: all 0.3s ease;
        padding: 0.75rem 1rem;
        font-size: 0.95rem;
    }
    
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1),
                    0 4px 12px rgba(102, 126, 234, 0.2);
        background: white;
        outline: none;
    }
    
    /* 选择框样式 */
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        background: rgba(255, 255, 255, 0.9);
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
    }
    
    /* 标签页样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.5);
        padding: 0.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 1rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* 代码块样式 */
    .stCodeBlock {
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        background: rgba(30, 30, 30, 0.95);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* 数据框样式 */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    /* 分隔线样式 */
    hr {
        border: none;
        height: 3px;
        background: linear-gradient(90deg, 
            transparent 0%, 
            #667eea 20%, 
            #764ba2 50%, 
            #f093fb 80%, 
            transparent 100%);
        margin: 3rem 0;
        border-radius: 2px;
    }
    
    /* 警告框美化 */
    .stAlert {
        border-radius: 12px;
        border-left: 5px solid;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* 成功消息样式 */
    .stSuccess {
        background: linear-gradient(135deg, 
            rgba(132, 250, 176, 0.95) 0%, 
            rgba(143, 211, 244, 0.95) 100%);
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 8px 20px rgba(132, 250, 176, 0.3);
    }
    
    /* 警告消息样式 */
    .stWarning {
        background: linear-gradient(135deg, 
            rgba(255, 236, 210, 0.95) 0%, 
            rgba(252, 182, 159, 0.95) 100%);
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 8px 20px rgba(252, 182, 159, 0.3);
    }
    
    /* 信息消息样式 */
    .stInfo {
        background: linear-gradient(135deg, 
            rgba(168, 237, 234, 0.95) 0%, 
            rgba(254, 214, 227, 0.95) 100%);
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 8px 20px rgba(168, 237, 234, 0.3);
    }
    
    /* 图表容器 */
    .js-plotly-plot {
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12),
                    0 0 0 1px rgba(102, 126, 234, 0.1) inset;
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .js-plotly-plot:hover {
        box-shadow: 0 12px 32px rgba(102, 126, 234, 0.2),
                    0 0 0 1px rgba(102, 126, 234, 0.2) inset;
        transform: translateY(-2px);
    }
    
    /* 加载动画 */
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
            transform: scale(1);
        }
        50% {
            opacity: 0.7;
            transform: scale(1.05);
        }
    }
    
    .spinner {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    /* Radio按钮样式 */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 12px;
        border: 2px solid rgba(102, 126, 234, 0.2);
    }
    
    /* 下载按钮样式 */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5);
    }
    
    /* 响应式设计 */
    @media (max-width: 768px) {
        .big-font {
            font-size: 2.5rem !important;
        }
        .medium-font {
            font-size: 1.5rem !important;
        }
        .stat-card h3 {
            font-size: 2rem !important;
        }
    }
    
    /* 滚动条美化 */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
</style>
""", unsafe_allow_html=True)


def load_data():
    """加载数据"""
    try:
        processor = DataProcessor(data_path='去噪后数据.xlsx', train_ratio=0.7, n_outputs=5)
        data_dict = processor.prepare_data()
        return processor, data_dict
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        return None, None


def load_model(model_type):
    """加载训练好的模型"""
    model_paths = {
        'XGBoost': 'models/xgboost_model.pkl',
        'LSTM': 'models/lstm_model.pkl',
        'Transformer': 'models/transformer_model.pkl'
    }
    
    try:
        if model_type == 'XGBoost':
            from train_xgboost import XGBoostMultiOutputRegressor
            return XGBoostMultiOutputRegressor.load(model_paths[model_type])
        elif model_type == 'LSTM':
            from train_lstm import LSTMTrainer
            return LSTMTrainer.load(model_paths[model_type])
        elif model_type == 'Transformer':
            from train_transformer import TransformerTrainer
            return TransformerTrainer.load(model_paths[model_type])
    except Exception as e:
        st.warning(f"模型加载失败: {str(e)}")
        return None


def plot_predictions_interactive(y_true, y_pred, output_idx, dataset_name):
    """使用Plotly绘制交互式预测结果对比图"""
    n_samples = len(y_true)
    x = np.arange(1, n_samples + 1)
    
    # 专业配色方案
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
    true_color = '#e74c3c'
    pred_color = colors[output_idx % len(colors)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x, y=y_true[:, output_idx],
        mode='lines+markers',
        name='真实值',
        line=dict(color=true_color, width=2.5),
        marker=dict(size=5, symbol='circle', opacity=0.7),
        hovertemplate='<b>真实值</b><br>样本: %{x}<br>值: %{y:.4f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=x, y=y_pred[:, output_idx],
        mode='lines+markers',
        name='预测值',
        line=dict(color=pred_color, width=2.5, dash='dash'),
        marker=dict(size=5, symbol='diamond', opacity=0.7),
        hovertemplate='<b>预测值</b><br>样本: %{x}<br>值: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f'{dataset_name} - 输出维度 {output_idx+1}',
            font=dict(size=18, color='#2d3748'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title=dict(text='样本序号', font=dict(size=14, color='#4a5568')),
        yaxis_title=dict(text='值', font=dict(size=14, color='#4a5568')),
        hovermode='x unified',
        height=450,
        template='plotly_white',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial, sans-serif'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        margin=dict(l=60, r=20, t=80, b=60)
    )
    
    return fig


def plot_scatter_interactive(y_true, y_pred, output_idx, dataset_name, metrics):
    """使用Plotly绘制交互式散点图"""
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
    scatter_color = colors[output_idx % len(colors)]
    
    fig = go.Figure()
    
    # 散点图
    fig.add_trace(go.Scatter(
        x=y_true[:, output_idx],
        y=y_pred[:, output_idx],
        mode='markers',
        name='预测点',
        marker=dict(
            size=10, 
            opacity=0.6, 
            color=scatter_color,
            line=dict(width=1, color='white')
        ),
        hovertemplate='<b>预测点</b><br>真实值: %{x:.4f}<br>预测值: %{y:.4f}<extra></extra>'
    ))
    
    # 理想线
    min_val = min(y_true[:, output_idx].min(), y_pred[:, output_idx].min())
    max_val = max(y_true[:, output_idx].max(), y_pred[:, output_idx].max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='理想线 (y=x)',
        line=dict(color='#2d3748', width=2.5, dash='dot'),
        hovertemplate='理想线<extra></extra>'
    ))
    
    # 拟合线
    z = np.polyfit(y_true[:, output_idx], y_pred[:, output_idx], 1)
    p = np.poly1d(z)
    x_line = np.linspace(min_val, max_val, 100)
    
    fig.add_trace(go.Scatter(
        x=x_line,
        y=p(x_line),
        mode='lines',
        name=f'拟合线: y={z[0]:.4f}x+{z[1]:.4f}',
        line=dict(color='#e74c3c', width=2.5),
        hovertemplate='拟合线<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f'{dataset_name} - 输出维度 {output_idx+1}<br><span style="font-size:0.8em">R² = {metrics["R2"]:.4f} | RMSE = {metrics["RMSE"]:.4f}</span>',
            font=dict(size=18, color='#2d3748'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title=dict(text='真实值', font=dict(size=14, color='#4a5568')),
        yaxis_title=dict(text='预测值', font=dict(size=14, color='#4a5568')),
        height=550,
        template='plotly_white',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial, sans-serif'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        margin=dict(l=60, r=20, t=100, b=60)
    )
    
    return fig


def plot_error_histogram(y_true, y_pred, output_idx):
    """绘制误差直方图"""
    errors = y_true[:, output_idx] - y_pred[:, output_idx]
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
    bar_color = colors[output_idx % len(colors)]
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=30,
        name='误差分布',
        marker_color=bar_color,
        marker_line_color='white',
        marker_line_width=1,
        opacity=0.8,
        hovertemplate='<b>误差分布</b><br>误差范围: %{x:.4f}<br>频数: %{y}<extra></extra>'
    ))
    
    # 添加均值线
    mean_error = np.mean(errors)
    fig.add_vline(
        x=mean_error,
        line_dash="dash",
        line_color="red",
        annotation_text=f"均值: {mean_error:.4f}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title=dict(
            text=f'误差分布直方图 - 输出维度 {output_idx+1}',
            font=dict(size=18, color='#2d3748'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title=dict(text='误差', font=dict(size=14, color='#4a5568')),
        yaxis_title=dict(text='频数', font=dict(size=14, color='#4a5568')),
        height=450,
        template='plotly_white',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial, sans-serif'),
        margin=dict(l=60, r=20, t=80, b=60),
        showlegend=False
    )
    
    return fig


def plot_metrics_comparison(metrics_dict):
    """绘制各输出维度的指标对比"""
    output_names = [f'输出{i+1}' for i in range(5)]
    metric_names = ['RMSE', 'R²', 'MAE', 'MAPE', 'MSE']
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=metric_names,
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
    
    for idx, metric in enumerate(metric_names):
        values = [metrics_dict[f'output_{i+1}'][metric] for i in range(5)]
        row, col = positions[idx]
        
        # 为每个输出维度使用不同颜色
        fig.add_trace(
            go.Bar(
                x=output_names, 
                y=values, 
                name=metric, 
                showlegend=False,
                marker=dict(
                    color=colors,
                    line=dict(color='white', width=1)
                ),
                hovertemplate=f'<b>{metric}</b><br>%{{x}}<br>值: %{{y:.4f}}<extra></extra>'
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=650, 
        showlegend=False, 
        title=dict(
            text="各输出维度评估指标对比",
            font=dict(size=20, color='#2d3748'),
            x=0.5,
            xanchor='center'
        ),
        template='plotly_white',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial, sans-serif')
    )
    
    # 更新所有子图的样式
    for i in range(1, 3):
        for j in range(1, 4):
            fig.update_xaxes(
                showgrid=True, 
                gridwidth=1, 
                gridcolor='rgba(0,0,0,0.1)',
                row=i, col=j
            )
            fig.update_yaxes(
                showgrid=True, 
                gridwidth=1, 
                gridcolor='rgba(0,0,0,0.1)',
                row=i, col=j
            )
    
    return fig


def main():
    # 标题
    st.markdown('<p class="big-font">机器学习模型可视化系统</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 侧边栏
    st.sidebar.markdown('<p class="medium-font">控制面板</p>', unsafe_allow_html=True)
    
    # 页面选择
    page = st.sidebar.radio(
        "选择功能",
        ["首页", "数据探索", "模型训练", "模型评估", "模型预测"]
    )
    
    # 加载数据
    if 'processor' not in st.session_state or 'data_dict' not in st.session_state:
        with st.spinner('正在加载数据...'):
            processor, data_dict = load_data()
            if processor and data_dict:
                st.session_state.processor = processor
                st.session_state.data_dict = data_dict
    
    processor = st.session_state.get('processor')
    data_dict = st.session_state.get('data_dict')
    
    if processor is None or data_dict is None:
        st.error("请确保数据文件 '去噪后数据.xlsx' 存在！")
        return
    
    # =============== 首页 ===============
    if page == "首页":
        # 欢迎横幅
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0; margin-bottom: 3rem;">
            <p style="font-size: 1.2rem; color: #718096; margin-bottom: 1rem;">
                专业的机器学习模型训练与可视化平台
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 统计数据卡片 - 使用渐变背景
        col1, col2, col3 = st.columns(3)
        
        gradients = [
            "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
            "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)"
        ]
        
        stats = [
            ("训练样本数", data_dict['n_train']),
            ("测试样本数", data_dict['n_test']),
            ("特征维度", data_dict['n_features'])
        ]
        
        for idx, (col, (label, value)) in enumerate(zip([col1, col2, col3], stats)):
            with col:
                st.markdown(f"""
                <div class="stat-card" style="background: {gradients[idx]};">
                    <p>{label}</p>
                    <h3>{value:,}</h3>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # 系统介绍卡片 - 改进设计
        st.markdown("### 系统介绍")
        st.markdown("""
        <div class="info-card">
            <h4 style="color: #667eea; margin-top: 0; font-size: 1.5rem; margin-bottom: 1.5rem;">
                强大的机器学习模型训练与可视化系统
            </h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem;">
                <div style="padding: 1rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 10px; border-left: 4px solid #667eea;">
                    <h5 style="color: #667eea; margin: 0 0 0.5rem 0; font-weight: 600;">三种模型</h5>
                    <p style="margin: 0; color: #4a5568;">XGBoost, LSTM, Transformer</p>
                </div>
                <div style="padding: 1rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 10px; border-left: 4px solid #764ba2;">
                    <h5 style="color: #764ba2; margin: 0 0 0.5rem 0; font-weight: 600;">自动优化</h5>
                    <p style="margin: 0; color: #4a5568;">贝叶斯超参数优化</p>
                </div>
                <div style="padding: 1rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 10px; border-left: 4px solid #f093fb;">
                    <h5 style="color: #f093fb; margin: 0 0 0.5rem 0; font-weight: 600;">多输出回归</h5>
                    <p style="margin: 0; color: #4a5568;">同时预测5个输出维度</p>
                </div>
                <div style="padding: 1rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 10px; border-left: 4px solid #4facfe;">
                    <h5 style="color: #4facfe; margin: 0 0 0.5rem 0; font-weight: 600;">可视化分析</h5>
                    <p style="margin: 0; color: #4a5568;">交互式图表展示</p>
                </div>
                <div style="padding: 1rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 10px; border-left: 4px solid #f5576c;">
                    <h5 style="color: #f5576c; margin: 0 0 0.5rem 0; font-weight: 600;">评估指标</h5>
                    <p style="margin: 0; color: #4a5568;">RMSE, R², MAE, MBE, MAPE, MSE</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 快速开始 - 改进卡片设计
        st.markdown("### 快速开始")
        col1, col2 = st.columns(2)
        
        features = [
            ("数据探索", "查看数据的基本信息、统计特征和分布情况", "linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)", "#667eea"),
            ("模型训练", "选择模型类型并训练（需要在命令行运行训练脚本）", "linear-gradient(135deg, rgba(240, 147, 251, 0.1) 0%, rgba(245, 87, 108, 0.1) 100%)", "#f093fb"),
            ("模型评估", "查看训练好的模型性能指标和可视化结果", "linear-gradient(135deg, rgba(79, 172, 254, 0.1) 0%, rgba(0, 242, 254, 0.1) 100%)", "#4facfe"),
            ("模型预测", "使用训练好的模型进行实时预测", "linear-gradient(135deg, rgba(245, 87, 108, 0.1) 0%, rgba(240, 147, 251, 0.1) 100%)", "#f5576c")
        ]
        
        with col1:
            for title, desc, bg, border_color in features[:2]:
                st.markdown(f"""
                <div class="info-card" style="background: {bg}; border-left-color: {border_color};">
                    <h4 style="color: {border_color}; margin-top: 0; font-size: 1.3rem; margin-bottom: 0.75rem; font-weight: 600;">
                        {title}
                    </h4>
                    <p style="margin: 0; color: #4a5568; line-height: 1.6;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            for title, desc, bg, border_color in features[2:]:
                st.markdown(f"""
                <div class="info-card" style="background: {bg}; border-left-color: {border_color};">
                    <h4 style="color: {border_color}; margin-top: 0; font-size: 1.3rem; margin-bottom: 0.75rem; font-weight: 600;">
                        {title}
                    </h4>
                    <p style="margin: 0; color: #4a5568; line-height: 1.6;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 训练模型 - 改进代码块展示
        st.markdown("### 训练模型")
        st.markdown("""
        <div class="info-card">
            <h4 style="color: #667eea; margin-top: 0; margin-bottom: 1rem; font-weight: 600;">
                在命令行中运行以下命令来训练模型
            </h4>
            <p style="color: #718096; margin-bottom: 1.5rem;">
                选择您想要训练的模型类型，然后执行相应的命令：
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 使用列布局展示命令
        col1, col2, col3 = st.columns(3)
        
        commands = [
            ("XGBoost", "python train_xgboost.py", "#667eea"),
            ("LSTM", "python train_lstm.py", "#f093fb"),
            ("Transformer", "python train_transformer.py", "#4facfe")
        ]
        
        for col, (model_name, cmd, color) in zip([col1, col2, col3], commands):
            with col:
                st.markdown(f"""
                <div style="background: rgba(255, 255, 255, 0.95); padding: 1.5rem; border-radius: 12px; 
                            border: 2px solid {color}; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); margin-bottom: 1rem;">
                    <h5 style="color: {color}; margin: 0 0 1rem 0; font-size: 1.1rem; font-weight: 600;">{model_name}</h5>
                    <code style="background: #f7fafc; padding: 0.5rem 1rem; border-radius: 6px; 
                                display: block; color: #2d3748; font-family: 'Courier New', monospace;">
                        {cmd}
                    </code>
                </div>
                """, unsafe_allow_html=True)
        
        # 添加提示信息
        st.markdown("""
        <div class="info-card" style="background: linear-gradient(135deg, rgba(132, 250, 176, 0.1) 0%, rgba(143, 211, 244, 0.1) 100%); 
                    border-left-color: #10b981; margin-top: 2rem;">
            <p style="margin: 0; color: #065f46;">
                <strong>提示：</strong>训练完成后，模型将保存在 <code style="background: rgba(16, 185, 129, 0.1); 
                padding: 0.2rem 0.5rem; border-radius: 4px;">models/</code> 目录下，可以在「模型评估」页面查看结果
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # =============== 数据探索 ===============
    elif page == "数据探索":
        st.markdown('<p class="medium-font">数据探索</p>', unsafe_allow_html=True)
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["数据统计", "数据可视化", "数据详情"])
        
        with tab1:
            st.markdown("#### 训练集统计信息")
            train_df = pd.DataFrame(data_dict['y_train'], columns=[f'输出{i+1}' for i in range(5)])
            st.dataframe(train_df.describe().style.background_gradient(cmap='Blues', axis=0), use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 测试集统计信息")
            test_df = pd.DataFrame(data_dict['y_test'], columns=[f'输出{i+1}' for i in range(5)])
            st.dataframe(test_df.describe().style.background_gradient(cmap='Reds', axis=0), use_container_width=True)
        
        with tab2:
            st.markdown("#### 输出维度分布")
            output_idx = st.selectbox("选择输出维度", range(5), format_func=lambda x: f"输出 {x+1}")
            
            col1, col2 = st.columns(2)
            colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
            bar_color = colors[output_idx % len(colors)]
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=data_dict['y_train'][:, output_idx],
                    name='训练集',
                    opacity=0.8,
                    marker_color=bar_color,
                    marker_line_color='white',
                    marker_line_width=1,
                    nbinsx=30
                ))
                fig.update_layout(
                    title=dict(text=f'训练集 - 输出 {output_idx+1} 分布', font=dict(size=16, color='#2d3748')),
                    xaxis_title='值',
                    yaxis_title='频数',
                    height=400,
                    template='plotly_white',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=data_dict['y_test'][:, output_idx],
                    name='测试集',
                    opacity=0.8,
                    marker_color='#e74c3c',
                    marker_line_color='white',
                    marker_line_width=1,
                    nbinsx=30
                ))
                fig.update_layout(
                    title=dict(text=f'测试集 - 输出 {output_idx+1} 分布', font=dict(size=16, color='#2d3748')),
                    xaxis_title='值',
                    yaxis_title='频数',
                    height=400,
                    template='plotly_white',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 相关性矩阵
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 输出维度相关性矩阵")
            corr_matrix = train_df.corr()
            fig = px.imshow(
                corr_matrix, 
                text_auto=True, 
                aspect="auto", 
                color_continuous_scale='RdYlBu_r',
                labels=dict(color="相关系数")
            )
            fig.update_layout(
                title=dict(text='相关性热图', font=dict(size=18, color='#2d3748')),
                height=500,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("#### 原始数据预览")
            
            data_choice = st.radio("选择数据集", ["训练集", "测试集"], horizontal=True)
            
            if data_choice == "训练集":
                display_df = pd.DataFrame(
                    np.hstack([data_dict['X_train'], data_dict['y_train']]),
                    columns=[f'特征{i+1}' for i in range(data_dict['n_features'])] + 
                            [f'输出{i+1}' for i in range(5)]
                )
            else:
                display_df = pd.DataFrame(
                    np.hstack([data_dict['X_test'], data_dict['y_test']]),
                    columns=[f'特征{i+1}' for i in range(data_dict['n_features'])] + 
                            [f'输出{i+1}' for i in range(5)]
                )
            
            st.dataframe(
                display_df.head(100).style.background_gradient(cmap='viridis', axis=0, subset=display_df.columns[-5:]),
                height=400,
                use_container_width=True
            )
            
            st.markdown(f"<p style='color: #718096; font-size: 0.9em;'>显示前100行，共 {len(display_df)} 行数据</p>", unsafe_allow_html=True)
            
            # 下载按钮
            csv = display_df.to_csv(index=False, encoding='utf-8-sig')
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                st.download_button(
                    label="下载数据",
                    data=csv,
                    file_name=f"{data_choice}_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    # =============== 模型训练 ===============
    elif page == "模型训练":
        st.markdown('<p class="medium-font">模型训练</p>', unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("""
        <div class="info-card">
            <p style="margin: 0;"><strong>提示：</strong>由于训练过程可能需要较长时间，请在命令行中运行训练脚本</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        model_choice = st.selectbox("选择模型", ["XGBoost", "LSTM", "Transformer"])
        
        st.markdown(f"### {model_choice} 模型训练")
        
        if model_choice == "XGBoost":
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                <div class="info-card">
                    <h4 style="color: #667eea; margin-top: 0;">XGBoost模型特点</h4>
                    <ul>
                        <li>基于梯度提升树</li>
                        <li>适合处理表格数据</li>
                        <li>训练速度快</li>
                        <li>可解释性强</li>
                    </ul>
                    <h4 style="color: #667eea;">超参数</h4>
                    <ul>
                        <li><strong>n_estimators</strong>: 树的数量 (10-5000)</li>
                        <li><strong>max_depth</strong>: 树的最大深度 (10-20)</li>
                        <li><strong>learning_rate</strong>: 学习率 (0.0001-1)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.code("python train_xgboost.py", language="bash")
            
        elif model_choice == "LSTM":
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                <div class="info-card">
                    <h4 style="color: #667eea; margin-top: 0;">LSTM模型特点</h4>
                    <ul>
                        <li>长短期记忆网络</li>
                        <li>适合处理序列数据</li>
                        <li>能捕捉时间依赖关系</li>
                    </ul>
                    <h4 style="color: #667eea;">超参数</h4>
                    <ul>
                        <li><strong>hidden_size</strong>: 隐藏层单元数 (20-50)</li>
                        <li><strong>learning_rate</strong>: 学习率 (1e-3 - 1e-2)</li>
                        <li><strong>l2_regularization</strong>: L2正则化 (1e-4 - 1e-3)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.code("python train_lstm.py", language="bash")
            
        elif model_choice == "Transformer":
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                <div class="info-card">
                    <h4 style="color: #667eea; margin-top: 0;">Transformer模型特点</h4>
                    <ul>
                        <li>自注意力机制</li>
                        <li>能并行处理序列</li>
                        <li>捕捉长距离依赖</li>
                    </ul>
                    <h4 style="color: #667eea;">超参数</h4>
                    <ul>
                        <li><strong>nhead</strong>: 注意力头数 (2-8)</li>
                        <li><strong>learning_rate</strong>: 学习率 (1e-4 - 1e-1)</li>
                        <li><strong>l2_regularization</strong>: L2正则化 (1e-6 - 1e-2)</li>
                        <li><strong>batch_size</strong>: 批大小 (16-128)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.code("python train_transformer.py", language="bash")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <p style="margin: 0;"><strong>提示：</strong>训练完成后，模型将保存在 <code>models/</code> 目录下，可以在「模型评估」页面查看结果</p>
        </div>
        """, unsafe_allow_html=True)
    
    # =============== 模型评估 ===============
    elif page == "模型评估":
        st.markdown('<p class="medium-font">模型评估</p>', unsafe_allow_html=True)
        st.markdown("---")
        
        model_choice = st.sidebar.selectbox("选择模型", ["XGBoost", "LSTM", "Transformer"])
        
        # 加载模型
        model = load_model(model_choice)
        
        if model is None:
            st.markdown(f"""
            <div class="info-card">
                <p style="margin: 0;"><strong>警告：</strong>{model_choice} 模型尚未训练，请先运行训练脚本</p>
            </div>
            """, unsafe_allow_html=True)
            st.code(f"python train_{model_choice.lower()}.py", language="bash")
            return
        
        st.markdown(f"""
        <div class="info-card" style="background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); color: white;">
            <p style="margin: 0; font-size: 1.1em;"><strong>{model_choice} 模型加载成功</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # 进行预测
        with st.spinner('正在进行预测...'):
            y_train_pred_norm = model.predict(data_dict['X_train_norm'])
            y_test_pred_norm = model.predict(data_dict['X_test_norm'])
            
            y_train_pred = processor.inverse_transform_output(y_train_pred_norm)
            y_test_pred = processor.inverse_transform_output(y_test_pred_norm)
        
        # 计算指标
        train_metrics = calculate_metrics_multioutput(data_dict['y_train'], y_train_pred)
        test_metrics = calculate_metrics_multioutput(data_dict['y_test'], y_test_pred)
        
        # 显示整体指标
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 整体性能指标")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 训练集")
            metrics_df_train = pd.DataFrame({
                '指标': ['RMSE', 'R²', 'MAE', 'MBE', 'MAPE', 'MSE'],
                '值': [
                    f"{train_metrics['average']['RMSE']:.6f}",
                    f"{train_metrics['average']['R2']:.6f}",
                    f"{train_metrics['average']['MAE']:.6f}",
                    f"{train_metrics['average']['MBE']:.6f}",
                    f"{train_metrics['average']['MAPE']:.2f}%",
                    f"{train_metrics['average']['MSE']:.6f}"
                ]
            })
            st.dataframe(
                metrics_df_train.style.background_gradient(cmap='Blues', axis=0, subset=['值']),
                hide_index=True,
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### 测试集")
            metrics_df_test = pd.DataFrame({
                '指标': ['RMSE', 'R²', 'MAE', 'MBE', 'MAPE', 'MSE'],
                '值': [
                    f"{test_metrics['average']['RMSE']:.6f}",
                    f"{test_metrics['average']['R2']:.6f}",
                    f"{test_metrics['average']['MAE']:.6f}",
                    f"{test_metrics['average']['MBE']:.6f}",
                    f"{test_metrics['average']['MAPE']:.2f}%",
                    f"{test_metrics['average']['MSE']:.6f}"
                ]
            })
            st.dataframe(
                metrics_df_test.style.background_gradient(cmap='Reds', axis=0, subset=['值']),
                hide_index=True,
                use_container_width=True
            )
        
        # 各输出维度指标对比
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 各输出维度指标对比")
        
        dataset_choice = st.radio("选择数据集", ["训练集", "测试集"], horizontal=True)
        metrics_to_show = train_metrics if dataset_choice == "训练集" else test_metrics
        
        fig = plot_metrics_comparison(metrics_to_show)
        st.plotly_chart(fig, use_container_width=True)
        
        # 详细评估
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 详细评估")
        
        output_idx = st.selectbox("选择输出维度", range(5), format_func=lambda x: f"输出 {x+1}")
        
        tab1, tab2, tab3 = st.tabs(["预测对比", "散点图", "误差分析"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = plot_predictions_interactive(
                    data_dict['y_train'], y_train_pred, output_idx, "训练集"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = plot_predictions_interactive(
                    data_dict['y_test'], y_test_pred, output_idx, "测试集"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = plot_scatter_interactive(
                    data_dict['y_train'], y_train_pred, output_idx, "训练集",
                    train_metrics[f'output_{output_idx+1}']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = plot_scatter_interactive(
                    data_dict['y_test'], y_test_pred, output_idx, "测试集",
                    test_metrics[f'output_{output_idx+1}']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = plot_error_histogram(data_dict['y_train'], y_train_pred, output_idx)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = plot_error_histogram(data_dict['y_test'], y_test_pred, output_idx)
                st.plotly_chart(fig, use_container_width=True)
        
        # 下载预测结果
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 下载预测结果")
        
        col1, col2 = st.columns(2)
        
        with col1:
            train_pred_df = pd.DataFrame(y_train_pred, columns=[f'预测维度_{i+1}' for i in range(5)])
            csv = train_pred_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="下载训练集预测结果",
                data=csv,
                file_name=f"{model_choice.lower()}_train_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            test_pred_df = pd.DataFrame(y_test_pred, columns=[f'预测维度_{i+1}' for i in range(5)])
            csv = test_pred_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="下载测试集预测结果",
                data=csv,
                file_name=f"{model_choice.lower()}_test_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # =============== 模型预测 ===============
    elif page == "模型预测":
        st.markdown('<p class="medium-font">模型预测</p>', unsafe_allow_html=True)
        st.markdown("---")
        
        model_choice = st.sidebar.selectbox("选择模型", ["XGBoost", "LSTM", "Transformer"])
        
        # 加载模型
        model = load_model(model_choice)
        
        if model is None:
            st.markdown(f"""
            <div class="info-card">
                <p style="margin: 0;"><strong>警告：</strong>{model_choice} 模型尚未训练，请先运行训练脚本</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        st.markdown(f"""
        <div class="info-card" style="background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); color: white;">
            <p style="margin: 0; font-size: 1.1em;"><strong>{model_choice} 模型加载成功</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 输入特征")
        
        # 方式1: 手动输入
        st.markdown("#### 方式1: 手动输入特征值")
        st.markdown("""
        <div class="info-card">
            <p style="margin: 0;">请在下方输入框中填入特征值，然后点击"开始预测"按钮</p>
        </div>
        """, unsafe_allow_html=True)
        
        n_features = data_dict['n_features']
        
        # 使用列布局
        cols_per_row = 4
        input_features = []
        
        for i in range(0, n_features, cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < n_features:
                    with cols[j]:
                        val = st.number_input(
                            f"特征 {i+j+1}",
                            value=0.0,
                            format="%.4f",
                            key=f"feature_{i+j}"
                        )
                        input_features.append(val)
        
        if st.button("开始预测", type="primary"):
            # 归一化输入
            X_input = np.array(input_features).reshape(1, -1)
            X_input_norm = processor.input_scaler.transform(X_input)
            
            # 预测
            with st.spinner('正在预测...'):
                y_pred_norm = model.predict(X_input_norm)
                y_pred = processor.inverse_transform_output(y_pred_norm)
            
            # 显示结果
            st.markdown("""
            <div class="info-card" style="background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); color: white;">
                <p style="margin: 0; font-size: 1.1em;"><strong>预测完成</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 预测结果")
            
            result_df = pd.DataFrame({
                '输出维度': [f'输出 {i+1}' for i in range(5)],
                '预测值': y_pred.flatten()
            })
            
            st.dataframe(
                result_df.style.background_gradient(cmap='viridis', axis=0, subset=['预测值']),
                hide_index=True,
                use_container_width=True
            )
            
            # 可视化
            colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
            fig = go.Figure(data=[
                go.Bar(
                    x=result_df['输出维度'], 
                    y=result_df['预测值'], 
                    marker_color=colors,
                    marker_line_color='white',
                    marker_line_width=2,
                    text=result_df['预测值'],
                    texttemplate='%{text:.4f}',
                    textposition='outside'
                )
            ])
            fig.update_layout(
                title=dict(text='预测结果可视化', font=dict(size=18, color='#2d3748')),
                xaxis_title=dict(text='输出维度', font=dict(size=14, color='#4a5568')),
                yaxis_title=dict(text='预测值', font=dict(size=14, color='#4a5568')),
                height=450,
                template='plotly_white',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 方式2: 使用测试集样本
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("#### 方式2: 从测试集选择样本")
        st.markdown("""
        <div class="info-card">
            <p style="margin: 0;">从测试集中选择一个样本进行预测，并对比真实值与预测值</p>
        </div>
        """, unsafe_allow_html=True)
        
        sample_idx = st.selectbox(
            "选择测试集样本",
            range(data_dict['n_test']),
            format_func=lambda x: f"样本 {x+1}"
        )
        
        if st.button("预测选定样本"):
            X_sample = data_dict['X_test_norm'][sample_idx:sample_idx+1]
            y_true_sample = data_dict['y_test'][sample_idx]
            
            # 预测
            with st.spinner('正在预测...'):
                y_pred_norm = model.predict(X_sample)
                y_pred_sample = processor.inverse_transform_output(y_pred_norm).flatten()
            
            # 对比结果
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 预测对比")
            
            comparison_df = pd.DataFrame({
                '输出维度': [f'输出 {i+1}' for i in range(5)],
                '真实值': y_true_sample,
                '预测值': y_pred_sample,
                '误差': y_true_sample - y_pred_sample,
                '相对误差(%)': np.abs((y_true_sample - y_pred_sample) / y_true_sample * 100)
            })
            
            st.dataframe(
                comparison_df.style.background_gradient(cmap='RdYlGn', axis=0, subset=['误差', '相对误差(%)']),
                hide_index=True,
                use_container_width=True
            )
            
            # 可视化对比
            colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='真实值',
                x=comparison_df['输出维度'],
                y=comparison_df['真实值'],
                marker_color='#e74c3c',
                marker_line_color='white',
                marker_line_width=2,
                text=comparison_df['真实值'],
                texttemplate='%{text:.4f}',
                textposition='outside'
            ))
            fig.add_trace(go.Bar(
                name='预测值',
                x=comparison_df['输出维度'],
                y=comparison_df['预测值'],
                marker_color=colors,
                marker_line_color='white',
                marker_line_width=2,
                text=comparison_df['预测值'],
                texttemplate='%{text:.4f}',
                textposition='outside'
            ))
            
            fig.update_layout(
                title=dict(text='真实值 vs 预测值', font=dict(size=18, color='#2d3748')),
                xaxis_title=dict(text='输出维度', font=dict(size=14, color='#4a5568')),
                yaxis_title=dict(text='值', font=dict(size=14, color='#4a5568')),
                barmode='group',
                height=450,
                template='plotly_white',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()

