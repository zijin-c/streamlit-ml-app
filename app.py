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
    
    /* 隐藏侧边栏 */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* 调整主内容区域宽度 */
    .main .block-container {
        max-width: 100% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    
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
    # 初始化页面状态
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "首页"
    
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
    if st.session_state.current_page == "首页":
        # 标题
        st.markdown('<p class="big-font">机器学习模型可视化系统</p>', unsafe_allow_html=True)
        st.markdown("---")
        
        # 欢迎横幅
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0; margin-bottom: 3rem;">
            <p style="font-size: 1.2rem; color: #718096; margin-bottom: 1rem;">
                专业的机器学习模型可视化与分析平台
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 主要功能按钮 - 可点击跳转
        st.markdown("### 主要功能")
        st.markdown("""
        <style>
        .clickable-card {
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .clickable-card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 20px 40px rgba(0,0,0,0.15) !important;
        }
        .non-clickable-card {
            cursor: default;
            opacity: 0.7;
        }
        </style>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        # 数据探索按钮
        with col1:
            st.markdown("""
            <div class="clickable-card" style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%); 
                        padding: 2.5rem 2rem; border-radius: 16px; border: 2px solid #667eea; 
                        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.2); text-align: center;">
                <h3 style="color: #667eea; margin: 0 0 1rem 0; font-weight: 700; font-size: 1.5rem;">数据探索</h3>
                <p style="margin: 0; color: #4a5568; line-height: 1.6; font-size: 0.95rem;">查看数据的基本信息、统计特征和分布情况</p>
                <div style="margin-top: 1.5rem; color: #667eea; font-weight: 600; font-size: 0.9rem;">点击进入 →</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("进入数据探索", key="btn_data_explore", use_container_width=True, type="primary"):
                st.session_state.current_page = "数据探索"
                st.rerun()
        
        # 模型评估按钮
        with col2:
            st.markdown("""
            <div class="clickable-card" style="background: linear-gradient(135deg, rgba(79, 172, 254, 0.15) 0%, rgba(0, 242, 254, 0.15) 100%); 
                        padding: 2.5rem 2rem; border-radius: 16px; border: 2px solid #4facfe; 
                        box-shadow: 0 8px 24px rgba(79, 172, 254, 0.2); text-align: center;">
                <h3 style="color: #4facfe; margin: 0 0 1rem 0; font-weight: 700; font-size: 1.5rem;">模型评估</h3>
                <p style="margin: 0; color: #4a5568; line-height: 1.6; font-size: 0.95rem;">查看模型性能指标和可视化分析结果</p>
                <div style="margin-top: 1.5rem; color: #4facfe; font-weight: 600; font-size: 0.9rem;">点击进入 →</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("进入模型评估", key="btn_model_eval", use_container_width=True, type="primary"):
                st.session_state.current_page = "模型评估"
                st.rerun()
        
        # 模型预测按钮
        with col3:
            st.markdown("""
            <div class="clickable-card" style="background: linear-gradient(135deg, rgba(245, 87, 108, 0.15) 0%, rgba(240, 147, 251, 0.15) 100%); 
                        padding: 2.5rem 2rem; border-radius: 16px; border: 2px solid #f5576c; 
                        box-shadow: 0 8px 24px rgba(245, 87, 108, 0.2); text-align: center;">
                <h3 style="color: #f5576c; margin: 0 0 1rem 0; font-weight: 700; font-size: 1.5rem;">模型预测</h3>
                <p style="margin: 0; color: #4a5568; line-height: 1.6; font-size: 0.95rem;">使用训练好的模型进行实时预测</p>
                <div style="margin-top: 1.5rem; color: #f5576c; font-weight: 600; font-size: 0.9rem;">点击进入 →</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("进入模型预测", key="btn_model_predict", use_container_width=True, type="primary"):
                st.session_state.current_page = "模型预测"
                st.rerun()
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # 系统功能说明 - 不可点击的信息卡片
        st.markdown("### 系统功能")
        st.markdown("""
        <div class="info-card">
            <h4 style="color: #667eea; margin-top: 0; font-size: 1.5rem; margin-bottom: 1.5rem;">
                专业的机器学习模型可视化与分析平台
            </h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem;">
                <div class="non-clickable-card" style="padding: 1.5rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 12px; border-left: 4px solid #667eea; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
                    <h5 style="color: #667eea; margin: 0 0 0.75rem 0; font-weight: 700; font-size: 1.1rem;">多模型支持</h5>
                    <p style="margin: 0; color: #4a5568; line-height: 1.6;">支持 XGBoost、LSTM、Transformer 三种先进的机器学习模型，可灵活切换查看不同模型的性能表现</p>
                </div>
                <div class="non-clickable-card" style="padding: 1.5rem; background: linear-gradient(135deg, rgba(118, 75, 162, 0.1) 0%, rgba(240, 147, 251, 0.1) 100%); border-radius: 12px; border-left: 4px solid #764ba2; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
                    <h5 style="color: #764ba2; margin: 0 0 0.75rem 0; font-weight: 700; font-size: 1.1rem;">性能评估</h5>
                    <p style="margin: 0; color: #4a5568; line-height: 1.6;">提供全面的模型性能指标评估，包括 RMSE、R²、MAE、MBE、MAPE、MSE 等多项指标</p>
                </div>
                <div class="non-clickable-card" style="padding: 1.5rem; background: linear-gradient(135deg, rgba(240, 147, 251, 0.1) 0%, rgba(245, 87, 108, 0.1) 100%); border-radius: 12px; border-left: 4px solid #f093fb; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
                    <h5 style="color: #f093fb; margin: 0 0 0.75rem 0; font-weight: 700; font-size: 1.1rem;">多维度预测</h5>
                    <p style="margin: 0; color: #4a5568; line-height: 1.6;">支持同时预测5个输出维度，适用于复杂的多目标回归任务</p>
                </div>
                <div class="non-clickable-card" style="padding: 1.5rem; background: linear-gradient(135deg, rgba(79, 172, 254, 0.1) 0%, rgba(0, 242, 254, 0.1) 100%); border-radius: 12px; border-left: 4px solid #4facfe; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
                    <h5 style="color: #4facfe; margin: 0 0 0.75rem 0; font-weight: 700; font-size: 1.1rem;">交互式可视化</h5>
                    <p style="margin: 0; color: #4a5568; line-height: 1.6;">提供丰富的交互式图表，包括预测对比、散点图分析、误差分布等可视化功能</p>
                </div>
                <div class="non-clickable-card" style="padding: 1.5rem; background: linear-gradient(135deg, rgba(245, 87, 108, 0.1) 0%, rgba(240, 147, 251, 0.1) 100%); border-radius: 12px; border-left: 4px solid #f5576c; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
                    <h5 style="color: #f5576c; margin: 0 0 0.75rem 0; font-weight: 700; font-size: 1.1rem;">实时预测</h5>
                    <p style="margin: 0; color: #4a5568; line-height: 1.6;">支持实时输入特征值进行预测，快速获得模型预测结果</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # =============== 数据探索 ===============
    elif st.session_state.current_page == "数据探索":
        # 返回首页按钮
        if st.button("← 返回首页", key="back_home_1"):
            st.session_state.current_page = "首页"
            st.rerun()
        
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
    
    
    # =============== 模型评估 ===============
    elif st.session_state.current_page == "模型评估":
        # 返回首页按钮和模型选择
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("← 返回首页", key="back_home_2"):
                st.session_state.current_page = "首页"
                st.rerun()
        with col2:
            model_choice = st.selectbox("选择模型", ["XGBoost", "LSTM", "Transformer"], key="model_select_eval")
        
        st.markdown('<p class="medium-font">模型评估</p>', unsafe_allow_html=True)
        st.markdown("---")
        
        # 加载模型
        model = load_model(model_choice)
        
        if model is None:
            st.markdown(f"""
            <div class="info-card" style="background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 152, 0, 0.1) 100%); 
                        border-left-color: #ff9800;">
                <p style="margin: 0; color: #e65100;"><strong>提示：</strong>{model_choice} 模型尚未加载，请检查模型文件是否存在</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # 进行预测
        with st.spinner('正在计算模型性能指标...'):
            y_train_pred_norm = model.predict(data_dict['X_train_norm'])
            y_test_pred_norm = model.predict(data_dict['X_test_norm'])
            
            y_train_pred = processor.inverse_transform_output(y_train_pred_norm)
            y_test_pred = processor.inverse_transform_output(y_test_pred_norm)
        
        # 计算指标
        train_metrics = calculate_metrics_multioutput(data_dict['y_train'], y_train_pred)
        test_metrics = calculate_metrics_multioutput(data_dict['y_test'], y_test_pred)
        
        # 模型信息卡片
        st.markdown(f"""
        <div class="info-card" style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                    border-left-color: #667eea; margin-bottom: 2rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="color: #667eea; margin: 0 0 0.5rem 0; font-weight: 700;">{model_choice} 模型</h3>
                    <p style="margin: 0; color: #4a5568; font-size: 0.95rem;">模型评估完成，性能指标已计算</p>
                </div>
                <div style="font-size: 2rem; color: #667eea;">●</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 显示整体指标 - 使用高级卡片设计
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 整体性能指标")
        
        # 训练集指标卡片
        st.markdown("#### 训练集性能")
        train_cols = st.columns(6)
        train_metric_values = [
            ('RMSE', train_metrics['average']['RMSE'], '#667eea'),
            ('R²', train_metrics['average']['R2'], '#764ba2'),
            ('MAE', train_metrics['average']['MAE'], '#f093fb'),
            ('MBE', train_metrics['average']['MBE'], '#f5576c'),
            ('MAPE', train_metrics['average']['MAPE'], '#4facfe'),
            ('MSE', train_metrics['average']['MSE'], '#00f2fe')
        ]
        
        for idx, (name, value, color) in enumerate(train_metric_values):
            with train_cols[idx]:
                if name == 'R²':
                    display_value = f"{value:.4f}"
                    better = value > 0.9
                elif name == 'MAPE':
                    display_value = f"{value:.2f}%"
                    better = value < 10
                else:
                    display_value = f"{value:.6f}"
                    better = value < 1.0 if name in ['RMSE', 'MAE', 'MBE'] else True
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color}15 0%, {color}05 100%); 
                            padding: 1.5rem 1rem; border-radius: 12px; border: 2px solid {color}40; 
                            text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.08); 
                            transition: all 0.3s ease; margin-bottom: 1rem;">
                    <div style="color: {color}; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.5rem; 
                                text-transform: uppercase; letter-spacing: 0.5px;">{name}</div>
                    <div style="color: #1a202c; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.25rem;">
                        {display_value}
                    </div>
                    <div style="color: {'#10b981' if better else '#ef4444'}; font-size: 0.75rem; font-weight: 500;">
                        {'良好' if better else '需改进'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 测试集指标卡片
        st.markdown("#### 测试集性能")
        test_cols = st.columns(6)
        test_metric_values = [
            ('RMSE', test_metrics['average']['RMSE'], '#667eea'),
            ('R²', test_metrics['average']['R2'], '#764ba2'),
            ('MAE', test_metrics['average']['MAE'], '#f093fb'),
            ('MBE', test_metrics['average']['MBE'], '#f5576c'),
            ('MAPE', test_metrics['average']['MAPE'], '#4facfe'),
            ('MSE', test_metrics['average']['MSE'], '#00f2fe')
        ]
        
        for idx, (name, value, color) in enumerate(test_metric_values):
            with test_cols[idx]:
                if name == 'R²':
                    display_value = f"{value:.4f}"
                    better = value > 0.9
                elif name == 'MAPE':
                    display_value = f"{value:.2f}%"
                    better = value < 10
                else:
                    display_value = f"{value:.6f}"
                    better = value < 1.0 if name in ['RMSE', 'MAE', 'MBE'] else True
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color}15 0%, {color}05 100%); 
                            padding: 1.5rem 1rem; border-radius: 12px; border: 2px solid {color}40; 
                            text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.08); 
                            transition: all 0.3s ease; margin-bottom: 1rem;">
                    <div style="color: {color}; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.5rem; 
                                text-transform: uppercase; letter-spacing: 0.5px;">{name}</div>
                    <div style="color: #1a202c; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.25rem;">
                        {display_value}
                    </div>
                    <div style="color: {'#10b981' if better else '#ef4444'}; font-size: 0.75rem; font-weight: 500;">
                        {'良好' if better else '需改进'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # 各输出维度指标对比
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 各输出维度指标对比")
        
        dataset_choice = st.radio("选择数据集", ["训练集", "测试集"], horizontal=True, key="dataset_choice")
        metrics_to_show = train_metrics if dataset_choice == "训练集" else test_metrics
        
        fig = plot_metrics_comparison(metrics_to_show)
        st.plotly_chart(fig, use_container_width=True)
        
        # 详细评估
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 详细分析")
        
        output_idx = st.selectbox("选择输出维度", range(5), format_func=lambda x: f"输出 {x+1}", key="output_dim")
        
        tab1, tab2, tab3 = st.tabs(["预测对比", "散点图分析", "误差分布"])
        
        with tab1:
            st.markdown("#### 预测值 vs 真实值对比")
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
            st.markdown("#### 散点图分析")
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
            st.markdown("#### 误差分布分析")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = plot_error_histogram(data_dict['y_train'], y_train_pred, output_idx)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = plot_error_histogram(data_dict['y_test'], y_test_pred, output_idx)
                st.plotly_chart(fig, use_container_width=True)
        
        # 下载预测结果
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 导出结果")
        
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
    elif st.session_state.current_page == "模型预测":
        # 返回首页按钮和模型选择
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("← 返回首页", key="back_home_3"):
                st.session_state.current_page = "首页"
                st.rerun()
        with col2:
            model_choice = st.selectbox("选择模型", ["XGBoost", "LSTM", "Transformer"], key="model_select_predict")
        
        st.markdown('<p class="medium-font">模型预测</p>', unsafe_allow_html=True)
        st.markdown("---")
        
        # 加载模型
        model = load_model(model_choice)
        
        if model is None:
            st.markdown(f"""
            <div class="info-card" style="background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 152, 0, 0.1) 100%); 
                        border-left-color: #ff9800;">
                <p style="margin: 0; color: #e65100;"><strong>提示：</strong>{model_choice} 模型尚未加载，请检查模型文件是否存在</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # 模型信息卡片
        st.markdown(f"""
        <div class="info-card" style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                    border-left-color: #667eea; margin-bottom: 2rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="color: #667eea; margin: 0 0 0.5rem 0; font-weight: 700;">{model_choice} 模型</h3>
                    <p style="margin: 0; color: #4a5568; font-size: 0.95rem;">模型已就绪，可以进行预测</p>
                </div>
                <div style="font-size: 2rem; color: #667eea;">●</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 特征输入")
        
        # 方式1: 手动输入
        st.markdown("#### 自定义输入")
        st.markdown("""
        <div class="info-card" style="margin-bottom: 1.5rem;">
            <p style="margin: 0; color: #4a5568;">在下方输入框中填入特征值，系统将使用 {model_choice} 模型进行预测</p>
        </div>
        """.format(model_choice=model_choice), unsafe_allow_html=True)
        
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
        
        if st.button("开始预测", type="primary", use_container_width=True):
            # 归一化输入
            X_input = np.array(input_features).reshape(1, -1)
            X_input_norm = processor.input_scaler.transform(X_input)
            
            # 预测
            with st.spinner('正在计算预测结果...'):
                y_pred_norm = model.predict(X_input_norm)
                y_pred = processor.inverse_transform_output(y_pred_norm)
            
            # 显示结果
            st.markdown("""
            <div class="info-card" style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%); 
                        border-left-color: #10b981; margin-top: 2rem;">
                <p style="margin: 0; color: #065f46; font-size: 1.1em;"><strong>预测完成</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 预测结果")
            
            # 使用卡片展示预测结果
            result_cols = st.columns(5)
            colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
            
            for idx, (col, color, value) in enumerate(zip(result_cols, colors, y_pred.flatten())):
                with col:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {color}15 0%, {color}05 100%); 
                                padding: 1.5rem 1rem; border-radius: 12px; border: 2px solid {color}40; 
                                text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 1rem;">
                        <div style="color: {color}; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.5rem; 
                                    text-transform: uppercase; letter-spacing: 0.5px;">输出 {idx+1}</div>
                        <div style="color: #1a202c; font-size: 1.8rem; font-weight: 700;">
                            {value:.4f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # 可视化
            st.markdown("<br>", unsafe_allow_html=True)
            fig = go.Figure(data=[
                go.Bar(
                    x=[f'输出 {i+1}' for i in range(5)], 
                    y=y_pred.flatten(), 
                    marker_color=colors,
                    marker_line_color='white',
                    marker_line_width=2,
                    text=y_pred.flatten(),
                    texttemplate='%{text:.4f}',
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>预测值: %{y:.4f}<extra></extra>'
                )
            ])
            fig.update_layout(
                title=dict(text='预测结果可视化', font=dict(size=18, color='#2d3748')),
                xaxis_title=dict(text='输出维度', font=dict(size=14, color='#4a5568')),
                yaxis_title=dict(text='预测值', font=dict(size=14, color='#4a5568')),
                height=450,
                template='plotly_white',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 方式2: 使用测试集样本
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("#### 测试样本预测")
        st.markdown("""
        <div class="info-card" style="margin-bottom: 1.5rem;">
            <p style="margin: 0; color: #4a5568;">从测试集中选择一个样本进行预测，系统将显示预测值与真实值的对比分析</p>
        </div>
        """, unsafe_allow_html=True)
        
        sample_idx = st.selectbox(
            "选择测试集样本",
            range(data_dict['n_test']),
            format_func=lambda x: f"样本 {x+1}",
            key="sample_select"
        )
        
        if st.button("预测选定样本", use_container_width=True):
            X_sample = data_dict['X_test_norm'][sample_idx:sample_idx+1]
            y_true_sample = data_dict['y_test'][sample_idx]
            
            # 预测
            with st.spinner('正在计算预测结果...'):
                y_pred_norm = model.predict(X_sample)
                y_pred_sample = processor.inverse_transform_output(y_pred_norm).flatten()
            
            # 对比结果
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 预测结果对比")
            
            # 使用卡片展示对比结果
            comparison_cols = st.columns(5)
            
            for idx, (col, color) in enumerate(zip(comparison_cols, colors)):
                with col:
                    true_val = y_true_sample[idx]
                    pred_val = y_pred_sample[idx]
                    error = abs(true_val - pred_val)
                    error_pct = (error / abs(true_val) * 100) if true_val != 0 else 0
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {color}15 0%, {color}05 100%); 
                                padding: 1.5rem 1rem; border-radius: 12px; border: 2px solid {color}40; 
                                text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 1rem;">
                        <div style="color: {color}; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.75rem; 
                                    text-transform: uppercase; letter-spacing: 0.5px;">输出 {idx+1}</div>
                        <div style="margin-bottom: 0.5rem;">
                            <div style="color: #718096; font-size: 0.75rem; margin-bottom: 0.25rem;">真实值</div>
                            <div style="color: #1a202c; font-size: 1.2rem; font-weight: 700;">{true_val:.4f}</div>
                        </div>
                        <div style="margin-bottom: 0.5rem; padding-top: 0.5rem; border-top: 1px solid {color}30;">
                            <div style="color: #718096; font-size: 0.75rem; margin-bottom: 0.25rem;">预测值</div>
                            <div style="color: #1a202c; font-size: 1.2rem; font-weight: 700;">{pred_val:.4f}</div>
                        </div>
                        <div style="padding-top: 0.5rem; border-top: 1px solid {color}30;">
                            <div style="color: {'#10b981' if error_pct < 5 else '#ef4444'}; font-size: 0.7rem; font-weight: 600;">
                                误差: {error_pct:.2f}%
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # 可视化对比
            st.markdown("<br>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='真实值',
                x=[f'输出 {i+1}' for i in range(5)],
                y=y_true_sample,
                marker_color='#e74c3c',
                marker_line_color='white',
                marker_line_width=2,
                text=y_true_sample,
                texttemplate='%{text:.4f}',
                textposition='outside',
                hovertemplate='<b>真实值</b><br>%{x}<br>值: %{y:.4f}<extra></extra>'
            ))
            fig.add_trace(go.Bar(
                name='预测值',
                x=[f'输出 {i+1}' for i in range(5)],
                y=y_pred_sample,
                marker_color=colors,
                marker_line_color='white',
                marker_line_width=2,
                text=y_pred_sample,
                texttemplate='%{text:.4f}',
                textposition='outside',
                hovertemplate='<b>预测值</b><br>%{x}<br>值: %{y:.4f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(text='真实值 vs 预测值对比', font=dict(size=18, color='#2d3748')),
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

