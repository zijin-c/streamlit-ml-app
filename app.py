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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 自定义样式
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
.medium-font {
    font-size:20px !important;
    font-weight: bold;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
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
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x, y=y_true[:, output_idx],
        mode='lines+markers',
        name='真实值',
        line=dict(color='red', width=2),
        marker=dict(size=6, symbol='star')
    ))
    
    fig.add_trace(go.Scatter(
        x=x, y=y_pred[:, output_idx],
        mode='lines+markers',
        name='预测值',
        line=dict(color='blue', width=2),
        marker=dict(size=6, symbol='circle')
    ))
    
    fig.update_layout(
        title=f'{dataset_name} - 输出维度 {output_idx+1}',
        xaxis_title='样本序号',
        yaxis_title='值',
        hovermode='x unified',
        height=400
    )
    
    return fig


def plot_scatter_interactive(y_true, y_pred, output_idx, dataset_name, metrics):
    """使用Plotly绘制交互式散点图"""
    fig = go.Figure()
    
    # 散点图
    fig.add_trace(go.Scatter(
        x=y_true[:, output_idx],
        y=y_pred[:, output_idx],
        mode='markers',
        name='预测值',
        marker=dict(size=8, opacity=0.6, color='blue')
    ))
    
    # 理想线
    min_val = min(y_true[:, output_idx].min(), y_pred[:, output_idx].min())
    max_val = max(y_true[:, output_idx].max(), y_pred[:, output_idx].max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='理想线',
        line=dict(color='black', width=2, dash='dash')
    ))
    
    # 拟合线
    z = np.polyfit(y_true[:, output_idx], y_pred[:, output_idx], 1)
    p = np.poly1d(z)
    x_line = np.linspace(min_val, max_val, 100)
    
    fig.add_trace(go.Scatter(
        x=x_line,
        y=p(x_line),
        mode='lines',
        name=f'拟合线: y={z[0]:.2f}x+{z[1]:.2f}',
        line=dict(color='magenta', width=2)
    ))
    
    fig.update_layout(
        title=f'{dataset_name} - 输出维度 {output_idx+1}<br>R²={metrics["R2"]:.4f}, RMSE={metrics["RMSE"]:.4f}',
        xaxis_title='真实值',
        yaxis_title='预测值',
        height=500
    )
    
    return fig


def plot_error_histogram(y_true, y_pred, output_idx):
    """绘制误差直方图"""
    errors = y_true[:, output_idx] - y_pred[:, output_idx]
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=30,
        name='误差分布',
        marker_color='steelblue'
    ))
    
    fig.update_layout(
        title=f'误差直方图 - 输出维度 {output_idx+1}',
        xaxis_title='误差',
        yaxis_title='频数',
        height=400
    )
    
    return fig


def plot_metrics_comparison(metrics_dict):
    """绘制各输出维度的指标对比"""
    output_names = [f'输出{i+1}' for i in range(5)]
    metric_names = ['RMSE', 'R2', 'MAE', 'MAPE', 'MSE']
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=metric_names,
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
    
    for idx, metric in enumerate(metric_names):
        values = [metrics_dict[f'output_{i+1}'][metric] for i in range(5)]
        row, col = positions[idx]
        
        fig.add_trace(
            go.Bar(x=output_names, y=values, name=metric, showlegend=False),
            row=row, col=col
        )
    
    fig.update_layout(height=600, showlegend=False, title_text="各输出维度评估指标对比")
    
    return fig


def main():
    # 标题
    st.markdown('<p class="big-font">📊 机器学习模型可视化系统</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 侧边栏
    st.sidebar.markdown('<p class="medium-font">⚙️ 控制面板</p>', unsafe_allow_html=True)
    
    # 页面选择
    page = st.sidebar.radio(
        "选择功能",
        ["🏠 首页", "📁 数据探索", "🎯 模型训练", "📈 模型评估", "🔮 模型预测"]
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
    if page == "🏠 首页":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**训练样本数**: {data_dict['n_train']}")
        with col2:
            st.info(f"**测试样本数**: {data_dict['n_test']}")
        with col3:
            st.info(f"**特征维度**: {data_dict['n_features']}")
        
        st.markdown("---")
        
        st.markdown("### 系统介绍")
        st.write("""
        本系统提供了完整的机器学习模型训练和可视化功能，包括：
        
        - **三种模型**: XGBoost, LSTM, Transformer
        - **自动优化**: 贝叶斯超参数优化
        - **多输出回归**: 同时预测5个输出维度
        - **可视化分析**: 交互式图表展示
        - **评估指标**: RMSE, R², MAE, MBE, MAPE, MSE
        """)
        
        st.markdown("### 快速开始")
        st.write("""
        1. **数据探索**: 查看数据的基本信息和分布
        2. **模型训练**: 选择模型类型并训练（需要在命令行运行训练脚本）
        3. **模型评估**: 查看训练好的模型性能指标
        4. **模型预测**: 使用训练好的模型进行预测
        """)
        
        st.markdown("### 训练模型")
        st.code("""
# 训练XGBoost模型
python train_xgboost.py

# 训练LSTM模型
python train_lstm.py

# 训练Transformer模型
python train_transformer.py
        """)
    
    # =============== 数据探索 ===============
    elif page == "📁 数据探索":
        st.markdown('<p class="medium-font">数据探索</p>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📊 数据统计", "📈 数据可视化", "🔍 数据详情"])
        
        with tab1:
            st.subheader("训练集统计信息")
            train_df = pd.DataFrame(data_dict['y_train'], columns=[f'输出{i+1}' for i in range(5)])
            st.write(train_df.describe())
            
            st.subheader("测试集统计信息")
            test_df = pd.DataFrame(data_dict['y_test'], columns=[f'输出{i+1}' for i in range(5)])
            st.write(test_df.describe())
        
        with tab2:
            st.subheader("输出维度分布")
            output_idx = st.selectbox("选择输出维度", range(5), format_func=lambda x: f"输出 {x+1}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=data_dict['y_train'][:, output_idx],
                    name='训练集',
                    opacity=0.7,
                    marker_color='blue'
                ))
                fig.update_layout(title=f'训练集 - 输出 {output_idx+1} 分布', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=data_dict['y_test'][:, output_idx],
                    name='测试集',
                    opacity=0.7,
                    marker_color='red'
                ))
                fig.update_layout(title=f'测试集 - 输出 {output_idx+1} 分布', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # 相关性矩阵
            st.subheader("输出维度相关性矩阵")
            corr_matrix = train_df.corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                           color_continuous_scale='RdBu_r')
            fig.update_layout(title='相关性热图', height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("原始数据预览")
            
            data_choice = st.radio("选择数据集", ["训练集", "测试集"])
            
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
            
            st.dataframe(display_df, height=400)
            
            # 下载按钮
            csv = display_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下载数据",
                data=csv,
                file_name=f"{data_choice}_data.csv",
                mime="text/csv"
            )
    
    # =============== 模型训练 ===============
    elif page == "🎯 模型训练":
        st.markdown('<p class="medium-font">模型训练</p>', unsafe_allow_html=True)
        
        st.info("⚠️ 由于训练过程可能需要较长时间，请在命令行中运行训练脚本")
        
        model_choice = st.selectbox("选择模型", ["XGBoost", "LSTM", "Transformer"])
        
        st.markdown(f"### {model_choice} 模型训练")
        
        if model_choice == "XGBoost":
            st.write("""
            **XGBoost模型特点**:
            - 基于梯度提升树
            - 适合处理表格数据
            - 训练速度快
            - 可解释性强
            
            **超参数**:
            - n_estimators: 树的数量 (10-5000)
            - max_depth: 树的最大深度 (10-20)
            - learning_rate: 学习率 (0.0001-1)
            """)
            
            st.code("python train_xgboost.py")
            
        elif model_choice == "LSTM":
            st.write("""
            **LSTM模型特点**:
            - 长短期记忆网络
            - 适合处理序列数据
            - 能捕捉时间依赖关系
            
            **超参数**:
            - hidden_size: 隐藏层单元数 (20-50)
            - learning_rate: 学习率 (1e-3 - 1e-2)
            - l2_regularization: L2正则化 (1e-4 - 1e-3)
            """)
            
            st.code("python train_lstm.py")
            
        elif model_choice == "Transformer":
            st.write("""
            **Transformer模型特点**:
            - 自注意力机制
            - 能并行处理序列
            - 捕捉长距离依赖
            
            **超参数**:
            - nhead: 注意力头数 (2-8)
            - learning_rate: 学习率 (1e-4 - 1e-1)
            - l2_regularization: L2正则化 (1e-6 - 1e-2)
            - batch_size: 批大小 (16-128)
            """)
            
            st.code("python train_transformer.py")
        
        st.markdown("---")
        st.warning("训练完成后，模型将保存在 `models/` 目录下，可以在「模型评估」页面查看结果")
    
    # =============== 模型评估 ===============
    elif page == "📈 模型评估":
        st.markdown('<p class="medium-font">模型评估</p>', unsafe_allow_html=True)
        
        model_choice = st.sidebar.selectbox("选择模型", ["XGBoost", "LSTM", "Transformer"])
        
        # 加载模型
        model = load_model(model_choice)
        
        if model is None:
            st.warning(f"⚠️ {model_choice} 模型尚未训练，请先运行训练脚本")
            st.code(f"python train_{model_choice.lower()}.py")
            return
        
        st.success(f"✅ {model_choice} 模型加载成功")
        
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
        st.subheader("📊 整体性能指标")
        
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
            st.dataframe(metrics_df_train, hide_index=True)
        
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
            st.dataframe(metrics_df_test, hide_index=True)
        
        # 各输出维度指标对比
        st.markdown("---")
        st.subheader("📈 各输出维度指标对比")
        
        dataset_choice = st.radio("选择数据集", ["训练集", "测试集"], horizontal=True)
        metrics_to_show = train_metrics if dataset_choice == "训练集" else test_metrics
        
        fig = plot_metrics_comparison(metrics_to_show)
        st.plotly_chart(fig, use_container_width=True)
        
        # 详细评估
        st.markdown("---")
        st.subheader("🔍 详细评估")
        
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
        st.markdown("---")
        st.subheader("💾 下载预测结果")
        
        col1, col2 = st.columns(2)
        
        with col1:
            train_pred_df = pd.DataFrame(y_train_pred, columns=[f'预测维度_{i+1}' for i in range(5)])
            csv = train_pred_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下载训练集预测结果",
                data=csv,
                file_name=f"{model_choice.lower()}_train_predictions.csv",
                mime="text/csv"
            )
        
        with col2:
            test_pred_df = pd.DataFrame(y_test_pred, columns=[f'预测维度_{i+1}' for i in range(5)])
            csv = test_pred_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下载测试集预测结果",
                data=csv,
                file_name=f"{model_choice.lower()}_test_predictions.csv",
                mime="text/csv"
            )
    
    # =============== 模型预测 ===============
    elif page == "🔮 模型预测":
        st.markdown('<p class="medium-font">模型预测</p>', unsafe_allow_html=True)
        
        model_choice = st.sidebar.selectbox("选择模型", ["XGBoost", "LSTM", "Transformer"])
        
        # 加载模型
        model = load_model(model_choice)
        
        if model is None:
            st.warning(f"⚠️ {model_choice} 模型尚未训练，请先运行训练脚本")
            return
        
        st.success(f"✅ {model_choice} 模型加载成功")
        
        st.subheader("输入特征")
        
        # 方式1: 手动输入
        st.markdown("#### 方式1: 手动输入特征值")
        
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
        
        if st.button("🚀 开始预测", type="primary"):
            # 归一化输入
            X_input = np.array(input_features).reshape(1, -1)
            X_input_norm = processor.input_scaler.transform(X_input)
            
            # 预测
            with st.spinner('正在预测...'):
                y_pred_norm = model.predict(X_input_norm)
                y_pred = processor.inverse_transform_output(y_pred_norm)
            
            # 显示结果
            st.success("预测完成！")
            
            st.subheader("📊 预测结果")
            
            result_df = pd.DataFrame({
                '输出维度': [f'输出 {i+1}' for i in range(5)],
                '预测值': y_pred.flatten()
            })
            
            st.dataframe(result_df, hide_index=True, use_container_width=True)
            
            # 可视化
            fig = go.Figure(data=[
                go.Bar(x=result_df['输出维度'], y=result_df['预测值'], marker_color='steelblue')
            ])
            fig.update_layout(
                title='预测结果可视化',
                xaxis_title='输出维度',
                yaxis_title='预测值',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 方式2: 使用测试集样本
        st.markdown("---")
        st.markdown("#### 方式2: 从测试集选择样本")
        
        sample_idx = st.selectbox(
            "选择测试集样本",
            range(data_dict['n_test']),
            format_func=lambda x: f"样本 {x+1}"
        )
        
        if st.button("🎯 预测选定样本"):
            X_sample = data_dict['X_test_norm'][sample_idx:sample_idx+1]
            y_true_sample = data_dict['y_test'][sample_idx]
            
            # 预测
            with st.spinner('正在预测...'):
                y_pred_norm = model.predict(X_sample)
                y_pred_sample = processor.inverse_transform_output(y_pred_norm).flatten()
            
            # 对比结果
            st.subheader("📊 预测对比")
            
            comparison_df = pd.DataFrame({
                '输出维度': [f'输出 {i+1}' for i in range(5)],
                '真实值': y_true_sample,
                '预测值': y_pred_sample,
                '误差': y_true_sample - y_pred_sample,
                '相对误差(%)': np.abs((y_true_sample - y_pred_sample) / y_true_sample * 100)
            })
            
            st.dataframe(comparison_df, hide_index=True, use_container_width=True)
            
            # 可视化对比
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='真实值',
                x=comparison_df['输出维度'],
                y=comparison_df['真实值'],
                marker_color='red'
            ))
            fig.add_trace(go.Bar(
                name='预测值',
                x=comparison_df['输出维度'],
                y=comparison_df['预测值'],
                marker_color='blue'
            ))
            
            fig.update_layout(
                title='真实值 vs 预测值',
                xaxis_title='输出维度',
                yaxis_title='值',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()

