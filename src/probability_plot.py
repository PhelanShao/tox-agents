import pandas as pd
import plotly.graph_objects as go
import numpy as np

def get_color(probability):
    """根据概率值返回颜色"""
    if probability < 0.45:
        return 'green'
    elif probability <= 0.6:
        return 'yellow'
    else:
        return 'red'

def create_probability_plot(csv_path):
    """创建概率热图"""
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        # 确保概率值在0-1范围内
        if 'probability' in df.columns:
            probabilities = df['probability'].values
            if np.max(probabilities) > 1 or np.min(probabilities) < 0:
                print("Warning: Invalid probability values detected")
                return None
        else:
            print("Error: No probability column found in CSV file")
            return None
        
        # 创建帧序列作为x轴
        frames = list(range(len(df)))
        
        # 创建图表
        fig = go.Figure()
        
        # 添加热图
        fig.add_trace(
            go.Heatmap(
                z=[probabilities],
                colorscale=[
                    [0, 'green'],
                    [0.45, 'green'],
                    [0.45, 'yellow'],
                    [0.6, 'yellow'],
                    [0.6, 'red'],
                    [1, 'red']
                ],
                showscale=False,
                hoverongaps=False,
                hovertemplate='Frame: %{x}<br>Probability: %{z:.3f}<extra></extra>'
            )
        )
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text='Probability Analysis',
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top'
            ),
            height=150,
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=30),
            xaxis=dict(
                title='Frame',
                showgrid=False,
                range=[-0.5, len(frames)-0.5]
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False
            ),
            plot_bgcolor='white'
        )
        
        # 添加交互模式配置
        fig.update_layout(
            dragmode='zoom',
            hovermode='x',
            hoverdistance=100,
            spikedistance=1000,
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
        # 添加x轴网格线
        fig.update_xaxes(
            showspikes=True,
            spikecolor="gray",
            spikesnap="cursor",
            spikemode="across",
            spikethickness=1
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating probability plot: {str(e)}")
        return None
