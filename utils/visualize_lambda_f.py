import plotly.graph_objects as go

def visualize_lambda_f(lambda_f_history, step, filename="lambda_f_visualization.html"):
    """
    lambda_f を3Dベクトルとして可視化。
    Args:
        lambda_f_history: lambda_f の履歴 (n_steps, 3)
        step: 現在のステップ
        filename: 出力ファイル名
    """
    lambda_f = lambda_f_history[step]
    fig = go.Figure()
    
    # 原点からのベクトル
    fig.add_trace(go.Scatter3d(
        x=[0, lambda_f[0]], y=[0, lambda_f[1]], z=[0, lambda_f[2]],
        mode='lines+markers',
        line=dict(color='blue', width=5),
        marker=dict(size=5),
        name=f'Step {step}'
    ))
    
    # 軸の設定
    fig.update_layout(
        scene=dict(
            xaxis_title='Bind', yaxis_title='Move', zaxis_title='Split',
            xaxis=dict(range=[-1, 1]), yaxis=dict(range=[-1, 1]), zaxis=dict(range=[-1, 1])
        ),
        title=f'Lambda_F Direction at Step {step}'
    )
    fig.write_html(filename)
