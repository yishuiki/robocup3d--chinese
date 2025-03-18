import numpy as np

def run_mlp(obs, weights, activation_function="tanh"):
    ''' 
    使用 NumPy 运行多层感知器（MLP）神经网络的前向传播过程。
    
    参数
    ----------
    obs : ndarray
        浮点数数组，表示神经网络的输入。
    weights : list
        包含 MLP 各层权重的列表，每一层的权重是一个元组 (bias, kernel)。
    activation_function : str
        隐藏层的激活函数，默认为 "tanh"。
        设置为 "none" 可以禁用激活函数。
    '''
    # 将输入数据转换为 float32 类型
    obs = obs.astype(np.float32, copy=False)
    # 初始化输出变量
    out = obs

    # 遍历隐藏层
    for w in weights[:-1]:  # 对于每个隐藏层
        # 计算当前层的输出
        out = np.matmul(w[1], out) + w[0]  # w[1] 是权重矩阵，w[0] 是偏置向量
        # 应用激活函数
        if activation_function == "tanh":
            np.tanh(out, out=out)  # 使用 tanh 激活函数
        elif activation_function != "none":
            raise NotImplementedError  # 如果激活函数不是 "tanh" 或 "none"，抛出异常

    # 计算最后一层的输出
    return np.matmul(weights[-1][1], out) + weights[-1][0]  # 最后一层没有激活函数
