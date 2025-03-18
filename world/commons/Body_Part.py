from math_ops.Matrix_4x4 import Matrix_4x4

class Body_Part():
    """
    表示机器人身体的一个部分（如手臂、腿部等）。
    """
    def __init__(self, mass) -> None:
        """
        初始化身体部分。
        :param mass: 身体部分的质量（单位：千克）。
        """
        self.mass = float(mass)  # 将质量转换为浮点数并保存
        self.joints = []  # 初始化关节列表，用于存储与该身体部分相连的关节
        self.transform = Matrix_4x4()  # 初始化变换矩阵，表示身体部分到头部的变换
