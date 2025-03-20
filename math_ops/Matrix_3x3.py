from math import asin, atan2, pi, sqrt
import numpy as np

class Matrix_3x3():
    """
    3x3 矩阵类，主要用于处理旋转矩阵。
    提供了多种构造方法、旋转操作和矩阵运算。
    """

    def __init__(self, matrix=None) -> None:
        """
        构造函数示例：
        a = Matrix_3x3()                           # 创建单位矩阵
        b = Matrix_3x3([[1,1,1],[2,2,2],[3,3,3]]) # 手动初始化矩阵
        c = Matrix_3x3([1,1,1,2,2,2,3,3,3])       # 手动初始化矩阵
        d = Matrix_3x3(b)                         # 复制构造函数
        """
        if matrix is None:
            self.m = np.identity(3)  # 创建单位矩阵
        elif type(matrix) == Matrix_3x3: 
            self.m = np.copy(matrix.m)  # 复制另一个 Matrix_3x3 对象的矩阵
        else:
            self.m = np.asarray(matrix)  # 将输入转换为 numpy 数组
            self.m.shape = (3,3)  # 确保矩阵形状为 3x3，如果需要的话进行重塑，否则抛出错误

        # 定义旋转快捷方式，用于快速调用旋转方法
        self.rotation_shortcuts = {
            (1, 0, 0): self.rotate_x_rad, (-1, 0, 0): self._rotate_x_neg_rad,
            (0, 1, 0): self.rotate_y_rad, (0, -1, 0): self._rotate_y_neg_rad,
            (0, 0, 1): self.rotate_z_rad, (0, 0, -1): self._rotate_z_neg_rad
        }

    @classmethod
    def from_rotation_deg(cls, euler_vec):
        """
        从欧拉角（以度为单位）创建旋转矩阵。
        旋转顺序：RotZ * RotY * RotX

        参数：
        euler_vec : array_like, length 3
            包含欧拉角（x,y,z）的向量，也称为（横滚角，俯仰角，偏航角）

        示例：
        Matrix_3x3.from_rotation_deg((roll,pitch,yaw))    # 创建：RotZ(yaw)*RotY(pitch)*RotX(roll)
        """
        mat = cls().rotate_z_deg(euler_vec[2], True).rotate_y_deg(euler_vec[1], True).rotate_x_deg(euler_vec[0], True)
        return mat

    def get_roll_deg(self):
        """
        获取绕 x 轴的角度（以度为单位），旋转顺序：RotZ*RotY*RotX=Rot
        """
        if self.m[2,1] == 0 and self.m[2,2] == 0: 
            return 180
        return atan2(self.m[2,1], self.m[2,2]) * 180 / pi

    def get_pitch_deg(self):
        """
        获取绕 y 轴的角度（以度为单位），旋转顺序：RotZ*RotY*RotX=Rot
        """
        return atan2(-self.m[2,0], sqrt(self.m[2,1]*self.m[2,1] + self.m[2,2]*self.m[2,2])) * 180 / pi

    def get_yaw_deg(self):
        """
        获取绕 z 轴的角度（以度为单位），旋转顺序：RotZ*RotY*RotX=Rot
        """
        if self.m[1,0] == 0 and self.m[0,0] == 0: 
            return atan2(self.m[0,1], self.m[1,1]) * 180 / pi
        return atan2(self.m[1,0], self.m[0,0]) * 180 / pi

    def get_inclination_deg(self):
        """
        获取 z 轴相对于参考 z 轴的倾斜角度
        """
        return 90 - (asin(self.m[2,2]) * 180 / pi)

    def rotate_deg(self, rotation_vec, rotation_deg, in_place=False):
        """
        旋转当前旋转矩阵

        参数：
        rotation_vec : array_like, length 3
            旋转向量
        rotation_deg : float
            旋转角度（以度为单位）
        in_place: bool, optional
            * True: 内部矩阵在原地更改（默认）
            * False: 返回一个新的矩阵，当前矩阵保持不变
        
        返回值：
        result : Matrix_3x3 
            如果 in_place 为 True，则返回 self
        """
        return self.rotate_rad(rotation_vec, rotation_deg * (pi/180) , in_place)

    def rotate_rad(self, rotation_vec, rotation_rad, in_place=False):
        """
        旋转当前旋转矩阵

        参数：
        rotation_vec : array_like, length 3
            旋转向量
        rotation_rad : float
            旋转角度（以弧度为单位）
        in_place: bool, optional
            * True: 内部矩阵在原地更改（默认）
            * False: 返回一个新的矩阵，当前矩阵保持不变
        
        返回值：
        result : Matrix_3x3 
            如果 in_place 为 True，则返回 self
        """
        if rotation_rad == 0: return

        shortcut = self.rotation_shortcuts.get(tuple(a for a in rotation_vec))
        if shortcut:
            return shortcut(rotation_rad, in_place)
            
        c = np.cos(rotation_rad)
        c1 = 1 - c
        s = np.sin(rotation_rad)
        x = rotation_vec[0]
        y = rotation_vec[1]
        z = rotation_vec[2]
        xxc1 = x * x * c1
        yyc1 = y * y * c1
        zzc1 = z * z * c1
        xyc1 = x * y * c1
        xzc1 = x * z * c1
        yzc1 = y * z * c1
        xs = x * s
        ys = y * s
        zs = z * s

        mat = np.array([
        [xxc1 +  c,  xyc1 - zs,  xzc1 + ys],
        [xyc1 + zs,  yyc1 +  c,  yzc1 - xs],
        [xzc1 - ys,  yzc1 + xs,  zzc1 +  c]])

        return self.multiply(mat, in_place)

    def _rotate_x_neg_rad(self, rotation_rad, in_place=False):
        self.rotate_x_rad(-rotation_rad, in_place)

    def _rotate_y_neg_rad(self, rotation_rad, in_place=False):
        self.rotate_y_rad(-rotation_rad, in_place)

    def _rotate_z_neg_rad(self, rotation_rad, in_place=False):
        self.rotate_z_rad(-rotation_rad, in_place)

    def rotate_x_rad(self, rotation_rad, in_place=False):
        """
        绕 x 轴旋转当前旋转矩阵

        参数：
        rotation_rad : float
            旋转角度（以弧度为单位）
        in_place: bool, optional
            * True: 内部矩阵在原地更改（默认）
            * False: 返回一个新的矩阵，当前矩阵保持不变
        
        返回值：
        result : Matrix_3x3 
            如果 in_place 为 True，则返回 self
        """
        if rotation_rad == 0: 
            return self if in_place else Matrix_3x3(self)
 
        c = np.cos(rotation_rad)
        s = np.sin(rotation_rad)

        mat = np.array([
        [1, 0, 0],
        [0, c,-s],
        [0, s, c]])

        return self.multiply(mat, in_place)

    def rotate_y_rad(self, rotation_rad, in_place=False):
        """
        绕 y 轴旋转当前旋转矩阵

        参数：
        rotation_rad : float
            旋转角度（以弧度为单位）
        in_place: bool, optional
            * True: 内部矩阵在原地更改（默认）
            * False: 返回一个新的矩阵，当前矩阵保持不变
        
        返回值：
        result : Matrix_3x3 
            如果 in_place 为 True，则返回 self
        """
        if rotation_rad == 0: 
            return self if in_place else Matrix_3x3(self)
 
        c = np.cos(rotation_rad)
        s = np.sin(rotation_rad)

        mat = np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]])

        return self.multiply(mat, in_place)

    def rotate_z_rad(self, rotation_rad, in_place=False):
        """
        绕 z 轴旋转当前旋转矩阵

        参数：
        rotation_rad : float
            旋转角度（以弧度为单位）
        in_place: bool, optional
            * True: 内部矩阵在原地更改（默认）
            * False: 返回一个新的矩阵，当前矩阵保持不变
        
        返回值：
        result : Matrix_3x3 
            如果 in_place 为 True，则返回 self
        """
        if rotation_rad == 0: 
            return self if in_place else Matrix_3x3(self)
 
        c = np.cos(rotation_rad)
        s = np.sin(rotation_rad)

        mat = np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1]])

        return self.multiply(mat, in_place)

    def rotate_x_deg(self, rotation_deg, in_place=False):
        """
        绕 x 轴旋转当前旋转矩阵

        参数：
        rotation_deg : float
            旋转角度（以度为单位）
        in_place: bool, optional
            * True: 内部矩阵在原地更改（默认）
            * False: 返回一个新的矩阵，当前矩阵保持不变
        
        返回值：
        result : Matrix_3x3 
            如果 in_place 为 True，则返回 self
        """
        return self.rotate_x_rad(rotation_deg * (pi/180), in_place)

    def rotate_y_deg(self, rotation_deg, in_place=False):
        """
        绕 y 轴旋转当前旋转矩阵

        参数：
        rotation_deg : float
            旋转角度（以度为单位）
        in_place: bool, optional
            * True: 内部矩阵在原地更改（默认）
            * False: 返回一个新的矩阵，当前矩阵保持不变
        
        返回值：
        result : Matrix_3x3 
            如果 in_place 为 True，则返回 self
        """
        return self.rotate_y_rad(rotation_deg * (pi/180), in_place)

    def rotate_z_deg(self, rotation_deg, in_place=False):
        """
        绕 z 轴旋转当前旋转矩阵

        参数：
        rotation_deg : float
            旋转角度（以度为单位）
        in_place: bool, optional
            * True: 内部矩阵在原地更改（默认）
            * False: 返回一个新的矩阵，当前矩阵保持不变
        
        返回值：
        result : Matrix_3x3 
            如果 in_place 为 True，则返回 self
        """
        return self.rotate_z_rad(rotation_deg * (pi/180), in_place)

    def invert(self, in_place=False):
        """
        反转当前旋转矩阵

        参数：
        in_place: bool, optional
            * True: 内部矩阵在原地更改（默认）
            * False: 返回一个新的矩阵，当前矩阵保持不变
        
        返回值：
        result : Matrix_3x3 
            如果 in_place 为 True，则返回 self
        """
        if in_place:
            self.m = np.linalg.inv(self.m)
            return self
        else:
            return Matrix_3x3(np.linalg.inv(self.m))

    def multiply(self, mat, in_place=False, reverse_order=False):
        """
        将当前旋转矩阵与 mat 相乘

        参数：
        mat : Matrix_3x3 或 array_like
            乘数矩阵或 3D 向量
        in_place: bool, optional
            - True: 内部矩阵在原地更改
            - False: 返回一个新的矩阵，当前矩阵保持不变（默认） 
        reverse_order: bool, optional
            - False: self * mat
            - True:  mat * self
        
        返回值：
        result : Matrix_3x3 | array_like
            如果 mat 是矩阵，则返回 Matrix_3x3（如果 in_place 为 True，则返回 self）；
            如果 mat 是向量，则返回 3D 向量
        """
        # 如果 mat 是 Matrix_3x3 对象，则获取其矩阵；否则将其转换为 numpy 数组
        mat = mat.m if type(mat) == Matrix_3x3 else np.asarray(mat)

        # 根据 reverse_order 参数确定乘法顺序
        a, b = (mat, self.m) if reverse_order else (self.m, mat)

        # 如果 mat 是向量，则执行矩阵与向量的乘法
        if mat.ndim == 1: 
            return np.matmul(a, b)  
        # 如果 in_place 为 True，则在原地执行矩阵乘法
        elif in_place:
            np.matmul(a, b, self.m) 
            return self
        # 如果 in_place 为 False，则返回一个新的 Matrix_3x3 对象
        else:                       
            return Matrix_3x3(np.matmul(a, b))
