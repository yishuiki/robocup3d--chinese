from math import asin, atan2, pi, sqrt
from math_ops.Math_Ops import Math_Ops as M
from math_ops.Matrix_3x3 import Matrix_3x3
import numpy as np
class Matrix_4x4():
    """
    4x4 变换矩阵类，用于表示刚体的平移和旋转。
    提供了多种构造方法、旋转操作、平移操作和矩阵运算。
    """

    def __init__(self, matrix=None) -> None:
        """
        构造函数示例：
        a = Matrix_4x4()                                           # 创建单位矩阵
        b = Matrix_4x4([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]]) # 手动初始化矩阵
        c = Matrix_4x4([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4])         # 手动初始化矩阵
        d = Matrix_4x4(b)                                          # 复制构造函数
        """
        if matrix is None:
            self.m = np.identity(4)  # 创建单位矩阵
        elif type(matrix) == Matrix_4x4: 
            self.m = np.copy(matrix.m)  # 复制另一个 Matrix_4x4 对象的矩阵
        elif type(matrix) == Matrix_3x3: 
            self.m = np.identity(4)  # 创建单位矩阵
            self.m[0:3,0:3] = matrix.m  # 将 3x3 矩阵嵌入到 4x4 矩阵中
        else:
            self.m = np.asarray(matrix)  # 将输入转换为 numpy 数组
            self.m.shape = (4,4)  # 确保矩阵形状为 4x4，如果需要的话进行重塑，否则抛出错误
    @classmethod
    def from_translation(cls, translation_vec):
        """
        从平移向量创建变换矩阵
        e.g. Matrix_4x4.from_translation((a,b,c))
            output: [[1,0,0,a],[0,1,0,b],[0,0,1,c],[0,0,0,1]]
        """
        mat = np.identity(4)
        mat[0:3,3] = translation_vec
        return cls(mat)

    @classmethod
    def from_3x3_and_translation(cls, mat3x3:Matrix_3x3, translation_vec):
        """
        从旋转矩阵（3x3）和平移向量创建变换矩阵
        e.g. Matrix_4x4.from_3x3_and_translation(r,(a,b,c))    
            output: [[r00,r01,r02,a],[r10,r11,r12,b],[r20,r21,r22,c],[0,0,0,1]]
        """
        mat = np.identity(4)
        mat[0:3,0:3] = mat3x3.m
        mat[0:3,3] = translation_vec
        return cls(mat)
    def translate(self, translation_vec, in_place=False):
        """
        平移当前变换矩阵

        参数：
        translation_vec : array_like, length 3
            平移向量
        in_place: bool, optional
            * True: 内部矩阵在原地更改
            * False: 返回一个新的矩阵，当前矩阵保持不变 

        返回值：
        result : Matrix_4x4 
            如果 in_place 为 True，则返回 self
        """
        vec = np.array([*translation_vec,1])  # 转换为 4D 向量
        np.matmul(self.m, vec, out=vec)       # 只计算第 4 列

        if in_place:
            self.m[:,3] = vec
            return self
        else:
            ret = Matrix_4x4(self.m)
            ret.m[:,3] = vec
            return ret
    def get_translation(self):
        ''' 获取平移向量 (x,y,z) '''
        return self.m[0:3,3]  # 返回视图

    def get_x(self):
        return self.m[0,3]

    def get_y(self):
        return self.m[1,3]

    def get_z(self):
        return self.m[2,3]

    def get_rotation_4x4(self):
        ''' 获取无平移的 4x4 旋转矩阵 ''' 
        mat = Matrix_4x4(self)
        mat.m[0:3,3] = 0
        return mat

    def get_rotation(self):
        ''' 获取旋转的 3x3 矩阵 '''
        return Matrix_3x3(self.m[0:3,0:3])

    def get_distance(self):
        ''' 获取平移向量的长度 '''
        return np.linalg.norm(self.m[0:3,3])
    def get_roll_deg(self):
        ''' 获取绕 x 轴的角度（以度为单位），旋转顺序：RotZ*RotY*RotX=Rot '''
        if self.m[2,1] == 0 and self.m[2,2] == 0: 
            return 180
        return atan2(self.m[2,1], self.m[2,2]) * 180 / pi

    def get_pitch_deg(self):
        ''' 获取绕 y 轴的角度（以度为单位），旋转顺序：RotZ*RotY*RotX=Rot '''
        return atan2(-self.m[2,0], sqrt(self.m[2,1]*self.m[2,1] + self.m[2,2]*self.m[2,2])) * 180 / pi

    def get_yaw_deg(self):
        ''' 获取绕 z 轴的角度（以度为单位），旋转顺序：RotZ*RotY*RotX=Rot '''
        if self.m[1,0] == 0 and self.m[0,0] == 0: 
            return atan2(self.m[0,1], self.m[1,1]) * 180 / pi
        return atan2(self.m[1,0], self.m[0,0]) * 180 / pi

    def get_inclination_deg(self):
        ''' 获取 z 轴相对于参考 z 轴的倾斜角度 '''
        return 90 - (asin(np.clip(self.m[2,2],-1,1)) * 180 / pi)
    def rotate_deg(self, rotation_vec, rotation_deg, in_place=False):
        '''
        旋转当前变换矩阵

        参数：
        rotation_vec : array_like, length 3
            旋转向量
        rotation_deg : float
            旋转角度（以度为单位）
        in_place: bool, optional
            * True: 内部矩阵在原地更改（默认）
            * False: 返回一个新的矩阵，当前矩阵保持不变 
        
        返回值：
        result : Matrix_4x4 
            如果 in_place 为 True，则返回 self
        '''
        return self.rotate_rad(rotation_vec, rotation_deg * (pi/180) , in_place)

    def rotate_rad(self, rotation_vec, rotation_rad, in_place=False):
        '''
        旋转当前变换矩阵

        参数：
        rotation_vec : array_like, length 3
            旋转向量
        rotation_rad : float
            旋转角度（以弧度为单位）
        in_place: bool, optional
            * True: 内部矩阵在原地更改（默认）
            * False: 返回一个新的矩阵，当前矩阵保持不变 
        
        返回值：
        result : Matrix_4x4 
            如果 in_place 为 True，则返回 self
        '''
        if rotation_rad == 0: 
            return self if in_place else Matrix_4x4(self)

        # 简化旋转操作
        if rotation_vec[0] == 0:
            if rotation_vec[1] == 0:
                if rotation_vec[2] == 1:
                    return self.rotate_z_rad(rotation_rad, in_place)
                elif rotation_vec[2] == -1:
                    return self.rotate_z_rad(-rotation_rad, in_place)
            elif rotation_vec[2] == 0:
                if rotation_vec[1] == 1:
                    return self.rotate_y_rad(rotation_rad, in_place)
                elif rotation_vec[1] == -1:
                    return self.rotate_y_rad(-rotation_rad, in_place)
        elif rotation_vec[1] == 0 and rotation_vec[2] == 0:
            if rotation_vec[0] == 1:
                return self.rotate_x_rad(rotation_rad, in_place)
            elif rotation_vec[0] == -1:
                return self.rotate_x_rad(-rotation_rad, in_place)

        # 如果不是简单的单轴旋转，则使用通用旋转矩阵公式
        c = np.cos(rotation_rad)  # 旋转角度的余弦值
        c1 = 1 - c  # 用于计算旋转矩阵的中间值
        s = np.sin(rotation_rad)  # 旋转角度的正弦值
        x = rotation_vec[0]  # 旋转向量的 x 分量
        y = rotation_vec[1]  # 旋转向量的 y 分量
        z = rotation_vec[2]  # 旋转向量的 z 分量
        xxc1 = x * x * c1  # 旋转矩阵的中间值
        yyc1 = y * y * c1
        zzc1 = z * z * c1
        xyc1 = x * y * c1
        xzc1 = x * z * c1
        yzc1 = y * z * c1
        xs = x * s  # 旋转矩阵的中间值
        ys = y * s
        zs = z * s

        # 构造旋转矩阵
        mat = np.array([
            [xxc1 + c, xyc1 - zs, xzc1 + ys, 0],
            [xyc1 + zs, yyc1 + c, yzc1 - xs, 0],
            [xzc1 - ys, yzc1 + xs, zzc1 + c, 0],
            [0, 0, 0, 1]
        ])

        # 将旋转矩阵与当前矩阵相乘
        return self.multiply(mat, in_place)

    def rotate_x_rad(self, rotation_rad, in_place=False):
        """
        绕 x 轴旋转当前变换矩阵

        参数：
        rotation_rad : float
            旋转角度（以弧度为单位）
        in_place: bool, optional
            * True: 内部矩阵在原地更改（默认）
            * False: 返回一个新的矩阵，当前矩阵保持不变
        
        返回值：
        result : Matrix_4x4 
            如果 in_place 为 True，则返回 self
        """
        if rotation_rad == 0: 
            return self if in_place else Matrix_4x4(self)
 
        c = np.cos(rotation_rad)  # 旋转角度的余弦值
        s = np.sin(rotation_rad)  # 旋转角度的正弦值

        # 构造绕 x 轴的旋转矩阵
        mat = np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ])

        # 将旋转矩阵与当前矩阵相乘
        return self.multiply(mat, in_place)

    def rotate_y_rad(self, rotation_rad, in_place=False):
        """
        绕 y 轴旋转当前变换矩阵

        参数：
        rotation_rad : float
            旋转角度（以弧度为单位）
        in_place: bool, optional
            * True: 内部矩阵在原地更改（默认）
            * False: 返回一个新的矩阵，当前矩阵保持不变
        
        返回值：
        result : Matrix_4x4 
            如果 in_place 为 True，则返回 self
        """
        if rotation_rad == 0: 
            return self if in_place else Matrix_4x4(self)
 
        c = np.cos(rotation_rad)  # 旋转角度的余弦值
        s = np.sin(rotation_rad)  # 旋转角度的正弦值

        # 构造绕 y 轴的旋转矩阵
        mat = np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ])

        # 将旋转矩阵与当前矩阵相乘
        return self.multiply(mat, in_place)

    def rotate_z_rad(self, rotation_rad, in_place=False):
        """
        绕 z 轴旋转当前变换矩阵

        参数：
        rotation_rad : float
            旋转角度（以弧度为单位）
        in_place: bool, optional
            * True: 内部矩阵在原地更改（默认）
            * False: 返回一个新的矩阵，当前矩阵保持不变
        
        返回值：
        result : Matrix_4x4 
            如果 in_place 为 True，则返回 self
        """
        if rotation_rad == 0: 
            return self if in_place else Matrix_4x4(self)
 
        c = np.cos(rotation_rad)  # 旋转角度的余弦值
        s = np.sin(rotation_rad)  # 旋转角度的正弦值

        # 构造绕 z 轴的旋转矩阵
        mat = np.array([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # 将旋转矩阵与当前矩阵相乘
        return self.multiply(mat, in_place)

    def rotate_x_deg(self, rotation_deg, in_place=False):
        """
        绕 x 轴旋转当前变换矩阵

        参数：
        rotation_deg : float
            旋转角度（以度为单位）
        in_place: bool, optional
            * True: 内部矩阵在原地更改（默认）
            * False: 返回一个新的矩阵，当前矩阵保持不变
        
        返回值：
        result : Matrix_4x4 
            如果 in_place 为 True，则返回 self
        """
        return self.rotate_x_rad(rotation_deg * (pi/180), in_place)

    def rotate_y_deg(self, rotation_deg, in_place=False):
        """
        绕 y 轴旋转当前变换矩阵

        参数：
        rotation_deg : float
            旋转角度（以度为单位）
        in_place: bool, optional
            * True: 内部矩阵在原地更改（默认）
            * False: 返回一个新的矩阵，当前矩阵保持不变
        
        返回值：
        result : Matrix_4x4 
            如果 in_place 为 True，则返回 self
        """
        return self.rotate_y_rad(rotation_deg * (pi/180), in_place)

    def rotate_z_deg(self, rotation_deg, in_place=False):
        """
        绕 z 轴旋转当前变换矩阵

        参数：
        rotation_deg : float
            旋转角度（以度为单位）
        in_place: bool, optional
            * True: 内部矩阵在原地更改（默认）
            * False: 返回一个新的矩阵，当前矩阵保持不变
        
        返回值：
        result : Matrix_4x4 
            如果 in_place 为 True，则返回 self
        """
        return self.rotate_z_rad(rotation_deg * (pi/180), in_place)
    def invert(self, in_place=False):
        """
        反转当前变换矩阵

        参数：
        in_place: bool, optional
            * True: 内部矩阵在原地更改（默认）
            * False: 返回一个新的矩阵，当前矩阵保持不变
        
        返回值：
        result : Matrix_4x4 
            如果 in_place 为 True，则返回 self
        """
        if in_place:
            self.m = np.linalg.inv(self.m)
            return self
        else:
            return Matrix_4x4(np.linalg.inv(self.m))

    def multiply(self, mat, in_place=False):
        """
        将当前变换矩阵与 mat 相乘

        参数：
        mat : Matrix_4x4 或 array_like
            乘数矩阵或 3D 向量
        in_place: bool, optional
            * True: 内部矩阵在原地更改（默认）
            * False: 返回一个新的矩阵，当前矩阵保持不变（如果 mat 是 4x4 矩阵）
        
        返回值：
        result : Matrix_4x4 | array_like
            如果 mat 是矩阵，则返回 Matrix_4x4（如果 in_place 为 True，则返回 self）；
            如果 mat 是向量，则返回 3D 向量
        """
        if type(mat) == Matrix_4x4:  
            mat = mat.m
        else:
            mat = np.asarray(mat)  # 如果需要的话，将输入转换为 numpy 数组
            if mat.ndim == 1:  # 如果 mat 是向量，则执行矩阵与向量的乘法
                vec = np.append(mat, 1)  # 转换为 4D 向量
                return np.matmul(self.m, vec)[0:3]  # 转换回 3D 向量

        if in_place:
            np.matmul(self.m, mat, self.m)  # 在原地执行矩阵乘法
            return self
        else:
            return Matrix_4x4(np.matmul(self.m, mat))  # 返回一个新的 Matrix_4x4 对象

    def __call__(self, mat, is_spherical=False):
        """
        将当前变换矩阵与 mat 相乘，并返回一个新的矩阵或向量。

        参数：
        mat : Matrix_4x4 或 array_like
            乘数矩阵或 3D 向量
        is_spherical : bool
            仅当 mat 是 3D 向量时相关，True 表示使用球坐标

        返回值：
        result : Matrix_4x4 | array_like
            如果 mat 是矩阵，则返回 Matrix_4x4；
            如果 mat 是向量，则返回 3D 向量
        """
        if is_spherical and mat.ndim == 1:  # 如果 mat 是球坐标向量
            mat = M.deg_sph2cart(mat)  # 将球坐标转换为笛卡尔坐标
        return self.multiply(mat, False)  # 调用 multiply 方法进行矩阵乘法
