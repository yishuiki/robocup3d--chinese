from math import acos, asin, atan2, cos, pi, sin, sqrt
import numpy as np
import sys

try:
    GLOBAL_DIR = sys._MEIPASS  # 临时文件夹，包含库和数据文件
except:
    GLOBAL_DIR = "."


class Math_Ops():
    '''
    这个类提供了一些 numpy 没有直接提供的通用数学操作
    '''
  
    @staticmethod
    def deg_sph2cart(spherical_vec):
        ''' 将 SimSpark 的球坐标（以度为单位）转换为笛卡尔坐标 '''
        r = spherical_vec[0]
        h = spherical_vec[1] * pi / 180
        v = spherical_vec[2] * pi / 180
        return np.array([r * cos(v) * cos(h), r * cos(v) * sin(h), r * sin(v)])

    @staticmethod
    def deg_sin(deg_angle):
        ''' 返回度数的正弦值 '''
        return sin(deg_angle * pi / 180)

    @staticmethod
    def deg_cos(deg_angle):
        ''' 返回度数的余弦值 '''
        return cos(deg_angle * pi / 180)

    @staticmethod
    def to_3d(vec_2d, value=0) -> np.ndarray:
        ''' 从二维向量生成新的三维向量 '''
        return np.append(vec_2d,value)

    @staticmethod
    def to_2d_as_3d(vec_3d) -> np.ndarray:
        ''' 返回新的三维向量，其中第三维为零 '''
        vec_2d_as_3d = np.copy(vec_3d)
        vec_2d_as_3d[2] = 0
        return vec_2d_as_3d

    @staticmethod
    def normalize_vec(vec) -> np.ndarray:
        ''' 将向量除以其长度进行归一化 '''
        size = np.linalg.norm(vec)
        if size == 0: return vec
        return vec / size

    @staticmethod
    def get_active_directory(dir:str) -> str:
        global GLOBAL_DIR
        return GLOBAL_DIR + dir

    @staticmethod
    def acos(val):
        ''' 限制输入的反余弦函数 '''
        return acos( np.clip(val,-1,1) )
    
    @staticmethod
    def asin(val):
        ''' 限制输入的反正弦函数 '''
        return asin( np.clip(val,-1,1) )

    @staticmethod
    def normalize_deg(val):
        ''' 将角度值归一化到范围 [-180,180) '''
        return (val + 180.0) % 360 - 180

    @staticmethod
    def normalize_rad(val):
        ''' 将弧度值归一化到范围 [-pi,pi) '''
        return (val + pi) % (2*pi) - pi

    @staticmethod
    def deg_to_rad(val):
        ''' 将度数转换为弧度 '''
        return val * 0.01745329251994330

    @staticmethod
    def rad_to_deg(val):
        ''' 将弧度转换为度数 '''
        return val * 57.29577951308232

    @staticmethod
    def vector_angle(vector, is_rad=False):
        ''' 返回二维向量的角度（以度或弧度为单位） '''
        if is_rad:
            return atan2(vector[1], vector[0])
        else:
            return atan2(vector[1], vector[0]) * 180 / pi

    @staticmethod
    def vectors_angle(vec1, vec2, is_rad=False):
        ''' 返回两个向量之间的夹角（以度或弧度为单位） '''
        ang_rad = acos(np.dot(Math_Ops.normalize_vec(vec1),Math_Ops.normalize_vec(vec2)))
        return ang_rad if is_rad else ang_rad * 180 / pi

    @staticmethod
    def vector_from_angle(angle, is_rad=False):
        ''' 返回方向由 `angle` 给出的单位向量 '''
        if is_rad:
            return np.array([cos(angle), sin(angle)], float)
        else:
            return np.array([Math_Ops.deg_cos(angle), Math_Ops.deg_sin(angle)], float)

    @staticmethod
    def target_abs_angle(pos2d, target, is_rad=False):
        ''' 返回向量 (target-pos2d) 的绝对角度（以度或弧度为单位） '''
        if is_rad:
            return atan2(target[1]-pos2d[1], target[0]-pos2d[0])
        else:
            return atan2(target[1]-pos2d[1], target[0]-pos2d[0]) * 180 / pi

    @staticmethod
    def target_rel_angle(pos2d, ori, target, is_rad=False):
        ''' 返回目标的相对角度（以度或弧度为单位），假设我们位于 `pos2d`，方向为 `ori`（以度或弧度为单位） '''
        if is_rad:
            return Math_Ops.normalize_rad( atan2(target[1]-pos2d[1], target[0]-pos2d[0]) - ori )
        else:
            return Math_Ops.normalize_deg( atan2(target[1]-pos2d[1], target[0]-pos2d[0]) * 180 / pi - ori )

    @staticmethod
    def rotate_2d_vec(vec, angle, is_rad=False):
        ''' 将二维向量绕原点逆时针旋转 `angle` '''
        cos_ang = cos(angle) if is_rad else cos(angle * pi / 180)
        sin_ang = sin(angle) if is_rad else sin(angle * pi / 180)
        return np.array([cos_ang*vec[0]-sin_ang*vec[1], sin_ang*vec[0]+cos_ang*vec[1]])

    @staticmethod
    def distance_point_to_line(p:np.ndarray, a:np.ndarray, b:np.ndarray):
        ''' 
        计算点 p 到二维直线 'ab' 的距离（以及点 p 所在的侧边）

        参数：
        a : ndarray
            定义直线的二维点
        b : ndarray
            定义直线的二维点
        p : ndarray
            二维点

        返回值：
        distance : float
            直线与点之间的距离
        side : str
            如果我们位于 a 点，面向 b 点，点 p 可能位于我们的 "左侧" 或 "右侧"
        '''
        line_len = np.linalg.norm(b-a)

        if line_len == 0:  # 假设为垂直线
            dist = sdist = np.linalg.norm(p-a)
        else:
            sdist = np.cross(b-a,p-a)/line_len
            dist = abs(sdist)

        return dist, "left" if sdist>0 else "right"

    @staticmethod
    def distance_point_to_segment(p:np.ndarray, a:np.ndarray, b:np.ndarray):
        ''' 计算点 p 到二维线段 'ab' 的距离 '''
        
        ap = p-a
        ab = b-a

        ad = Math_Ops.vector_projection(ap,ab)

        # 判断 d 是否在 ab 上？我们可以通过 (ad = k * ab) 计算 k，而无需计算任何范数
        # 我们使用 ab 的最大维度来避免除以零
        k = ad[0]/ab[0] if abs(ab[0])>abs(ab[1]) else ad[1]/ab[1]

        if   k <= 0: return np.linalg.norm(ap)
        elif k >= 1: return np.linalg.norm(p-b)
        else:        return np.linalg.norm(p-(ad + a))  # p-d

    @staticmethod
    def distance_point_to_ray(p:np.ndarray, ray_start:np.ndarray, ray_direction:np.ndarray):
        ''' 计算点 p 到二维射线的距离 '''
        
        rp = p-ray_start
        rd = Math_Ops.vector_projection(rp,ray_direction)

        # 判断 d 是否在射线上？我们可以通过 (rd = k * ray_direction) 计算 k，而无需计算任何范数
        # 我们使用 ray_direction 的最大维度来避免除以零
        k = rd[0]/ray_direction[0] if abs(ray_direction[0])>abs(ray_direction[1]) else rd[1]/ray_direction[1]

        if   k <= 0: return np.linalg.norm(rp)
        else:        return np.linalg.norm(p-(rd + ray_start))  # p-d

    @staticmethod
    def closest_point_on_ray_to_point(p:np.ndarray, ray_start:np.ndarray, ray_direction:np.ndarray):
        ''' 返回射线上最接近点 p 的点 '''
        
        rp = p-ray_start
        rd = Math_Ops.vector_projection(rp,ray_direction)

        # 判断 d 是否在射线上？我们可以通过 (rd = k * ray_direction) 计算 k，而无需计算任何范数
        # 我们使用 ray_direction 的最大维度来避免除以零
        k = rd[0] / ray_direction[0] if abs(ray_direction[0]) > abs(ray_direction[1]) else rd[1] / ray_direction[1]

        if k <= 0: 
            return ray_start  # 如果 k <= 0，最接近的点是射线的起点
        else:       
            return rd + ray_start  # 否则，最接近的点是投影点加上射线起点

    @staticmethod
    def does_circle_intersect_segment(p:np.ndarray, r, a:np.ndarray, b:np.ndarray):
        ''' 判断圆（中心 p，半径 r）是否与二维线段相交 '''

        ap = p - a  # 从 a 到圆心 p 的向量
        ab = b - a  # 线段 ab 的方向向量

        ad = Math_Ops.vector_projection(ap, ab)  # 将向量 ap 投影到 ab 上

        # 判断投影点 d 是否在 ab 上？我们可以通过 (ad = k * ab) 计算 k，而无需计算任何范数
        # 我们使用 ab 的最大维度来避免除以零
        k = ad[0] / ab[0] if abs(ab[0]) > abs(ab[1]) else ad[1] / ab[1]

        if k <= 0:  # 如果投影点在 a 的左侧
            return np.dot(ap, ap) <= r * r  # 判断 ap 的长度是否小于等于半径
        elif k >= 1:  # 如果投影点在 b 的右侧
            return np.dot(p - b, p - b) <= r * r  # 判断 pb 的长度是否小于等于半径
        else:        
            dp = p - (ad + a)  # 计算从投影点到圆心的距离
            return np.dot(dp, dp) <= r * r  # 判断 dp 的长度是否小于等于半径

    @staticmethod
    def vector_projection(a:np.ndarray, b:np.ndarray):
        ''' 向量 a 在向量 b 上的投影 '''
        b_dot = np.dot(b, b)  # 计算 b 的点积
        return b * np.dot(a, b) / b_dot if b_dot != 0 else b  # 如果 b 不为零向量，计算投影

    @staticmethod
    def do_noncollinear_segments_intersect(a, b, c, d):
        ''' 判断两条非共线的二维线段是否相交
        解释：https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
        '''

        # 使用行列式判断点的相对位置
        ccw = lambda a, b, c: (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

    @staticmethod
    def intersection_segment_opp_goal(a:np.ndarray, b:np.ndarray):
        ''' 计算二维线段 'ab' 与对方球门（前门线）的交点 '''
        vec_x = b[0] - a[0]  # 线段的 x 分量

        # 如果线段与 x 轴平行，则没有交点
        if vec_x == 0: 
            return None
        
        k = (15.01 - a[0]) / vec_x  # 计算交点的参数

        # 如果交点不在线段上，则没有碰撞
        if k < 0 or k > 1: 
            return None

        intersection_pt = a + (b - a) * k  # 计算交点

        # 判断交点是否在球门范围内
        if -1.01 <= intersection_pt[1] <= 1.01:
            return intersection_pt
        else:
            return None

    @staticmethod
    def intersection_circle_opp_goal(p:np.ndarray, r):
        ''' 
        计算圆（中心 p，半径 r）与对方球门（前门线）的交点
        只返回 y 坐标，因为 x 坐标始终为 15
        '''

        x_dev = abs(15 - p[0])  # 圆心到球门的 x 偏移量

        # 如果圆心到球门的距离大于半径，则没有交点
        if x_dev > r:
            return None  

        y_dev = sqrt(r * r - x_dev * x_dev)  # 计算 y 偏移量

        p1 = max(p[1] - y_dev, -1.01)  # 下交点
        p2 = min(p[1] + y_dev, 1.01)   # 上交点

        # 判断交点数量
        if p1 == p2:
            return p1  # 返回单个交点的 y 坐标
        elif p2 < p1:
            return None  # 没有交点
        else:
            return p1, p2  # 返回两个交点的 y 坐标

    @staticmethod
    def distance_point_to_opp_goal(p:np.ndarray):
        ''' 计算点 'p' 到对方球门（前门线）的距离 '''

        if p[1] < -1.01:  # 如果点在球门下方
            return np.linalg.norm(p - (15, -1.01))  # 计算到球门下边界的距离
        elif p[1] > 1.01:  # 如果点在球门上方
            return np.linalg.norm(p - (15, 1.01))  # 计算到球门上边界的距离
        else:  # 如果点在球门范围内
            return abs(15 - p[0])  # 计算到球门的 x 距离

    @staticmethod
    def circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9):
        """ 计算圆与线段的交点。可能有 0、1 或 2 个交点。

        :param circle_center: 圆心的 (x, y) 坐标
        :param circle_radius: 圆的半径
        :param pt1: 线段起点的 (x, y) 坐标
        :param pt2: 线段终点的 (x, y) 坐标
        :param full_line: 如果为 True，则考虑整条直线；如果为 False，则只考虑线段
        :param tangent_tol: 判断切线的容差
        :return: 交点列表，每个交点是一个 (x, y) 坐标
        """
        (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
        (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
        dx, dy = (x2 - x1), (y2 - y1)
        dr = (dx ** 2 + dy ** 2)**.5
        big_d = x1 * y2 - x2 * y1
        discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

        if discriminant < 0:  # 圆与直线无交点
            return []
        else:  # 圆与直线可能有 0、1 或 2 个交点
            intersections = [
                (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
                cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
                for sign in ((1, -1) if dy < 0 else (-1, 1))]  # 确保交点顺序正确
            if not full_line:  # 如果只考虑线段
                fraction_along_segment = [
                    (xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
                intersections = [pt for pt, frac in zip(
                    intersections, fraction_along_segment) if 0 <= frac<= 1]  # 筛选出位于线段内的交点
            # 如果直线与圆相切，返回一个交点（两个交点重合）
            if len(intersections) == 2 and abs(discriminant) <= tangent_tol:
                return [intersections[0]]
            else:
                return intersections

    @staticmethod
    def get_line_intersection(a1, a2, b1, b2):
        """ 
        计算两条直线的交点，直线分别通过点 a1 和 a2，以及点 b1 和 b2。
        a1: [x, y] 第一条直线上的一个点
        a2: [x, y] 第一条直线上的另一个点
        b1: [x, y] 第二条直线上的一个点
        b2: [x, y] 第二条直线上的另一个点
        """
        s = np.vstack([a1, a2, b1, b2])  # 将所有点堆叠起来
        h = np.hstack((s, np.ones((4, 1))))  # 添加齐次坐标
        l1 = np.cross(h[0], h[1])  # 计算第一条直线
        l2 = np.cross(h[2], h[3])  # 计算第二条直线
        x, y, z = np.cross(l1, l2)  # 计算交点
        if z == 0:  # 如果 z 为零，说明直线平行
            return np.array([float('inf'), float('inf')])
        return np.array([x / z, y / z], float)  # 返回交点坐标

