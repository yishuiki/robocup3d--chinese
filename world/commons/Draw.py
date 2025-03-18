import socket
from math_ops.Math_Ops import Math_Ops as M
import numpy as np

class Draw():
    _socket = None  # 类变量，用于存储共享的 UDP 套接字

    def __init__(self, is_enabled: bool, unum: int, host: str, port: int) -> None:
        """
        初始化 Draw 类。
        :param is_enabled: 是否启用绘图功能。
        :param unum: 球衣号码。
        :param host: 服务器主机地址。
        :param port: 服务器端口号。
        """
        self.enabled = is_enabled  # 是否启用绘图
        self._is_team_right = None  # 球队是否在右侧
        self._unum = unum  # 球衣号码
        self._prefix = f'?{unum}_'.encode()  # 临时前缀，正常情况下不应使用

        # 创建一个共享的 UDP 套接字
        if Draw._socket is None:
            Draw._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            Draw._socket.connect((host, port))
            Draw.clear_all()  # 清除所有绘图

    def set_team_side(self, is_right):
        """
        设置球队在场地的哪一侧。
        :param is_right: 球队是否在右侧。
        """
        self._is_team_right = is_right
        # 根据球队位置生成合适的前缀
        self._prefix = f"{'r' if is_right else 'l'}{'_' if self._unum < 10 else '-'}{self._unum}_".encode()

    @staticmethod
    def _send(msg, id, flush):
        """
        私有方法，用于发送消息到 RoboViz。
        :param msg: 要发送的消息。
        :param id: 图形的 ID。
        :param flush: 是否刷新缓冲区。
        """
        try:
            if flush:
                Draw._socket.send(msg + id + b'\x00\x00\x00' + id + b'\x00')
            else:
                Draw._socket.send(msg + id + b'\x00')
        except ConnectionRefusedError:
            pass

    def circle(self, pos2d, radius, thickness, color: bytes, id: str, flush=True):
        """
        绘制圆形。
        :param pos2d: 圆心位置（2D）。
        :param radius: 半径。
        :param thickness: 线条厚度。
        :param color: 颜色（字节类型）。
        :param id: 图形的 ID。
        :param flush: 是否刷新缓冲区。
        """
        if not self.enabled: return
        assert type(color) == bytes, "颜色必须是字节类型，例如红色：b'\xFF\x00\x00'"
        assert not np.isnan(pos2d).any(), "位置参数包含 'nan' 值"

        if self._is_team_right:
            pos2d = (-pos2d[0], -pos2d[1])  # 如果球队在右侧，反转坐标

        msg = b'\x01\x00' + (
            f'{f"{pos2d[0]:.4f}":.6s}'
            f'{f"{pos2d[1]:.4f}":.6s}'
            f'{f"{radius:.4f}":.6s}'
            f'{f"{thickness:.4f}":.6s}').encode() + color

        Draw._send(msg, self._prefix + id.encode(), flush)

    def line(self, p1, p2, thickness, color: bytes, id: str, flush=True):
        """
        绘制直线。
        :param p1: 起点位置。
        :param p2: 终点位置。
        :param thickness: 线条厚度。
        :param color: 颜色（字节类型）。
        :param id: 图形的 ID。
        :param flush: 是否刷新缓冲区。
        """
        if not self.enabled: return
        assert type(color) == bytes, "颜色必须是字节类型，例如红色：b'\xFF\x00\x00'"
        assert not np.isnan(p1).any(), "起点参数包含 'nan' 值"
        assert not np.isnan(p2).any(), "终点参数包含 'nan' 值"

        z1 = p1[2] if len(p1) == 3 else 0
        z2 = p2[2] if len(p2) == 3 else 0

        if self._is_team_right:
            p1 = (-p1[0], -p1[1], p1[2]) if len(p1) == 3 else (-p1[0], -p1[1])
            p2 = (-p2[0], -p2[1], p2[2]) if len(p2) == 3 else (-p2[0], -p2[1])

        msg = b'\x01\x01' + (
            f'{f"{p1[0]:.4f}":.6s}'
            f'{f"{p1[1]:.4f}":.6s}'
            f'{f"{z1:.4f}":.6s}'
            f'{f"{p2[0]:.4f}":.6s}'
            f'{f"{p2[1]:.4f}":.6s}'
            f'{f"{z2:.4f}":.6s}'
            f'{f"{thickness:.4f}":.6s}').encode() + color

        Draw._send(msg, self._prefix + id.encode(), flush)

    def point(self, pos, size, color: bytes, id: str, flush=True):
        """
        绘制点。
        :param pos: 点的位置。
        :param size: 点的大小。
        :param color: 颜色（字节类型）。
        :param id: 图形的 ID。
        :param flush: 是否刷新缓冲区。
        """
        if not self.enabled: return
        assert type(color) == bytes, "颜色必须是字节类型，例如红色：b'\xFF\x00\x00'"
        assert not np.isnan(pos).any(), "位置参数包含 'nan' 值"

        z = pos[2] if len(pos) == 3 else 0

        if self._is_team_right:
            pos = (-pos[0], -pos[1], pos[2]) if len(pos) == 3 else (-pos[0], -pos[1])

        msg = b'\x01\x02' + (
            f'{f"{pos[0]:.4f}":.6s}'
            f'{f"{pos[1]:.4f}":.6s}'
            f'{f"{z:.4f}":.6s}'
            f'{f"{size:.4f}":.6s}').encode() + color

        Draw._send(msg, self._prefix + id.encode(), flush)

    def sphere(self, pos, radius, color: bytes, id: str, flush=True):
        """
        绘制球体。
        :param pos: 球体中心位置。
        :param radius: 半径。
        :param color: 颜色（字节类型）。
        :param id: 图形的 ID。
        :param flush: 是否刷新缓冲区。
        """
        if not self.enabled: return
        assert type(color) == bytes, "颜色必须是字节类型，例如红色：b'\xFF\x00\x00'"
        assert not np.isnan(pos).any(), "位置参数包含 'nan' 值"

        z = pos[2] if len(pos) == 3 else 0

        if self._is_team_right:
            pos = (-pos[0], -pos[1], pos[2]) if len(pos) == 3 else (-pos[0], -pos[1])

        msg = b'\x01\x03' + (
            f'{f"{pos[0]:.4f}":.6s}'
            f'{f"{pos[1]:.4f}":.6s}'
            f'{f"{z:.4f}":.6s}'
            f'{f"{radius:.4f}":.6s}').encode() + color

        Draw._send(msg, self._prefix + id.encode(), flush)

    def polygon(self, vertices, color: bytes, alpha: int, id: str, flush=True):
        """
        绘制多边形。
        :param vertices: 顶点列表。
        :param color: 颜色（字节类型）。
        :param alpha: 透明度（0-255）。
        :param id: 图形的 ID。
        :param flush: 是否刷新缓冲区。
        """
        if not self.enabled: return
        assert type(color) == bytes, "颜色必须是字节类型，例如红色：b'\xFF\x00\x00'"
        assert 0 <= alpha <= 255, "透明度必须在范围 [0,255] 内"

        if self._is_team_right:
            vertices = [(-v[0], -v[1], v[2]) for v in vertices]

        msg = b'\x01\x04' + bytes([len(vertices)]) + color + alpha.to_bytes(1, 'big')

        for v in vertices:
            msg += (
                f'{f"{v[0]:.4f}":.6s}'
                f'{f"{v[1]:.4f}":.6s}'
                f'{f"{v[2]:.4f}":.6s}').encode()

        Draw._send(msg, self._prefix + id.encode(), flush)

    def annotation(self, pos, text, color: bytes, id: str, flush=True):
        """
        绘制注释。
        :param pos: 注释位置。
        :param text: 注释文本。
        :param color: 颜色（字节类型）。
        :param id: 图形的 ID。
        :param flush: 是否刷新缓冲区。
        """
        if not self.enabled: return
        if type(text) != bytes: text = str(text).encode()
        assert type(color) == bytes, "颜色必须是字节类型，例如红色：b'\xFF\x00\x00'"
        z = pos[2] if len(pos) == 3 else 0

        if self._is_team_right:
            pos = (-pos[0], -pos[1], pos[2]) if len(pos) == 3 else (-pos[0], -pos[1])

        msg = b'\x02\x00' + (
            f'{f"{pos[0]:.4f}":.6s}'
            f'{f"{pos[1]:.4f}":.6s}'
            f'{f"{z:.4f}":.6s}').encode() + color + text + b'\x00'

        Draw._send(msg, self._prefix + id.encode(), flush)

    def arrow(self, p1, p2, arrowhead_size, thickness, color: bytes, id: str, flush=True):
        """
        绘制箭头。
        :param p1: 起点位置。
        :param p2: 终点位置。
        :param arrowhead_size: 箭头大小。
        :param thickness: 线条厚度。
        :param color: 颜色（字节类型）。
        :param id: 图形的 ID。
        :param flush: 是否刷新缓冲区。
        """
        if not self.enabled: return
        assert type(color) == bytes, "颜色必须是字节类型，例如红色：b'\xFF\x00\x00'"

        if len(p1) == 2: p1 = M.to_3d(p1)
        else: p1 = np.asarray(p1)
        if len(p2) == 2: p2 = M.to_3d(p2)
        else: p2 = np.asarray(p2)

        vec = p2 - p1
        vec_size = np.linalg.norm(vec)
        if vec_size == 0: return  # 如果向量长度为 0，则不绘制
        if arrowhead_size > vec_size: arrowhead_size = vec_size

        ground_proj_perpendicular = np.array([vec[1], -vec[0], 0])

        if np.all(ground_proj_perpendicular == 0):  # 垂直箭头
            ground_proj_perpendicular = np.array([arrowhead_size / 2, 0, 0])
        else:
            ground_proj_perpendicular *= arrowhead_size / 2 / np.linalg.norm(ground_proj_perpendicular)

        head_start = p2 - vec * (arrowhead_size / vec_size)
        head_pt1 = head_start + ground_proj_perpendicular
        head_pt2 = head_start - ground_proj_perpendicular

        self.line(p1, p2, thickness, color, id, False)
        self.line(p2, head_pt1, thickness, color, id, False)
        self.line(p2, head_pt2, thickness, color, id, flush)

    def flush(self, id):
        """
        刷新特定图形的缓冲区。
        :param id: 图形的 ID。
        """
        if not self.enabled: return

        Draw._send(b'\x00\x00', self._prefix + id.encode(), False)

    def clear(self, id):
        """
        清除特定图形。
        :param id: 图形的 ID。
        """
        if not self.enabled: return

        Draw._send(b'\x00\x00', self._prefix + id.encode(), True)  # 交换缓冲区两次

    def clear_player(self):
        """
        清除当前球员绘制的所有图形。
        """
        if not self.enabled: return

        Draw._send(b'\x00\x00', self._prefix, True)  # 交换缓冲区两次

    @staticmethod
    def clear_all():
        """
        清除所有球员绘制的所有图形。
        """
        if Draw._socket is not None:
            Draw._send(b'\x00\x00\x00\x00\x00', b'', False)  # 交换缓冲区两次，不使用 ID

    class Color:
        """
        基于 X11 颜色的预定义颜色列表。
        """
        pink_violet = b'\xC7\x15\x85'
        pink_hot = b'\xFF\x14\x93'
        pink_violet_pale = b'\xDB\x70\x93'
        pink = b'\xFF\x69\xB4'
        pink_pale = b'\xFF\xB6\xC1'

        red_dark = b'\x8B\x00\x00'
        red = b'\xFF\x00\x00'
        red_brick = b'\xB2\x22\x22'
        red_crimson = b'\xDC\x14\x3C'
        red_indian = b'\xCD\x5C\x5C'
        red_salmon = b'\xFA\x80\x72'

        orange_red = b'\xFF\x45\x00'
        orange = b'\xFF\x8C\x00'
        orange_ligth = b'\xFF\xA5\x00'

        yellow_gold = b'\xFF\xD7\x00'
        yellow = b'\xFF\xFF\x00'
        yellow_light = b'\xBD\xB7\x6B'

        brown_maroon = b'\x80\x00\x00'
        brown_dark = b'\x8B\x45\x13'
        brown = b'\xA0\x52\x2D'
        brown_gold = b'\xB8\x86\x0B'
        brown_light = b'\xCD\x85\x3F'
        brown_pale = b'\xDE\xB8\x87'

        green_dark = b'\x00\x64\x00'
        green = b'\x00\x80\x00'
        green_lime = b'\x32\xCD\x32'
        green_light = b'\x00\xFF\x00'
        green_lawn = b'\x7C\xFC\x00'
        green_pale = b'\x90\xEE\x90'

        cyan_dark = b'\x00\x80\x80'
        cyan_medium = b'\x00\xCE\xD1'
        cyan = b'\x00\xFF\xFF'
        cyan_light = b'\xAF\xEE\xEE'

        blue_dark = b'\x00\x00\x8B'
        blue = b'\x00\x00\xFF'
        blue_royal = b'\x41\x69\xE1'
        blue_medium = b'\x1E\x90\xFF'
        blue_light = b'\x00\xBF\xFF'
        blue_pale = b'\x87\xCE\xEB'

        purple_violet = b'\x94\x00\xD3'
        purple_magenta = b'\xFF\x00\xFF'
        purple_light = b'\xBA\x55\xD3'
        purple_pale = b'\xDD\xA0\xDD'

        white = b'\xFF\xFF\xFF'
        gray_10 = b'\xE6\xE6\xE6'
        gray_20 = b'\xCC\xCC\xCC'
        gray_30 = b'\xB2\xB2\xB2'
        gray_40 = b'\x99\x99\x99'
        gray_50 = b'\x80\x80\x80'
        gray_60 = b'\x66\x66\x66'
        gray_70 = b'\x4C\x4C\x4C'
        gray_80 = b'\x33\x33\x33'
        gray_90 = b'\x1A\x1A\x1A'
        black = b'\x00\x00\x00'

        @staticmethod
        def get(r, g, b):
            """
            获取自定义 RGB 颜色。
            :param r: 红色分量（0-255）。
            :param g: 绿色分量（0-255）。
            :param b: 蓝色分量（0-255）。
            :return: RGB 颜色的字节表示。
            """
            return bytes([int(r), int(g), int(b)])

