from cpp.a_star import a_star
from math_ops.Math_Ops import Math_Ops as M
from world.World import World
import math
import numpy as np


class Path_Manager():
    """
    路径管理器类，用于机器人足球比赛中的路径规划。
    """
    MODE_CAUTIOUS = 0  # 谨慎模式
    MODE_DRIBBLE = 1   # 带球模式（安全距离增加）
    MODE_AGGRESSIVE = 2  # 激进模式（安全距离减少）

    STATUS_SUCCESS = 0  # 路径规划算法正常执行
    STATUS_TIMEOUT = 1  # 超时（目标未到达）
    STATUS_IMPOSSIBLE = 2  # 无法到达目标（所有选项已尝试）
    STATUS_DIRECT = 3  # 起点和目标之间无障碍（路径仅包含起点和目标）

    HOT_START_DIST_WALK = 0.05  # 行走时的热启动预测距离
    HOT_START_DIST_DRIBBLE = 0.10  # 带球时的热启动预测距离

    def __init__(self, world: World) -> None:
        """
        初始化路径管理器。
        :param world: 世界对象，包含机器人和球的信息。
        """
        self.world = world

        self._draw_obstacles = False  # 是否绘制障碍物
        self._draw_path = False  # 是否绘制路径
        self._use_team_channel = False  # 是否使用团队绘图通道

        # 内部变量，用于热启动路径（减少路径不稳定性）
        self.last_direction_rad = None
        self.last_update = 0
        self.last_start_dist = None
    def draw_options(self, enable_obstacles, enable_path, use_team_drawing_channel=False):
        """
        启用或禁用绘图，并选择绘图通道。
        :param enable_obstacles: 是否绘制障碍物。
        :param enable_path: 是否绘制路径。
        :param use_team_drawing_channel: 是否使用团队绘图通道。
        """
        self._draw_obstacles = enable_obstacles
        self._draw_path = enable_path
        self._use_team_channel = use_team_drawing_channel
    def get_obstacles(self, include_teammates, include_opponents, include_play_mode_restrictions, max_distance=4, max_age=500, 
                    ball_safety_margin=0, goalpost_safety_margin=0, mode=MODE_CAUTIOUS, priority_unums=[]):
        """
        获取路径规划中的障碍物。
        :param include_teammates: 是否包含队友。
        :param include_opponents: 是否包含对手。
        :param include_play_mode_restrictions: 是否包含比赛模式限制。
        :param max_distance: 考虑的最大距离（米）。
        :param max_age: 考虑的最大时间（毫秒）。
        :param ball_safety_margin: 球的安全距离。
        :param goalpost_safety_margin: 球门柱的安全距离。
        :param mode: 路径规划模式（谨慎、带球、激进）。
        :param priority_unums: 优先避免的队友编号。
        :return: 障碍物列表，每个障碍物是一个包含 5 个浮点数的元组 (x, y, 硬半径, 软半径, 排斥力)。
        """
        w = self.world

        ball_2d = w.ball_abs_pos[:2]
        obstacles = []

        # 检查队友和对手的可见性
        check_age = lambda last_update, comparator=w.time_local_ms - max_age: last_update > 0 and last_update >= comparator

        # 添加队友为障碍物
        if include_teammates:
            soft_radius = 1.1 if mode == Path_Manager.MODE_DRIBBLE else 0.6

            def get_hard_radius(t):
                return 1.0 if t.unum in priority_unums else t.state_ground_area[1] + 0.2

            obstacles.extend(
                (*t.state_ground_area[0],
                get_hard_radius(t),
                1.5 if t.unum in priority_unums else soft_radius,
                1.0)
                for t in w.teammates if not t.is_self and check_age(t.state_last_update) and t.state_horizontal_dist < max_distance
            )

        # 添加对手为障碍物
        if include_opponents:
            if mode == Path_Manager.MODE_AGGRESSIVE:
                soft_radius = 0.6
                hard_radius = lambda o: 0.2
            elif mode == Path_Manager.MODE_DRIBBLE:
                soft_radius = 2.3
                hard_radius = lambda o: o.state_ground_area[1] + 0.9
            else:
                soft_radius = 1.0
                hard_radius = lambda o: o.state_ground_area[1] + 0.2

            obstacles.extend(
                (*o.state_ground_area[0],
                hard_radius(o),
                soft_radius,
                1.5 if o.unum == 1 else 1.0)
                for o in w.opponents if o.state_last_update > 0 and w.time_local_ms - o.state_last_update <= max_age and o.state_horizontal_dist < max_distance
            )

        # 添加比赛模式限制
        if include_play_mode_restrictions:
            if w.play_mode == World.M_THEIR_GOAL_KICK:
                obstacles.extend((15, i, 2.1, 0, 0) for i in range(-2, 3))
            elif w.play_mode == World.M_THEIR_PASS:
                obstacles.append((*ball_2d, 1.2, 0, 0))
            elif w.play_mode in [World.M_THEIR_KICK_IN, World.M_THEIR_CORNER_KICK, World.M_THEIR_FREE_KICK, World.M_THEIR_DIR_FREE_KICK, World.M_THEIR_OFFSIDE]:
                obstacles.append((*ball_2d, 2.5, 0, 0))

        # 添加球为障碍物
        if ball_safety_margin > 0:
            if (w.play_mode_group != w.MG_OTHER) or abs(ball_2d[1]) > 9.5 or abs(ball_2d[0]) > 14.5:
                ball_safety_margin += 0.12
            obstacles.append((*ball_2d, 0, ball_safety_margin, 8))

        # 添加球门柱为障碍物
        if goalpost_safety_margin > 0:
            obstacles.append((14.75, 1.10, goalpost_safety_margin, 0, 0))
            obstacles.append((14.75, -1.10, goalpost_safety_margin, 0, 0))

        # 绘制障碍物
        if self._draw_obstacles:
            d = w.team_draw if self._use_team_channel else w.draw
            if d.enabled:
                for o in obstacles:
                    if o[3] > 0:
                        d.circle(o[:2], o[3], o[4] / 2, d.Color.orange, "path_obstacles", False)
                    if o[2] > 0:
                        d.circle(o[:2], o[2], 1, d.Color.red, "path_obstacles", False)
                d.flush("path_obstacles")

        return obstacles
    def _get_hot_start(self, start_distance):
        """
        获取路径的热启动位置（考虑上一次路径）。
        :param start_distance: 起始距离。
        :return: 热启动位置。
        """
        if self.last_update > 0 and self.world.time_local_ms - self.last_update == 20 and self.last_start_dist == start_distance:
            return self.world.robot.loc_head_position[:2] + M.vector_from_angle(self.last_direction_rad, is_rad=True) * start_distance
        else:
            return self.world.robot.loc_head_position[:2]
    def _update_hot_start(self, next_dir_rad, start_distance):
        """
        更新热启动位置。
        :param next_dir_rad: 下一个方向（弧度）。
        :param start_distance: 起始距离。
        """
        self.last_direction_rad = next_dir_rad
        self.last_update = self.world.time_local_ms
        self.last_start_dist = start_distance
    def _extract_target_from_path(self, path, path_len, ret_segments):
        """
        从路径中提取目标位置。
        :param path: 路径。
        :param path_len: 路径长度。
        :param ret_segments: 返回的目标段数。
        :return: 目标位置。
        """
        ret_seg_ceil = math.ceil(ret_segments)  # 向上取整

        if path_len >= ret_seg_ceil:
            i = ret_seg_ceil * 2  # 路径索引（x 坐标）
            if ret_seg_ceil == ret_segments:
                return path[i:i + 2]  # 直接返回目标位置
            else:
                floor_w = ret_seg_ceil - ret_segments  # 计算权重
                return path[i - 2:i] * floor_w + path[i:i + 2] * (1 - floor_w)  # 插值计算目标位置
        else:
            return path[-2:]  # 如果路径长度不足，返回路径的最后一个点
    def get_path_to_ball(self, x_ori=None, x_dev=-0.2, y_dev=0, torso_ori=None, torso_ori_thrsh=1,
                        priority_unums=[], is_aggressive=True, safety_margin=0.25, timeout=3000):
        """
        获取到球的路径的下一个目标位置（下一个绝对位置 + 下一个绝对朝向）。
        :param x_ori: 自定义参考系的 x 轴绝对朝向（如果为 None，则使用机器人到球的向量方向）。
        :param x_dev: 自定义参考系的 x 轴目标位置偏差。
        :param y_dev: 自定义参考系的 y 轴目标位置偏差。
        :param torso_ori: 躯干的目标绝对朝向（如果为 None，则使用机器人到目标的向量方向）。
        :param torso_ori_thrsh: 当机器人与最终目标的距离小于该阈值时，应用躯干朝向。
        :param priority_unums: 优先避免的队友编号。
        :param is_aggressive: 是否激进（减少对手的安全距离）。
        :param safety_margin: 球周围的排斥半径，以避免碰撞。
        :param timeout: 最大执行时间（微秒）。
        :return: 下一个位置、下一个朝向、到球的距离。
        """
        w = self.world
        r = w.robot
        dev = np.array([x_dev, y_dev])
        dev_len = np.linalg.norm(dev)
        dev_mult = 1

        # 如果机器人距离球超过 0.5 米且比赛进行中，则使用球的预测位置
        if np.linalg.norm(w.ball_abs_pos[:2] - r.loc_head_position[:2]) > 0.5 and w.play_mode_group == w.MG_OTHER:
            ball_2d = w.get_intersection_point_with_ball(0.4)[0]  # 以 0.4 米/秒的速度移动时的交点
        else:
            ball_2d = w.ball_abs_pos[:2]

        # 自定义参考系的朝向
        vec_me_ball = ball_2d - r.loc_head_position[:2]
        if x_ori is None:
            x_ori = M.vector_angle(vec_me_ball)

        distance_boost = 0  # 目标距离的增强值
        if torso_ori is not None and dev_len > 0:
            approach_ori_diff = abs(M.normalize_deg(r.imu_torso_orientation - torso_ori))
            if approach_ori_diff > 15:  # 如果机器人与目标朝向的差异较大，则增加移动速度
                distance_boost = 0.15
            if approach_ori_diff > 30:  # 如果差异更大，则增加目标距离
                dev_mult = 1.3
            if approach_ori_diff > 45:  # 如果差异非常大，则增加球周围的排斥半径
                safety_margin = max(0.32, safety_margin)

        # 获取目标位置
        front_unit_vec = M.vector_from_angle(x_ori)
        left_unit_vec = np.array([-front_unit_vec[1], front_unit_vec[0]])

        rel_target = front_unit_vec * dev[0] + left_unit_vec * dev[1]
        target = ball_2d + rel_target * dev_mult
        target_vec = target - r.loc_head_position[:2]
        target_dist = np.linalg.norm(target_vec)

        # 绘制目标位置
        if self._draw_path:
            d = self.world.team_draw if self._use_team_channel else self.world.draw
            d.point(target, 4, d.Color.red, "path_target")  # 如果绘图对象内部禁用，则不会绘制

        # 获取障碍物
        obstacles = self.get_obstacles(include_teammates=True, include_opponents=True, include_play_mode_restrictions=True,
                                    ball_safety_margin=safety_margin,
                                    mode=Path_Manager.MODE_AGGRESSIVE if is_aggressive else Path_Manager.MODE_CAUTIOUS,
                                    priority_unums=priority_unums)

        # 在球的对面添加障碍物
        if dev_len > 0 and safety_margin > 0:
            center = ball_2d - M.normalize_vec(rel_target) * safety_margin
            obstacles.append((*center, 0, safety_margin * 0.9, 5))
            if self._draw_obstacles:
                d = w.team_draw if self._use_team_channel else w.draw
                if d.enabled:
                    d.circle(center, safety_margin * 0.8, 2.5, d.Color.orange, "path_obstacles_1")

        # 获取路径
        start_pos = self._get_hot_start(Path_Manager.HOT_START_DIST_WALK) if target_dist > 0.4 else self.world.robot.loc_head_position[:2]
        path, path_len, path_status, path_cost = self.get_path(start_pos, True, obstacles, target, timeout)
        path_end = path[-2:]  # A* 允许的最后一个位置

        # 获取相关距离
        if w.ball_last_seen > w.time_local_ms - w.VISUALSTEP_MS:  # 球在视野范围内
            raw_ball_dist = np.linalg.norm(w.ball_rel_torso_cart_pos[:2])  # 躯干中心到球中心的距离
        else:  # 否则使用绝对坐标计算距离
            raw_ball_dist = np.linalg.norm(vec_me_ball)  # 头部中心到球中心的距离

        avoid_touching_ball = (w.play_mode_group != w.MG_OTHER)
        distance_to_final_target = np.linalg.norm(path_end - r.loc_head_position[:2])
        distance_to_ball = max(0.07 if avoid_touching_ball else 0.14, raw_ball_dist - 0.13)
        caution_dist = min(distance_to_ball, distance_to_final_target)

        # 获取下一个目标位置
        next_pos = self._extract_target_from_path(path, path_len, ret_segments=1 if caution_dist < 1 else 2)

        # 获取下一个目标朝向
        if torso_ori is not None:
            if caution_dist > torso_ori_thrsh:
                next_ori = M.vector_angle(target_vec)
            else:
                mid_ori = M.normalize_deg(M.vector_angle(vec_me_ball) - M.vector_angle(-dev) - x_ori + torso_ori)
                mid_ori_diff = abs(M.normalize_deg(mid_ori - r.imu_torso_orientation))
                final_ori_diff = abs(M.normalize_deg(torso_ori - r.imu_torso_orientation))
                next_ori = mid_ori if mid_ori_diff + 10 < final_ori_diff else torso_ori
        elif target_dist > 0.1:
            next_ori = M.vector_angle(target_vec)
        else:
            next_ori = r.imu_torso_orientation

        # 更新热启动位置
        if path_len != 0:
            next_pos_vec = next_pos - self.world.robot.loc_head_position[:2]
            next_pos_dist = np.linalg.norm(next_pos_vec)
            self._update_hot_start(M.vector_angle(next_pos_vec, is_rad=True), min(Path_Manager.HOT_START_DIST_WALK, next_pos_dist))

        return next_pos, next_ori, min(distance_to_ball, distance_to_final_target + distance_boost)
    def get_path_to_target(self, target, ret_segments=1.0, torso_ori=None, priority_unums=[], is_aggressive=True, timeout=3000):
        """
        获取到目标的路径的下一个位置（下一个绝对位置 + 下一个绝对朝向）。
        :param target: 目标位置。
        :param ret_segments: 返回的目标段数。
        :param torso_ori: 躯干的目标绝对朝向（如果为 None，则使用机器人到目标的向量方向）。
        :param priority_unums: 优先避免的队友编号。
        :param is_aggressive: 是否激进（减少对手的安全距离）。
        :param timeout: 最大执行时间（微秒）。
        :return: 下一个位置、下一个朝向、到最终目标的距离。
        """
        w = self.world

        # 获取目标向量和距离
        target_vec = target - w.robot.loc_head_position[:2]
        target_dist = np.linalg.norm(target_vec)

        # 获取障碍物
        obstacles = self.get_obstacles(include_teammates=True, include_opponents=True, include_play_mode_restrictions=True,
                                       mode=Path_Manager.MODE_AGGRESSIVE if is_aggressive else Path_Manager.MODE_CAUTIOUS,
                                       priority_unums=priority_unums)

        # 获取路径
        start_pos = self._get_hot_start(Path_Manager.HOT_START_DIST_WALK) if target_dist > 0.4 else self.world.robot.loc_head_position[:2]
        path, path_len, path_status, path_cost = self.get_path(start_pos, True, obstacles, target, timeout)
        path_end = path[-2:]  # A* 允许的最后一个位置

        # 获取下一个目标位置
        next_pos = self._extract_target_from_path(path, path_len, ret_segments)

        # 获取下一个目标朝向
        if torso_ori is not None:
            next_ori = torso_ori
        elif target_dist > 0.1:
            next_ori = M.vector_angle(target_vec)
        else:
            next_ori = w.robot.imu_torso_orientation

        # 更新热启动位置
        if path_len != 0:
            next_pos_vec = next_pos - self.world.robot.loc_head_position[:2]
            next_pos_dist = np.linalg.norm(next_pos_vec)
            self._update_hot_start(M.vector_angle(next_pos_vec, is_rad=True), min(Path_Manager.HOT_START_DIST_WALK, next_pos_dist))

        # 计算到最终目标的距离
        distance_to_final_target = np.linalg.norm(path_end - w.robot.loc_head_position[:2])

        return next_pos, next_ori, distance_to_final_target
    def get_dribble_path(self, ret_segments=None, optional_2d_target=None, goalpost_safety_margin=0.4, timeout=3000):
        """
        获取带球路径的下一个位置（下一个相对朝向）。
        :param ret_segments: 返回的目标段数（如果为 None，则根据机器人的速度动态设置）。
        :param optional_2d_target: 2D 目标位置（如果为 None，则目标是对方球门）。
        :param goalpost_safety_margin: 球门柱的安全距离。
        :param timeout: 最大执行时间（微秒）。
        :return: 下一个位置、下一个相对朝向。
        """
        r = self.world.robot
        ball_2d = self.world.ball_abs_pos[:2]

        # 获取障碍物
        obstacles = self.get_obstacles(include_teammates=True, include_opponents=True, include_play_mode_restrictions=False,
                                    max_distance=5, max_age=1000, goalpost_safety_margin=goalpost_safety_margin, mode=Path_Manager.MODE_DRIBBLE)

        # 获取路径
        start_pos = self._get_hot_start(Path_Manager.HOT_START_DIST_DRIBBLE)
        path, path_len, path_status, path_cost = self.get_path(start_pos, False, obstacles, optional_2d_target, timeout)

        # 获取下一个目标位置和朝向
        if ret_segments is None:
            ret_segments = 2.0

        next_pos = self._extract_target_from_path(path, path_len, ret_segments)
        next_rel_ori = M.normalize_deg(M.vector_angle(next_pos - ball_2d) - r.imu_torso_orientation)

        # 更新热启动位置
        if path_len != 0:
            self._update_hot_start(M.deg_to_rad(r.imu_torso_orientation), Path_Manager.HOT_START_DIST_DRIBBLE)

        # 绘制路径
        if self._draw_path and path_status != Path_Manager.STATUS_DIRECT:
            d = self.world.team_draw if self._use_team_channel else self.world.draw
            d.point(next_pos, 2, d.Color.pink, "path_next_pos", False)  # 如果绘图对象内部禁用，则不会绘制
            d.line(ball_2d, next_pos, 2, d.Color.pink, "path_next_pos")  # 如果绘图对象内部禁用，则不会绘制

        return next_pos, next_rel_ori
    def get_push_path(self, ret_segments=1.5, optional_2d_target=None, avoid_opponents=False, timeout=3000):
        """
        获取推球路径的下一个位置。
        :param ret_segments: 返回的目标段数。
        :param optional_2d_target: 2D 目标位置（如果为 None，则目标是对方球门）。
        :param avoid_opponents: 是否避免对手。
        :param timeout: 最大执行时间（微秒）。
        :return: 下一个位置。
        """
        ball_2d = self.world.ball_abs_pos[:2]

        # 获取障碍物
        obstacles = self.get_obstacles(include_teammates=False, include_opponents=avoid_opponents, include_play_mode_restrictions=False)

        # 获取路径
        path, path_len, path_status, path_cost = self.get_path(ball_2d, False, obstacles, optional_2d_target, timeout)

        # 获取下一个目标位置
        next_pos = self._extract_target_from_path(path, path_len, ret_segments)

        return next_pos
    def get_path(self, start, allow_out_of_bounds, obstacles=[], optional_2d_target=None, timeout=3000):
        """
        获取路径。
        :param start: 起始位置。
        :param allow_out_of_bounds: 是否允许路径超出边界。
        :param obstacles: 障碍物列表。
        :param optional_2d_target: 2D 目标位置（如果为 None，则目标是对方球门）。
        :param timeout: 最大执行时间（微秒）。
        :return: 路径、路径长度、路径状态、路径成本。
        """
        go_to_goal = int(optional_2d_target is None)

        if optional_2d_target is None:
            optional_2d_target = (0, 0)

        # 展平障碍物
        obstacles = sum(obstacles, tuple())
        assert len(obstacles) % 5 == 0, "每个障碍物应由 5 个浮点值表示"

        # 路径参数：起始位置、是否允许超出边界、是否前往球门、目标位置、超时时间（微秒）、障碍物
        params = np.array([*start, int(allow_out_of_bounds), go_to_goal, *optional_2d_target, timeout, *obstacles], np.float32)
        path_ret = a_star.compute(params)
        path = path_ret[:-2]
        path_status = path_ret[-2]

        # 绘制路径段
        if self._draw_path:
            d = self.world.team_draw if self._use_team_channel else self.world.draw
            if d.enabled:
                c = {0: d.Color.green_lawn, 1: d.Color.yellow, 2: d.Color.red, 3: d.Color.cyan}[path_status]
                for j in range(0, len(path) - 2, 2):
                    d.line((path[j], path[j + 1]), (path[j + 2], path[j + 3]), 1, c, "path_segments", False)
                d.flush("path_segments")

        return path, len(path) // 2 - 1, path_status, path_ret[-1]  # 路径、路径长度（段数）、路径状态、路径成本
