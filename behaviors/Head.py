from math_ops.Math_Ops import Math_Ops as M
from world.World import World
import numpy as np

class Head:
    """
    控制机器人头部方向的类。
    通过动态调整头部朝向，帮助机器人在比赛中有效地观察球、标志或其他目标。
    """
    FIELD_FLAGS = World.FLAGS_CORNERS_POS + World.FLAGS_POSTS_POS  # 球场标志位置（角旗和球门标志）
    HEAD_PITCH = -35  # 头部的俯仰角（固定值）

    def __init__(self, world: World) -> None:
        """
        初始化 Head 类。

        :param world: World 类型的对象，表示比赛的全局状态。
        """
        self.world = world  # 全局状态对象
        self.look_left = True  # 头部搜索方向（左或右）
        self.state = 0  # 头部搜索状态（0: 调整位置，1..TIMEOUT-1: 引导搜索，TIMEOUT: 随机搜索）

    def execute(self):
        """
        根据当前状态和感知信息，计算头部的最佳朝向。
        如果可能，尝试将头部指向球或其他重要目标；否则执行搜索策略。
        """
        TIMEOUT = 30  # 引导搜索的最大时间（单位：帧）
        w = self.world  # 全局状态对象
        r = w.robot  # 机器人对象
        can_self_locate = r.loc_last_update > w.time_local_ms - w.VISUALSTEP_MS  # 机器人是否能够自定位

        #--------------------------------------- A. 球在视野内且机器人能够自定位
        if w.ball_last_seen > w.time_local_ms - w.VISUALSTEP_MS:  # 球在视野内
            if can_self_locate:
                # 计算最佳朝向（使用视觉信息中的球位置）
                best_dir = self.compute_best_direction(can_self_locate, use_ball_from_vision=True)
                self.state = 0  # 重置状态为“调整位置”
            elif self.state < TIMEOUT:
                #--------------------------------------- B. 球在视野内但机器人无法自定位
                # 计算最佳朝向（使用视觉信息中的球位置）
                best_dir = self.compute_best_direction(can_self_locate, use_ball_from_vision=True)
                self.state += 1  # 进入引导搜索并增加时间计数
        elif self.state < TIMEOUT:
            #--------------------------------------- C. 球不在视野内
            # 计算最佳朝向（不使用视觉信息中的球位置）
            best_dir = self.compute_best_direction(can_self_locate)
            self.state += 1  # 进入引导搜索并增加时间计数

        # 如果引导搜索超时，进入随机搜索模式
        if self.state == TIMEOUT:
            if w.ball_last_seen > w.time_local_ms - w.VISUALSTEP_MS:  # 球在视野内
                # 在球的两侧搜索（45°）
                ball_dir = M.vector_angle(w.ball_rel_torso_cart_pos[:2])  # 球相对于机器人的方向
                targ = np.clip(ball_dir + (45 if self.look_left else -45), -119, 119)
            else:  # 球不在视野内
                # 在两侧搜索（119°）
                targ = 119 if self.look_left else -119

            # 切换搜索方向
            if r.set_joints_target_position_direct([0, 1], np.array([targ, Head.HEAD_PITCH]), False) <= 0:
                self.look_left = not self.look_left  # 切换搜索方向（左/右）
        else:
            # 调整位置或引导搜索
            r.set_joints_target_position_direct([0, 1], np.array([best_dir, Head.HEAD_PITCH]), False)

    def compute_best_direction(self, can_self_locate, use_ball_from_vision=False):
        """
        根据当前状态和感知信息，计算头部的最佳朝向。

        :param can_self_locate: 布尔值，表示机器人是否能够自定位。
        :param use_ball_from_vision: 布尔值，表示是否使用视觉信息中的球位置。
        :return: 最佳朝向（角度，单位：度）
        """
        FOV_MARGIN = 15  # 视野安全边距（避免视野边缘）
        SAFE_RANGE = 120 - FOV_MARGIN * 2  # 安全视野范围
        HALF_RANGE = SAFE_RANGE / 2  # 安全视野范围的一半

        w = self.world  # 全局状态对象
        r = w.robot  # 机器人对象

        # 计算球与机器人的距离
        if use_ball_from_vision:
            ball_2d_dist = np.linalg.norm(w.ball_rel_torso_cart_pos[:2])  # 使用视觉信息中的球位置
        else:
            ball_2d_dist = np.linalg.norm(w.ball_abs_pos[:2] - r.loc_head_position[:2])  # 使用绝对位置信息

        # 如果球距离较远，计算球的方向
        if ball_2d_dist > 0.12:
            if use_ball_from_vision:
                ball_dir = M.vector_angle(w.ball_rel_torso_cart_pos[:2])  # 球相对于机器人的方向
            else:
                ball_dir = M.target_rel_angle(r.loc_head_position, r.imu_torso_orientation, w.ball_abs_pos)  # 球的绝对方向
        else:  # 球距离较近
            ball_dir = 0  # 朝向球的中心方向

        flags_diff = dict()  # 存储标志与球的相对角度

        # 遍历所有标志（角旗和球门标志）
        for f in Head.FIELD_FLAGS:
            flag_dir = M.target_rel_angle(r.loc_head_position, r.imu_torso_orientation, f)  # 标志的方向
            diff = M.normalize_deg(flag_dir - ball_dir)  # 标志与球的相对角度
            if abs(diff) < HALF_RANGE and can_self_locate:
                return ball_dir  # 如果机器人能够自定位且标志在视野内，直接返回球的方向
            flags_diff[f] = diff  # 存储标志与球的相对角度

        # 找到最近的标志
        closest_flag = min(flags_diff, key=lambda k: abs(flags_diff[k]))
        closest_diff = flags_diff[closest_flag]

        # 如果机器人能够自定位
        if can_self_locate:
            # 返回球的方向（尽量将球和标志都纳入视野）
            final_diff = min(abs(closest_diff) - HALF_RANGE, SAFE_RANGE) * np.sign(closest_diff)
        else:
            # 返回标志的方向（确保球在视野内）
            final_diff = np.clip(closest_diff, -SAFE_RANGE, SAFE_RANGE)
            return np.clip(ball_dir + final_diff, -119, 119)  # 限制角度范围

        return M.normalize_deg(ball_dir + final_diff)  # 返回最终朝向