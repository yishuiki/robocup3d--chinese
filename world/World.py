from collections import deque
from cpp.ball_predictor import ball_predictor
from cpp.localization import localization
from logs.Logger import Logger
from math import atan2, pi
from math_ops.Matrix_4x4 import Matrix_4x4
from world.commons.Draw import Draw
from world.commons.Other_Robot import Other_Robot
from world.Robot import Robot
import numpy as np

class World():
    # 固定的时间步长常量（以秒和毫秒为单位）
    STEPTIME = 0.02    # 固定时间步长（秒）
    STEPTIME_MS = 20   # 固定时间步长（毫秒）
    VISUALSTEP = 0.04  # 固定视觉更新时间步长（秒）
    VISUALSTEP_MS = 40 # 固定视觉更新时间步长（毫秒）

    """
    M_OUR_KICKOFF = 0：我方开球。
    M_OUR_KICK_IN = 1：我方边线球。
    M_OUR_CORNER_KICK = 2：我方角球。
    M_OUR_GOAL_KICK = 3：我方球门球。
    M_OUR_FREE_KICK = 4：我方任意球。
    M_OUR_PASS = 5：我方传球（可能是一个自定义模式）。
    M_OUR_DIR_FREE_KICK = 6：我方直接任意球。
    M_OUR_GOAL = 7：我方进球。
    M_OUR_OFFSIDE = 8：我方越位。
    """
    # 有利于我方的球权模式
    M_OUR_KICKOFF = 0
    M_OUR_KICK_IN = 1
    M_OUR_CORNER_KICK = 2
    M_OUR_GOAL_KICK = 3
    M_OUR_FREE_KICK = 4
    M_OUR_PASS = 5
    M_OUR_DIR_FREE_KICK = 6
    M_OUR_GOAL = 7
    M_OUR_OFFSIDE = 8

    """
    M_THEIR_KICKOFF = 9：对方开球。
    M_THEIR_KICK_IN = 10：对方边线球。
    M_THEIR_CORNER_KICK = 11：对方角球。
    M_THEIR_GOAL_KICK = 12：对方球门球。
    M_THEIR_FREE_KICK = 13：对方任意球。
    M_THEIR_PASS = 14：对方传球（可能是一个自定义模式）。
    M_THEIR_DIR_FREE_KICK = 15：对方直接任意球。
    M_THEIR_GOAL = 16：对方进球。
    M_THEIR_OFFSIDE = 17：对方越位。
    """
    # 有利于对方的球权模式
    M_THEIR_KICKOFF = 9
    M_THEIR_KICK_IN = 10
    M_THEIR_CORNER_KICK = 11
    M_THEIR_GOAL_KICK = 12
    M_THEIR_FREE_KICK = 13
    M_THEIR_PASS = 14
    M_THEIR_DIR_FREE_KICK = 15
    M_THEIR_GOAL = 16
    M_THEIR_OFFSIDE = 17

    """
    M_BEFORE_KICKOFF = 18：比赛开始前。
    M_GAME_OVER = 19：比赛结束。
    M_PLAY_ON = 20：比赛进行中（无特殊状态）。
    """
    # 中立球权模式
    M_BEFORE_KICKOFF = 18
    M_GAME_OVER = 19
    M_PLAY_ON = 20

    # 球权模式分组（便于处理相似的球权模式）
    MG_OUR_KICK = 0  # 我方球权（如开球、角球、任意球等）
    MG_THEIR_KICK = 1  # 对方球权（如开球、角球、任意球等）
    MG_ACTIVE_BEAM = 2  # 比赛前或对方进球（需要激活光束的场景）
    MG_PASSIVE_BEAM = 3  # 我方进球（被动等待光束激活）
    MG_OTHER = 4  # 其他状态（如比赛进行中、比赛结束）

    # 角旗和球门柱的位置（固定坐标）
    FLAGS_CORNERS_POS = ((-15,-10,0), (-15,+10,0), (+15,-10,0), (+15,+10,0))
    FLAGS_POSTS_POS = ((-15,-1.05,0.8),(-15,+1.05,0.8),(+15,-1.05,0.8),(+15,+1.05,0.8))

    def __init__(self, robot_type:int, team_name:str, unum:int, apply_play_mode_correction:bool, 
                 enable_draw:bool, logger:Logger, host:str) -> None:
        """
        初始化 World 类，用于管理机器人在比赛中的状态和环境信息。

        参数：
            robot_type (int): 机器人类型
            team_name (str): 我方队伍名称
            unum (int): 机器人编号
            apply_play_mode_correction (bool): 是否根据球权模式调整球的位置
            enable_draw (bool): 是否启用绘图功能
            logger (Logger): 日志记录器
            host (str): 服务器主机地址
        """
        self.team_name = team_name  # 我方队伍名称
        self.team_name_opponent = None  # 对方队伍名称（稍后由世界解析器更新）
        self.apply_play_mode_correction = apply_play_mode_correction  # 是否根据球权模式调整球的位置
        self.step = 0  # 总共接收到的模拟步数（与 self.time_local_ms 同步）
        self.time_server = 0.0  # 服务器时间（秒，不可靠，仅用于同步）
        self.time_local_ms = 0  # 可靠的本地模拟时间（毫秒）
        self.time_game = 0.0  # 比赛时间（秒）
        self.goals_scored = 0  # 我方进球数
        self.goals_conceded = 0  # 我方失球数
        self.team_side_is_left = None  # 我方是否在左侧（稍后由世界解析器更新）
        self.play_mode = None  # 当前球权模式（由服务器提供）
        self.play_mode_group = None  # 球权模式分组
        self.flags_corners = None  # 角旗位置（以我方在左侧为基准）
        self.flags_posts = None  # 球门柱位置（以我方在左侧为基准）
        self.ball_rel_head_sph_pos = np.zeros(3)  # 球相对于头部的位置（球面坐标）
        self.ball_rel_head_cart_pos = np.zeros(3)  # 球相对于头部的位置（笛卡尔坐标）
        self.ball_rel_torso_cart_pos = np.zeros(3)  # 球相对于躯干的位置（笛卡尔坐标）
        self.ball_rel_torso_cart_pos_history = deque(maxlen=20)  # 球相对于躯干的位置历史记录（最多保存20个）
        self.ball_abs_pos = np.zeros(3)  # 球的绝对位置（如果可见且机器人定位更新）
        self.ball_abs_pos_history = deque(maxlen=20)  # 球的绝对位置历史记录（最多保存20个）
        self.ball_abs_pos_last_update = 0  # 球的绝对位置上次更新的时间（毫秒）
        self.ball_abs_vel = np.zeros(3)  # 球的绝对速度（基于最近两个位置计算）
        self.ball_abs_speed = 0  # 球的绝对速度（标量）
        self.ball_is_visible = False  # 球是否可见（上次服务器消息是否包含球的视觉信息）
        self.is_ball_abs_pos_from_vision = False  # 球的绝对位置是否来自视觉（否则来自广播）
        self.ball_last_seen = 0  # 球最后一次被看到的时间（毫秒）
        self.ball_cheat_abs_pos = np.zeros(3)  # 球的绝对位置（由服务器提供的“作弊”信息）
        self.ball_cheat_abs_vel = np.zeros(3)  # 球的绝对速度（基于作弊信息）
        self.ball_2d_pred_pos = np.zeros((1,2))  # 预测的2D球位置
        self.ball_2d_pred_vel = np.zeros((1,2))  # 预测的2D球速度
        self.ball_2d_pred_spd = np.zeros(1)  # 预测的2D球速度（标量）
        self.lines = np.zeros((30,6))  # 可见线条的位置（相对于头部）
        self.line_count = 0  # 可见线条的数量
        self.vision_last_update = 0  # 上次视觉信息更新的时间
        self.vision_is_up_to_date = False  # 视觉信息是否最新
        self.teammates = [Other_Robot(i, True) for i in range(1,12)]  # 队友列表（按编号排序）
        self.opponents = [Other_Robot(i, False) for i in range(1,12)]  # 对手列表（按编号排序）
        self.teammates[unum-1].is_self = True  # 当前机器人是自己
        self.draw = Draw(enable_draw, unum, host, 32769)  # 当前机器人的绘图对象
        self.team_draw = Draw(enable_draw, 0, host, 32769)  # 队友共享的绘图对象
        self.logger = logger  # 日志记录器
        self.robot = Robot(unum, robot_type)  # 当前机器人对象

    def log(self, msg:str):
        """
        日志记录快捷方法。

        参数：
            msg (str): 要记录的消息
        """
        self.logger.write(msg, True, self.step)

    def get_ball_rel_vel(self, history_steps:int):
        """
        获取球相对于躯干的速度（m/s）。

        参数：
            history_steps (int): 使用的历史步数（范围为1到20）

        示例：
            history_steps=1：使用最近两个位置计算速度
            history_steps=2：使用最近三个位置计算速度
        """
        assert 1 <= history_steps <= 20, "参数 'history_steps' 必须在范围 [1,20] 内"

        if len(self.ball_rel_torso_cart_pos_history) == 0:
            # 如果历史记录为空，返回零速度
            return np.zeros(3)

        # 选择历史记录中的有效步数
        h_step = min(history_steps, len(self.ball_rel_torso_cart_pos_history))
        t = h_step * World.VISUALSTEP  # 时间间隔

        # 计算相对速度
        return (self.ball_rel_torso_cart_pos - self.ball_rel_torso_cart_pos_history[h_step-1]) / t

    def get_ball_abs_vel(self, history_steps:int):
        """
        获取球的绝对速度（m/s）。

        参数：
            history_steps (int): 使用的历史步数（范围为1到20）

        示例：
            history_steps=1：使用最近两个位置计算速度
            history_steps=2：使用最近三个位置计算速度
        """
        assert 1 <= history_steps <= 20, "参数 'history_steps' 必须在范围 [1,20] 内"

        if len(self.ball_abs_pos_history) == 0:
            # 如果历史记录为空，返回零速度
            return np.zeros(3)

        # 选择历史记录中的有效步数
        h_step = min(history_steps, len(self.ball_abs_pos_history))
        t = h_step * World.VISUALSTEP  # 时间间隔

        # 计算绝对速度
        return (self.ball_abs_pos - self.ball_abs_pos_history[h_step-1]) / t

    def get_predicted_ball_pos(self, max_speed):
        """
        获取球在预测速度小于等于 `max_speed` 时的2D预测位置。

        如果预测位置超出范围，则返回最后一个可用的预测位置。

        参数：
            max_speed (float): 球在返回的未来位置时的速度上限
        """
        b_sp = self.ball_2d_pred_spd  # 预测速度数组
        # 使用二分查找找到第一个速度小于等于 max_speed 的索引
        index = len(b_sp) - max(1, np.searchsorted(b_sp[::-1], max_speed, side='right'))
        return self.ball_2d_pred_pos[index]  # 返回对应位置的预测位置

    def get_intersection_point_with_ball(self, player_speed):
        """
        获取机器人与移动球的2D交点。

        参数：
            player_speed (float): 机器人追赶球的平均速度

        返回：
            交点位置 (ndarray)：机器人与球的交点（二维）
            交点距离 (float)：当前机器人位置到交点的距离
        """
        # 将相关参数打包为数组
        params = np.array([*self.robot.loc_head_position[:2], player_speed * 0.02, *self.ball_2d_pred_pos.flat], np.float32)
        # 调用预测函数
        pred_ret = ball_predictor.get_intersection(params)
        return pred_ret[:2], pred_ret[2]  # 返回交点位置和距离

    def update(self):
        """
        更新世界状态，包括球权模式、机器人定位、球的位置和速度等。
        """
        r = self.robot  # 当前机器人
        PM = self.play_mode  # 当前球权模式
        W = World  # 当前类的引用，用于访问类变量

        # 重置变量
        r.loc_is_up_to_date = False  # 定位是否更新
        r.loc_head_z_is_up_to_date = False  # 头部高度是否更新

        # 更新球权模式分组
        if PM in (W.M_PLAY_ON, W.M_GAME_OVER):  # 最常见的分组
            self.play_mode_group = W.MG_OTHER
        elif PM in (W.M_OUR_KICKOFF, W.M_OUR_KICK_IN, W.M_OUR_CORNER_KICK, W.M_OUR_GOAL_KICK,
                    W.M_OUR_OFFSIDE, W.M_OUR_PASS, W.M_OUR_DIR_FREE_KICK, W.M_OUR_FREE_KICK):
            self.play_mode_group = W.MG_OUR_KICK  # 我方球权
        elif PM in (W.M_THEIR_KICK_IN, W.M_THEIR_CORNER_KICK, W.M_THEIR_GOAL_KICK, W.M_THEIR_OFFSIDE,
                    W.M_THEIR_PASS, W.M_THEIR_DIR_FREE_KICK, W.M_THEIR_FREE_KICK, W.M_THEIR_KICKOFF):
            self.play_mode_group = W.MG_THEIR_KICK  # 对方球权
        elif PM in (W.M_BEFORE_KICKOFF, W.M_THEIR_GOAL):
            self.play_mode_group = W.MG_ACTIVE_BEAM  # 比赛前或对方进球
        elif PM in (W.M_OUR_GOAL,):
            self.play_mode_group = W.MG_PASSIVE_BEAM  # 我方进球
        elif PM is not None:
            raise ValueError(f'未知的球权模式 ID: {PM}')

        # 更新机器人的姿态（前向运动学）
        r.update_pose()

        # 如果球可见，计算球相对于躯干的位置
        if self.ball_is_visible:
            self.ball_rel_torso_cart_pos = r.head_to_body_part_transform("torso", self.ball_rel_head_cart_pos)

        # 如果视觉信息更新，执行基于视觉的定位
        if self.vision_is_up_to_date:
            # 准备定位所需的变量
            feet_contact = np.zeros(6)  # 脚部接触信息

            # 获取左脚和右脚的接触点
            lf_contact = r.frp.get('lf', None)
            rf_contact = r.frp.get('rf', None)
            if lf_contact is not None:
                feet_contact[0:3] = Matrix_4x4(r.body_parts["lfoot"].transform).translate(lf_contact[0:3], True).get_translation()
            if rf_contact is not None:
                feet_contact[3:6] = Matrix_4x4(r.body_parts["rfoot"].transform).translate(rf_contact[0:3], True).get_translation()

            # 球的位置信息
            ball_pos = np.concatenate((self.ball_rel_head_cart_pos, self.ball_cheat_abs_pos))

            # 角旗和球门柱的信息
            corners_list = [[key in self.flags_corners, 1.0, *key, *self.flags_corners.get(key, (0, 0, 0))] for key in World.FLAGS_CORNERS_POS]
            posts_list = [[key in self.flags_posts, 0.0, *key, *self.flags_posts.get(key, (0, 0, 0))] for key in World.FLAGS_POSTS_POS]
            all_landmarks = np.array(corners_list + posts_list, float)

            # 调用定位算法
            loc = localization.compute(
                r.feet_toes_are_touching['lf'],
                r.feet_toes_are_touching['rf'],
                feet_contact,
                self.ball_is_visible,
                ball_pos,
                r.cheat_abs_pos,
                all_landmarks,
                self.lines[0:self.line_count])

            # 更新机器人定位
            r.update_localization(loc, self.time_local_ms)

            # 更新当前机器人在队友列表中的状态
            me = self.teammates[r.unum - 1]
            me.state_last_update = r.loc_last_update
            me.state_abs_pos = r.loc_head_position
            me.state_fallen = r.loc_head_z < 0.3  # 使用头部高度判断是否摔倒
            me.state_orientation = r.loc_torso_orientation
            me.state_ground_area = (r.loc_head_position[:2], 0.2)  # 用于定位演示

            # 将球的当前位置保存到历史记录中（即使未更新）
            self.ball_abs_pos_history.appendleft(self.ball_abs_pos)
            self.ball_rel_torso_cart_pos_history.appendleft(self.ball_rel_torso_cart_pos)

            # 根据球权模式或视觉信息更新球的绝对位置
            ball = None
            if self.apply_play_mode_correction:
                if PM == W.M_OUR_CORNER_KICK:
                    ball = np.array([15, 5.483 if self.ball_abs_pos[1] > 0 else -5.483, 0.042], float)
                elif PM == W.M_THEIR_CORNER_KICK:
                    ball = np.array([-15, 5.483 if self.ball_abs_pos[1] > 0 else -5.483, 0.042], float)
                elif PM in [W.M_OUR_KICKOFF, W.M_THEIR_KICKOFF, W.M_OUR_GOAL, W.M_THEIR_GOAL]:
                    ball = np.array([0, 0, 0.042], float)
                elif PM == W.M_OUR_GOAL_KICK:
                    ball = np.array([-14, 0, 0.042], float)
                elif PM == W.M_THEIR_GOAL_KICK:
                    ball = np.array([14, 0, 0.042],float)
                # 如果机器人靠近硬编码的球位置，则忽略硬编码位置（优先使用视觉信息）
                if ball is not None and np.linalg.norm(r.loc_head_position[:2] - ball[:2]) < 1:
                    ball = None

            # 如果球位置为空且球可见且机器人定位更新，则根据视觉信息计算球的绝对位置
            if ball is None and self.ball_is_visible and r.loc_is_up_to_date:
                ball = r.loc_head_to_field_transform(self.ball_rel_head_cart_pos)
                ball[2] = max(ball[2], 0.042)  # 球的最低高度为球的半径
                if PM != W.M_BEFORE_KICKOFF:  # 兼容没有激活足球规则的测试
                    ball[:2] = np.clip(ball[:2], [-15, -10], [15, 10])  # 强制球位置在场地内

            # 更新内部球位置（也会被广播更新）
            if ball is not None:
                time_diff = (self.time_local_ms - self.ball_abs_pos_last_update) / 1000
                self.ball_abs_vel = (ball - self.ball_abs_pos) / time_diff  # 计算球的速度
                self.ball_abs_speed = np.linalg.norm(self.ball_abs_vel)  # 计算球的标量速度
                self.ball_abs_pos_last_update = self.time_local_ms
                self.ball_abs_pos = ball
                self.is_ball_abs_pos_from_vision = True

            # 对队友和对手的速度进行衰减（稍后会在更新时抵消）
            for p in self.teammates:
                p.state_filtered_velocity *= p.vel_decay
            for p in self.opponents:
                p.state_filtered_velocity *= p.vel_decay

            # 更新队友和对手的状态
            if r.loc_is_up_to_date:
                for p in self.teammates:
                    if not p.is_self:  # 如果不是自己
                        if p.is_visible:  # 如果队友可见，执行完整更新
                            self.update_other_robot(p)
                        elif p.state_abs_pos is not None:  # 否则更新其水平距离（假设最后已知位置不变）
                            p.state_horizontal_dist = np.linalg.norm(r.loc_head_position[:2] - p.state_abs_pos[:2])

                for p in self.opponents:
                    if p.is_visible:  # 如果对手可见，执行完整更新
                        self.update_other_robot(p)
                    elif p.state_abs_pos is not None:  # 否则更新其水平距离（假设最后已知位置不变）
                        p.state_horizontal_dist = np.linalg.norm(r.loc_head_position[:2] - p.state_abs_pos[:2])

        # 更新球位置/速度的预测
        if self.play_mode_group != W.MG_OTHER:  # 不是“比赛进行中”或“比赛结束”，球必须静止
            self.ball_2d_pred_pos = self.ball_abs_pos[:2].copy().reshape(1, 2)
            self.ball_2d_pred_vel = np.zeros((1, 2))
            self.ball_2d_pred_spd = np.zeros(1)

        elif self.ball_abs_pos_last_update == self.time_local_ms:  # 如果球位置更新（来自视觉或广播）
            # 准备参数并调用预测函数
            params = np.array([*self.ball_abs_pos[:2], *np.copy(self.get_ball_abs_vel(6)[:2])], np.float32)
            pred_ret = ball_predictor.predict_rolling_ball(params)
            sample_no = len(pred_ret) // 5 * 2
            self.ball_2d_pred_pos = pred_ret[:sample_no].reshape(-1, 2)
            self.ball_2d_pred_vel = pred_ret[sample_no:sample_no * 2].reshape(-1, 2)
            self.ball_2d_pred_spd = pred_ret[sample_no * 2:]

        elif len(self.ball_2d_pred_pos) > 1:  # 否则，推进到下一个预测步骤（如果可用）
            self.ball_2d_pred_pos = self.ball_2d_pred_pos[1:]
            self.ball_2d_pred_vel = self.ball_2d_pred_vel[1:]
            self.ball_2d_pred_spd = self.ball_2d_pred_spd[1:]

        # 更新机器人的IMU（必须在定位更新后执行）
        r.update_imu(self.time_local_ms)

    def update_other_robot(self, other_robot: Other_Robot):
        """
        根据可见的身体部位更新其他机器人的状态（也会被广播更新，除了朝向）。
        """
        o = other_robot  # 当前要更新的其他机器人
        r = self.robot  # 当前机器人

        # 更新其他机器人的身体部位绝对位置
        o.state_body_parts_abs_pos = o.body_parts_cart_rel_pos.copy()
        for bp, pos in o.body_parts_cart_rel_pos.items():
            # 使用头部到场地的变换矩阵计算身体部位的绝对位置
            o.state_body_parts_abs_pos[bp] = r.loc_head_to_field_transform(pos, False)

        # 辅助变量
        bps_apos = o.state_body_parts_abs_pos  # 身体部位的绝对位置
        bps_2d_apos_list = [v[:2] for v in bps_apos.values()]  # 身体部位的二维绝对位置列表
        avg_2d_pt = np.average(bps_2d_apos_list, axis=0)  # 可见身体部位的二维平均位置
        head_is_visible = 'head' in bps_apos  # 头部是否可见

        # 根据头部是否可见评估机器人的状态
        if head_is_visible:
            o.state_fallen = bps_apos['head'][2] < 0.3  # 如果头部高度小于0.3，则认为机器人摔倒

        # 如果已知机器人位置，则计算速度
        if o.state_abs_pos is not None:
            time_diff = (self.time_local_ms - o.state_last_update) / 1000
            if head_is_visible:
                # 如果上次位置是二维的，假设z坐标不变，速度的z分量为0
                old_p = o.state_abs_pos if len(o.state_abs_pos) == 3 else np.append(o.state_abs_pos, bps_apos['head'][2])
                velocity = (bps_apos['head'] - old_p) / time_diff
                decay = o.vel_decay  # 抵消衰减
            else:  # 如果头部不可见，仅更新x和y分量的速度
                velocity = np.append((avg_2d_pt - o.state_abs_pos[:2]) / time_diff, 0)
                decay = (o.vel_decay, o.vel_decay, 1)  # 抵消衰减（除了z轴）
            # 应用滤波器
            if np.linalg.norm(velocity - o.state_filtered_velocity) < 4:  # 如果速度变化不大，认为是正常运动
                o.state_filtered_velocity /= decay  # 抵消衰减
                o.state_filtered_velocity += o.vel_filter * (velocity - o.state_filtered_velocity)

        # 根据头部是否可见计算机器人的位置
        if head_is_visible:
            o.state_abs_pos = bps_apos['head']  # 如果头部可见，使用头部位置
        else:
            o.state_abs_pos = avg_2d_pt  # 否则使用可见身体部位的二维平均位置

        # 计算机器人与当前机器人的水平距离
        o.state_horizontal_dist = np.linalg.norm(r.loc_head_position[:2] - o.state_abs_pos[:2])

        # 根据一对下臂或双脚（或两者的平均值）计算朝向
        lr_vec = None
        if 'llowerarm' in bps_apos and 'rlowerarm' in bps_apos:
            lr_vec = bps_apos['rlowerarm'] - bps_apos['llowerarm']
        if 'lfoot' in bps_apos and 'rfoot' in bps_apos:
            if lr_vec is None:
                lr_vec = bps_apos['rfoot'] - bps_apos['lfoot']
            else:
                lr_vec = (lr_vec + (bps_apos['rfoot'] - bps_apos['lfoot'])) / 2
        if lr_vec is not None:
            o.state_orientation = atan2(lr_vec[1], lr_vec[0]) * 180 / pi + 90  # 将弧度转换为角度

        # 计算机器人在地面上的投影区域（圆形）
        if o.state_horizontal_dist < 4:  # 如果机器人距离较近，需要精确计算
            max_dist = np.max(np.linalg.norm(bps_2d_apos_list - avg_2d_pt, axis=1))
        else:
            max_dist = 0.2
        o.state_ground_area = (avg_2d_pt, max_dist)

        # 更新时间戳
        o.state_last_update = self.time_local_ms