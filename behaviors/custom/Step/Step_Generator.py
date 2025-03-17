import math


class Step_Generator():
    """
    步态生成器类，用于计算每条腿的目标位置。
    """
    GRAVITY = 9.81  # 重力加速度
    Z0 = 0.2  # 初始Z值

    def __init__(self, feet_y_dev, sample_time, max_ankle_z) -> None:
        """
        初始化函数。
        :param feet_y_dev: 脚部Y轴偏移量。
        :param sample_time: 采样时间。
        :param max_ankle_z: 踝关节Z轴最大值。
        """
        self.feet_y_dev = feet_y_dev  # 脚部Y轴偏移量
        self.sample_time = sample_time  # 采样时间
        self.state_is_left_active = False  # 当前是否左腿活动
        self.state_current_ts = 0  # 当前时间步
        self.switch = False  # 是否切换腿
        self.external_progress = 0  # 非重叠进度
        self.max_ankle_z = max_ankle_z  # 踝关节Z轴最大值


    def get_target_positions(self, reset, ts_per_step, z_span, z_extension):
        """
        获取每条腿的目标位置。
        :param reset: 是否重置。
        :param ts_per_step: 每步的时间步数。
        :param z_span: Z轴跨度。
        :param z_extension: Z轴最大延伸距离。
        :return: 目标位置元组（左腿Y，左腿Z，右腿Y，右腿Z）。
        """
        assert type(ts_per_step) == int and ts_per_step > 0, "ts_per_step必须是正整数！"

        #-------------------------- 时间步前进
        if reset:
            self.ts_per_step = ts_per_step  # 步态持续时间（时间步数）
            self.swing_height = z_span  # 摆动高度
            self.max_leg_extension = z_extension  # 腿部最大延伸距离
            self.state_current_ts = 0  # 当前时间步重置为0
            self.state_is_left_active = False  # 左腿活动状态重置为False
            self.switch = False  # 切换状态重置为False
        elif self.switch:
            self.state_current_ts = 0  # 时间步重置为0
            self.state_is_left_active = not self.state_is_left_active  # 切换活动腿
            self.switch = False  # 切换状态重置为False
        else:
            self.state_current_ts += 1  # 时间步加1

        #-------------------------- 计算质心Y位置
        W = math.sqrt(self.Z0 / self.GRAVITY)  # 计算W值

        step_time = self.ts_per_step * self.sample_time  # 步态总时间
        time_delta = self.state_current_ts * self.sample_time  # 当前时间步对应的时间

        y0 = self.feet_y_dev  # 初始Y值
        y_swing = y0 + y0 * (math.sinh((step_time - time_delta) / W) + math.sinh(time_delta / W)) / math.sinh(-step_time / W)  # 计算Y摆动值

        #-------------------------- 限制最大延伸和摆动高度
        z0 = min(-self.max_leg_extension, self.max_ankle_z)  # 限制初始Z值
        zh = min(self.swing_height, self.max_ankle_z - z0)  # 限制摆动高度

        #-------------------------- 计算Z摆动
        progress = self.state_current_ts / self.ts_per_step  # 当前进度
        self.external_progress = self.state_current_ts / (self.ts_per_step - 1)  # 非重叠进度
        active_z_swing = zh * math.sin(math.pi * progress)  # 活动腿的Z摆动值

        #-------------------------- 在最后一步后接受新参数
        if self.state_current_ts + 1 >= self.ts_per_step:
            self.ts_per_step = ts_per_step  # 更新步态持续时间
            self.swing_height = z_span  # 更新摆动高度
            self.max_leg_extension = z_extension  # 更新腿部最大延伸距离
            self.switch = True  # 设置切换状态为True

        #-------------------------- 区分活动腿
        if self.state_is_left_active:
            return y0 + y_swing, active_z_swing + z0, -y0 + y_swing, z0  # 左腿活动时的目标位置
        else:
            return y0 - y_swing, z0, -y0 - y_swing, active_z_swing + z0  # 右腿活动时的目标位置