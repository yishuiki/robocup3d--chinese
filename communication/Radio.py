from typing import List
from world.commons.Other_Robot import Other_Robot
from world.World import World
import numpy as np

class Radio():
    '''
    地图范围是硬编码的：
        队友/对手的位置 (x,y) 在 ([-16,16],[-11,11]) 内
        球的位置 (x,y) 在 ([-15,15],[-10,10]) 内
    已知服务器限制：
        声明：所有 ASCII 字符从 0x20 到 0x7E，除了 ' ', '(', ')'
        错误：
            - ' 或 " 会截断消息
            - '\' 在末尾或靠近另一个 '\' 时
            - ';' 在消息开头
    '''
    # 地图范围是硬编码的：

    # 行数、列数、半行索引、半列索引、(行数-1)/x_span、(列数-1)/y_span、组合数、组合数*2状态，
    TP = 321, 221, 160, 110, 10, 10, 70941, 141882  # 队友位置
    OP = 201, 111, 100, 55, 6.25, 5, 22311, 44622  # 对手位置
    BP = 301, 201, 150, 100, 10, 10, 60501  # 球位置
    SYMB = "!#$%&*+,-./0123456789:<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~;"
    SLEN = len(SYMB)
    SYMB_TO_IDX = {ord(s): i for i, s in enumerate(SYMB)}

    def __init__(self, world: World, commit_announcement) -> None:
        '''
        初始化 Radio 类
        :param world: 世界状态
        :param commit_announcement: 提交广播消息的函数
        '''
        self.world = world
        self.commit_announcement = commit_announcement
        r = world.robot
        t = world.teammates
        o = world.opponents
        self.groups = (  # 玩家队伍/球衣号码，组是否有球，自己是否在组内
            [(t[9], t[10], o[6], o[7], o[8], o[9], o[10]), True],  # 2 名队友，5 名对手，有球
            [(t[0], t[1], t[2], t[3], t[4], t[5], t[6]), False],  # 7 名队友
            [(t[7], t[8], o[0], o[1], o[2], o[3], o[4], o[5]), False]  # 2 名队友，6 名对手
        )
        for g in self.groups:  # 添加 '自己是否在组内'
            g.append(any(i.is_self for i in g[0]))

    def get_player_combination(self, pos, is_unknown, is_down, info):
        '''
        返回组合（从 0 开始）和可能的组合数
        :param pos: 位置
        :param is_unknown: 是否未知
        :param is_down: 是否倒地
        :param info: 位置信息
        :return: 组合和可能的组合数
        '''
        if is_unknown:
            return info[7] + 1, info[7] + 2  # 返回未知组合

        x, y = pos[:2]

        if x < -17 or x > 17 or y < -12 or y > 12:
            return info[7], info[7] + 2  # 返回越界组合（如果在任何轴上超出 1 米）

        # 转换为整数以避免后续溢出
        l = int(np.clip(round(info[4] * x + info[2]), 0, info[0] - 1))  # 吸收越界位置（每个轴最多 1 米）
        c = int(np.clip(round(info[5] * y + info[3]), 0, info[1] - 1))

        return (l * info[1] + c) + (info[6] if is_down else 0), info[7] + 2  # 返回有效组合

    def get_ball_combination(self, x, y):
        '''
        返回组合（从 0 开始）和可能的组合数
        :param x: 球的 x 坐标
        :param y: 球的 y 坐标
        :return: 组合和可能的组合数
        '''
        # 如果球越界，强制将其拉回
        l = int(np.clip(round(Radio.BP[4] * x + Radio.BP[2]), 0, Radio.BP[0] - 1))
        c = int(np.clip(round(Radio.BP[5] * y + Radio.BP[3]), 0, Radio.BP[1] - 1))

        return l * Radio.BP[1] + c, Radio.BP[6]  # 返回有效组合

    def get_ball_position(self, comb):
        '''
        根据组合获取球的位置
        :param comb: 组合
        :return: 球的位置
        '''
        l = comb // Radio.BP[1]
        c = comb % Radio.BP[1]

        return np.array([l / Radio.BP[4] - 15, c / Radio.BP[5] - 10, 0.042])  # 假设球在地面上

    def get_player_position(self, comb, info):
        '''
        根据组合获取球员位置
        :param comb: 组合
        :param info: 位置信息
        :return: 球员位置
        '''
        if comb == info[7]:
            return -1  # 球员越界
        if comb == info[7] + 1:
            return -2  # 球员位置未知

        is_down = comb >= info[6]
        if is_down:
            comb -= info[6]

        l = comb // info[1]
        c = comb % info[1]

        return l / info[4] - 16, c / info[5] - 11, is_down

    def check_broadcast_requirements(self):
        '''
        检查是否满足广播组的要求

        返回值
        -------
        ready : bool
            如果满足所有要求，则为 True

        序列：g0,g1,g2, ig0,ig1,ig2, iig0,iig1,iig2  （完整周期：0.36 秒）
            igx  表示      “不完整的组”，其中最多有 1 个元素最近可能丢失
            iigx 表示 “非常不完整的组”，其中最多有 2 个元素最近可能丢失
            理由：防止不完整的消息垄断广播空间

        然而：
        - 第一轮：当组内没有成员丢失时，该组每 0.36 秒更新 3 次
        - 第二轮：当组内有 1 个成员最近丢失时，该组每 0.36 秒更新 2 次
        - 第三轮：当组内有 2 个成员最近丢失时，该组每 0.36 秒更新 1 次
        -          当组内有 >2 个成员最近丢失时，该组不会更新

        从未被看到或听到的球员不被视为 “最近丢失”。
        如果从一开始就只有 1 个组成员，则相应的组可以更新，但不在第一轮。
        这样，第一轮就不会被无知的代理垄断，这在有 22 名球员的比赛中很重要。
        '''
        w = self.world
        r = w.robot
        ago40ms = w.time_local_ms - 40
        ago370ms = w.time_local_ms - 370  # 最大延迟（最多 2 个 MIA）是 360 毫秒，因为广播有 20 毫秒的延迟（否则最大延迟将是 340 毫秒）
        group: List[Other_Robot]

        idx9 = int((w.time_server * 25) + 0.1) % 9  # 9 个阶段的序列
        max_MIA = idx9 // 3  # 最大 MIA 球员数量（基于服务器时间）
        group_idx = idx9 % 3  # 组编号（基于服务器时间）
        group, has_ball, is_self_included = self.groups[group_idx]

        #============================================ 0. 检查组是否有效

        if has_ball and w.ball_abs_pos_last_update < ago40ms:  # 包含球且球的位置未更新
            return False

        if is_self_included and r.loc_last_update < ago40ms:  # 包含自己且无法自我定位
            return False

        # 获取之前被看到或听到但最近未被看到的球员
        MIAs = [not ot.is_self and ot.state_last_update < ago370ms and ot.state_last_update > 0 for ot in group]
        self.MIAs = [ot.state_last_update == 0 or MIAs[i] for i, ot in enumerate(group)]  # 添加从未被看到的球员

        if sum(MIAs) > max_MIA:  # 检查最近丢失的成员数量是否超过阈值
            return False

        # 从未被看到的球员总是被忽略，除非：
        # - 这是 0 MIA 的轮次（见上文解释）
        # - 所有球员都丢失了
        if (max_MIA == 0 and any(self.MIAs)) or all(self.MIAs):
            return False

        # 检查无效成员。条件：
        # - 球员是其他球员且不是 MIA 且：
        #      - 最后更新时间 >40ms 或
        #      - 最后更新未包含头部（头部对于提供状态和准确位置很重要）

        if any(
            (not ot.is_self and not self.MIAs[i] and
                (ot.state_last_update < ago40ms or ot.state_last_update == 0 or len(ot.state_abs_pos) < 3)  # （最后更新：没有头部或时间过旧）
            ) for i, ot in enumerate(group)
        ):
            return False

        return True

    def broadcast(self):
        '''
        如果满足某些条件，则向队友提交消息
        消息包含：每个移动实体的位置/状态
        '''

        if not self.check_broadcast_requirements():
            return

        w = self.world
        ot: Other_Robot

        group_idx = int((w.time_server * 25) + 0.1) % 3  # 基于服务器时间的组编号
        group, has_ball, _ = self.groups[group_idx]

        #============================================ 1. 创建组合

        # 添加消息编号
        combination = group_idx
        no_of_combinations = 3

        # 添加球的组合
        if has_ball:
            c, n = self.get_ball_combination(w.ball_abs_pos[0], w.ball_abs_pos[1])
            combination += c * no_of_combinations
            no_of_combinations *= n

        # 添加组组合
        for i, ot in enumerate(group):
            c, n = self.get_player_combination(ot.state_abs_pos,  # 球员位置
                                               self.MIAs[i], ot.state_fallen,  # 是否未知，是否倒地
                                               Radio.TP if ot.is_teammate else Radio.OP)  # 是否为队友
            combination += c * no_of_combinations
            no_of_combinations *= n

        assert(no_of_combinations < 9.61e38)  # 88*89^19（第一个字符不能是 ';'）

        #============================================ 2. 创建消息

        # 第一个消息符号：由于服务器错误，不能是 ';'
        msg = Radio.SYMB[combination % (Radio.SLEN - 1)]
        combination //= (Radio.SLEN - 1)

        # 后续消息符号
        while combination:
            msg += Radio.SYMB[combination % Radio.SLEN]
            combination //= Radio.SLEN

        #============================================ 3. 提交消息

        self.commit_announcement(msg.encode())  # 提交消息

    def receive(self, msg: bytearray):
        w = self.world
        r = w.robot
        ago40ms = w.time_local_ms - 40
        ago110ms = w.time_local_ms - 110
        msg_time = w.time_local_ms - 20  # 消息是在上一步发送的

        #============================================ 1. 获取组合

        # 读取第一个符号，由于服务器错误，不能是 ';'
        combination = Radio.SYMB_TO_IDX[msg[0]]
        total_combinations = Radio.SLEN - 1

        if len(msg) > 1:
            for m in msg[1:]:
                combination += total_combinations * Radio.SYMB_TO_IDX[m]
                total_combinations *= Radio.SLEN

        #============================================ 2. 获取消息编号

        message_no = combination % 3
        combination //= 3
        group, has_ball, _ = self.groups[message_no]

        #============================================ 3. 获取数据

        if has_ball:
            ball_comb = combination % Radio.BP[6]
            combination //= Radio.BP[6]

        players_combs = []
        ot: Other_Robot
        for ot in group:
            info = Radio.TP if ot.is_teammate else Radio.OP
            players_combs.append(combination % (info[7] + 2))
            combination //= info[7] + 2

        #============================================ 4. 更新世界状态

        if has_ball and w.ball_abs_pos_last_update < ago40ms:  # 如果球未被看到，则更新球的位置
            time_diff = (msg_time - w.ball_abs_pos_last_update) / 1000
            ball = self.get_ball_position(ball_comb)
            w.ball_abs_vel = (ball - w.ball_abs_pos) / time_diff
            w.ball_abs_speed = np.linalg.norm(w.ball_abs_vel)
            w.ball_abs_pos_last_update = msg_time  # （误差：0-40 毫秒）
            w.ball_abs_pos = ball
            w.is_ball_abs_pos_from_vision = False

        for c, ot in zip(players_combs, group):

            # 处理自己的情况
            if ot.is_self:
                # 球的位置有一定噪声，无论我们是否看到它
                # 但我们的自我定位机制通常比其他玩家感知我们的方式要好得多
                if r.loc_last_update < ago110ms:  # 因此我们等待直到错过 2 个视觉步骤
                    data = self.get_player_position(c, Radio.TP)
                    if type(data) == tuple:
                        x, y, is_down = data
                        r.loc_head_position[:2] = x, y  # z 保持不变
                        r.loc_head_position_last_update = msg_time
                        r.radio_fallen_state = is_down
                        r.radio_last_update = msg_time
                continue

            # 如果其他机器人最近被看到，则不更新
            if ot.state_last_update >= ago40ms:
                continue

            info = Radio.TP if ot.is_teammate else Radio.OP
            data = self.get_player_position(c, info)
            if type(data) == tuple:
                x, y, is_down = data
                p = np.array([x, y])

                if ot.state_abs_pos is not None:  # 更新 x 和 y 分量的速度
                    time_diff = (msg_time - ot.state_last_update) / 1000
                    velocity = np.append((p - ot.state_abs_pos[:2]) / time_diff, 0)  # v.z = 0
                    vel_diff = velocity - ot.state_filtered_velocity
                    if np.linalg.norm(vel_diff) < 4:  # 否则假设它被传送了
                        ot.state_filtered_velocity /= (ot.vel_decay, ot.vel_decay, 1)  # 中和衰减（除了 z 轴）
                        ot.state_filtered_velocity += ot.vel_filter * vel_diff

                ot.state_fallen = is_down
                ot.state_last_update = msg_time
                ot.state_body_parts_abs_pos = {"head": p}
                ot.state_abs_pos = p
                ot.state_horizontal_dist = np.linalg.norm(p - r.loc_head_position[:2])
                ot.state_ground_area = (p, 0.3 if is_down else 0.2)  # 不太精确，但我们无法看到机器人

