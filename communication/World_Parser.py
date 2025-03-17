from math_ops.Math_Ops import Math_Ops as M
from world.Robot import Robot
from world.World import World
import math
import numpy as np


class World_Parser:
    """
    用于解析机器人足球比赛中的环境信息，并更新全局状态。
    输入是一个字节流，解析后更新 World 对象的状态。
    """

    def __init__(self, world: World, hear_callback):
        """
        初始化 World_Parser 对象。
        
        :param world: World 类型的对象，表示比赛的全局状态。
        :param hear_callback: 回调函数，用于处理从队友接收到的消息。
        """
        self.LOG_PREFIX = "World_Parser.py: "  # 日志前缀，用于标识日志来源
        self.world = world  # 全局状态对象
        self.hear_callback = hear_callback  # 队友消息回调函数
        self.exp = None  # 当前解析的字节流
        self.depth = None  # 当前解析的嵌套深度

        # 左半场和右半场的标志位置信息（如球门和角旗）
        self.LEFT_SIDE_FLAGS = {
            b'F2L': (-15, -10, 0), b'F1L': (-15, +10, 0),
            b'F2R': (+15, -10, 0), b'F1R': (+15, +10, 0),
            b'G2L': (-15, -1.05, 0.8), b'G1L': (-15, +1.05, 0.8),
            b'G2R': (+15, -1.05, 0.8), b'G1R': (+15, +1.05, 0.8)
        }
        self.RIGHT_SIDE_FLAGS = {
            b'F2L': (+15, +10, 0), b'F1L': (+15, -10, 0),
            b'F2R': (-15, +10, 0), b'F1R': (-15, -10, 0),
            b'G2L': (+15, +1.05, 0.8), b'G1L': (+15, -1.05, 0.8),
            b'G2R': (-15, +1.05, 0.8), b'G1R': (-15, -1.05, 0.8)
        }

        # 比赛模式到 ID 的映射
        self.play_mode_to_id = None
        self.LEFT_PLAY_MODE_TO_ID = {
            "KickOff_Left": World.M_OUR_KICKOFF, "KickIn_Left": World.M_OUR_KICK_IN,
            "corner_kick_left": World.M_OUR_CORNER_KICK, "goal_kick_left": World.M_OUR_GOAL_KICK,
            "free_kick_left": World.M_OUR_FREE_KICK, "pass_left": World.M_OUR_PASS,
            "direct_free_kick_left": World.M_OUR_DIR_FREE_KICK, "Goal_Left": World.M_OUR_GOAL,
            "offside_left": World.M_OUR_OFFSIDE, "KickOff_Right": World.M_THEIR_KICKOFF,
            "KickIn_Right": World.M_THEIR_KICK_IN, "corner_kick_right": World.M_THEIR_CORNER_KICK,
            "goal_kick_right": World.M_THEIR_GOAL_KICK, "free_kick_right": World.M_THEIR_FREE_KICK,
            "pass_right": World.M_THEIR_PASS, "direct_free_kick_right": World.M_THEIR_DIR_FREE_KICK,
            "Goal_Right": World.M_THEIR_GOAL, "offside_right": World.M_THEIR_OFFSIDE,
            "BeforeKickOff": World.M_BEFORE_KICKOFF, "GameOver": World.M_GAME_OVER, "PlayOn": World.M_PLAY_ON
        }
        self.RIGHT_PLAY_MODE_TO_ID = {
            "KickOff_Left": World.M_THEIR_KICKOFF, "KickIn_Left": World.M_THEIR_KICK_IN,
            "corner_kick_left": World.M_THEIR_CORNER_KICK, "goal_kick_left": World.M_THEIR_GOAL_KICK,
            "free_kick_left": World.M_THEIR_FREE_KICK, "pass_left": World.M_THEIR_PASS,
            "direct_free_kick_left": World.M_THEIR_DIR_FREE_KICK, "Goal_Left": World.M_THEIR_GOAL,
            "offside_left": World.M_THEIR_OFFSIDE, "KickOff_Right": World.M_OUR_KICKOFF,
            "KickIn_Right": World.M_OUR_KICK_IN, "corner_kick_right": World.M_OUR_CORNER_KICK,
            "goal_kick_right": World.M_OUR_GOAL_KICK, "free_kick_right": World.M_OUR_FREE_KICK,
            "pass_right": World.M_OUR_PASS, "direct_free_kick_right": World.M_OUR_DIR_FREE_KICK,
            "Goal_Right": World.M_OUR_GOAL, "offside_right": World.M_OUR_OFFSIDE,
            "BeforeKickOff": World.M_BEFORE_KICKOFF, "GameOver": World.M_GAME_OVER, "PlayOn": World.M_PLAY_ON
        }

    def find_non_digit(self, start):
        """
        从指定位置开始，查找第一个非数字且非小数点的字符位置。
        
        :param start: 起始位置
        :return: 非数字字符的位置
        """
        while True:
            if (self.exp[start] < ord('0') or self.exp[start] > ord('9')) and self.exp[start] != ord('.'):
                return start
            start += 1

    def find_char(self, start, char):
        """
        从指定位置开始，查找第一个等于指定字符的位置。
        
        :param start: 起始位置
        :param char: 目标字符
        :return: 目标字符的位置
        """
        while True:
            if self.exp[start] == char:
                return start
            start += 1

    def read_float(self, start):
        """
        从指定位置开始解析一个浮点数。
        
        :param start: 起始位置
        :return: 解析的浮点数和结束位置
        """
        if self.exp[start:start + 3] == b'nan':  # 处理 NaN 值
            return float('nan'), start + 3
        end = self.find_non_digit(start + 1)  # 查找非数字字符
        try:
            retval = float(self.exp[start:end])  # 尝试解析为浮点数
        except:
            self.world.log(f"{self.LOG_PREFIX}String to float conversion failed: {self.exp[start:end]} at msg[{start},{end}], \nMsg: {self.exp.decode()}")
            retval = 0  # 解析失败时返回 0
        return retval, end

    def read_int(self, start):
        """
        从指定位置开始解析一个整数。
        
        :param start: 起始位置
        :return: 解析的整数和结束位置
        """
        end = self.find_non_digit(start + 1)  # 查找非数字字符
        return int(self.exp[start:end]), end

    def read_bytes(self, start):
        """
        从指定位置开始解析一个字节序列，直到遇到空格或右括号。
        
        :param start: 起始位置
        :return: 解析的字节序列和结束位置
        """
        end = start
        while True:
            if self.exp[end] == ord(' ') or self.exp[end] == ord(')'):
                break
            end += 1
        return self.exp[start:end], end

    def read_str(self, start):
        """
        从指定位置开始解析一个字符串（字节序列解码为 UTF-8）。
        
        :param start: 起始位置
        :return: 解析的字符串和结束位置
        """
        b, end = self.read_bytes(start)  # 调用 read_bytes 获取字节序列
        return b.decode(), end

    def get_next_tag(self, start):
        """
        从指定位置开始解析下一个标签（例如 XML 格式的 <tag>）。
        
        :param start: 起始位置
        :return: 标签名称、结束位置和最小嵌套深度
        """
        min_depth = self.depth
        while True:
            if self.exp[start] == ord(")"):  # 监测嵌套深度
                self.depth -= 1
                if min_depth > self.depth:
                    min_depth = self.depth
            elif self.exp[start] == ord("("):
                break
            start += 1
            if start >= len(self.exp):
                return None, start, 0
        self.depth += 1  # 增加嵌套深度
        start += 1  # 跳过左括号
        end = self.find_char(start, ord(" "))  # 查找标签名称的结束位置
        return self.exp[start:end], end, min_depth  # 返回标签名称、结束位置和最小嵌套深度

    def parse(self, exp):
        """
        主解析函数，用于解析输入的字节流并更新 World 对象的状态。
        
        :param exp: 输入的字节流，包含环境信息
        """
        self.exp = exp  # 当前解析的字节流
        self.depth = 0  # 初始化嵌套深度
        self.world.step += 1  # 增加时间步
        self.world.line_count = 0  # 重置线条计数器
        self.world.robot.frp = dict()  # 重置机器人接触点信息
        self.world.flags_posts = dict()  # 重置标志杆信息
        self.world.flags_corners = dict()  # 重置角旗信息
        self.world.vision_is_up_to_date = False  # 视觉信息未更新
        self.world.ball_is_visible = False  # 球不可见
        self.world.robot.feet_toes_are_touching = dict.fromkeys(self.world.robot.feet_toes_are_touching, False)  # 重置脚趾接触状态
        self.world.time_local_ms += World.STEPTIME_MS  # 更新本地时间

        # 重置队友和对手的可见状态
        for p in self.world.teammates:
            p.is_visible = False
        for p in self.world.opponents:
            p.is_visible = False

        # 从字节流中获取第一个标签
        tag, end, _ = self.get_next_tag(0)

        # 遍历字节流，直到解析完成
        while end < len(exp):
            # 根据标签类型进行解析
            if tag == b'time':
                """
                时间信息标签，包含服务器时间等。
                """
                while True:
                    tag, end, min_depth = self.get_next_tag(end)
                    if min_depth == 0:  # 如果嵌套深度为 0，退出当前标签解析
                        break

                    if tag == b'now':
                        # 解析服务器当前时间
                        self.world.time_server, end = self.read_float(end + 1)
                    else:
                        # 记录未知标签
                        self.world.log(f"{self.LOG_PREFIX}Unknown tag inside 'time': {tag} at {end}, \nMsg: {exp.decode()}")

            elif tag == b'GS':
                """
                比赛状态标签，包含比赛模式、比分、球队信息等。
                """
                while True:
                    tag, end, min_depth = self.get_next_tag(end)
                    if min_depth == 0:  # 如果嵌套深度为 0，退出当前标签解析
                        break

                    if tag == b'team':
                        # 解析球队信息，判断球队是否在左半场
                        aux, end = self.read_str(end + 1)
                        is_left = bool(aux == "left")
                        if self.world.team_side_is_left != is_left:
                            self.world.team_side_is_left = is_left
                            self.play_mode_to_id = self.LEFT_PLAY_MODE_TO_ID if is_left else self.RIGHT_PLAY_MODE_TO_ID
                            self.world.draw.set_team_side(not is_left)
                            self.world.team_draw.set_team_side(not is_left)
                    elif tag == b'sl':
                        # 解析左半场得分
                        if self.world.team_side_is_left:
                            self.world.goals_scored, end = self.read_int(end + 1)
                        else:
                            self.world.goals_conceded, end = self.read_int(end + 1)
                    elif tag == b'sr':
                        # 解析右半场得分
                        if self.world.team_side_is_left:
                            self.world.goals_conceded, end = self.read_int(end + 1)
                        else:
                            self.world.goals_scored, end = self.read_int(end + 1)
                    elif tag == b't':
                        # 解析比赛时间
                        self.world.time_game, end = self.read_float(end + 1)
                    elif tag == b'pm':
                        # 解析比赛模式
                        aux, end = self.read_str(end + 1)
                        if self.play_mode_to_id is not None:
                            self.world.play_mode = self.play_mode_to_id[aux]
                    else:
                        # 记录未知标签
                        self.world.log(f"{self.LOG_PREFIX}Unknown tag inside 'GS': {tag} at {end}, \nMsg: {exp.decode()}")

            elif tag == b'GYR':
                """
                陀螺仪数据标签，包含机器人旋转速度。
                """
                while True:
                    tag, end, min_depth = self.get_next_tag(end)
                    if min_depth == 0:  # 如果嵌套深度为 0，退出当前标签解析
                        break

                    if tag == b'rt':
                        # 解析陀螺仪数据
                        self.world.robot.gyro[1], end = self.read_float(end + 1)
                        self.world.robot.gyro[0], end = self.read_float(end + 1)
                        self.world.robot.gyro[2], end = self.read_float(end + 1)
                        self.world.robot.gyro[1] *= -1  # 调整坐标系
                    else:
                        # 记录未知标签
                        self.world.log(f"{self.LOG_PREFIX}Unknown tag inside 'GYR': {tag} at {end}, \nMsg: {exp.decode()}")

            elif tag == b'ACC':
                """
                加速度计数据标签，包含机器人加速度。
                """
                while True:
                    tag, end, min_depth = self.get_next_tag(end)
                    if min_depth == 0:  # 如果嵌套深度为 0，退出当前标签解析
                        break

                    if tag == b'a':
                        # 解析加速度计数据
                        self.world.robot.acc[1], end = self.read_float(end + 1)
                        self.world.robot.acc[0], end = self.read_float(end + 1)
                        self.world.robot.acc[2], end = self.read_float(end + 1)
                        self.world.robot.acc[1] *= -1  # 调整坐标系
                    else:
                        # 记录未知标签
                        self.world.log(f"{self.LOG_PREFIX}Unknown tag inside 'ACC': {tag} at {end}, \nMsg: {exp.decode()}")

            elif tag == b'HJ':
                """
                关节信息标签，包含机器人关节角度和速度。
                """
                while True:
                    tag, end, min_depth = self.get_next_tag(end)
                    if min_depth == 0:  # 如果嵌套深度为 0，退出当前标签解析
                        break

                    if tag == b'n':
                        # 解析关节名称
                        joint_name, end = self.read_str(end + 1)
                        joint_index = Robot.MAP_PERCEPTOR_TO_INDEX[joint_name]
                    elif tag == b'ax':
                        # 解析关节角度
                        joint_angle, end = self.read_float(end + 1)

                        # 修正关节角度（对称性问题）
                        if joint_name in Robot.FIX_PERCEPTOR_SET:
                            joint_angle = -joint_angle

                        old_angle = self.world.robot.joints_position[joint_index]
                        self.world.robot.joints_speed[joint_index] = (joint_angle - old_angle) / World.STEPTIME * math.pi / 180
                        self.world.robot.joints_position[joint_index] = joint_angle
                    else:
                        # 记录未知标签
                        self.world.log(f"{self.LOG_PREFIX}Unknown tag inside 'HJ': {tag} at {end}, \nMsg: {exp.decode()}")

            elif tag == b'FRP':
                """
                脚部接触点信息标签，包含接触点位置和力。
                """
                while True:
                    tag, end, min_depth = self.get_next_tag(end)
                    if min_depth == 0:  # 如果嵌套深度为 0，退出当前标签解析
                        break

                    if tag == b'n':
                        # 解析脚部或脚趾 ID
                        foot_toe_id, end = self.read_str(end + 1)
                        self.world.robot.frp[foot_toe_id] = foot_toe_ref = np.empty(6)
                        self.world.robot.feet_toes_last_touch[foot_toe_id] = self.world.time_local_ms
                        self.world.robot.feet_toes_are_touching[foot_toe_id] = True
                    elif tag == b'c':
                        # 解析接触点位置
                        foot_toe_ref[1], end = self.read_float(end + 1)
                        foot_toe_ref[0], end = self.read_float(end + 1)
                        foot_toe_ref[2], end = self.read_float(end + 1)
                        foot_toe_ref[1] *= -1  # 调整坐标系
                    elif tag == b'f':
                        # 解析接触点的力向量
                        foot_toe_ref[4], end = self.read_float(end + 1)
                        foot_toe_ref[3], end = self.read_float(end + 1)
                        foot_toe_ref[5], end = self.read_float(end + 1)
                        foot_toe_ref[4] *= -1  # 调整坐标系
                    else:
                        # 记录未知标签
                        self.world.log(f"{self.LOG_PREFIX}Unknown tag inside 'FRP': {tag} at {end}, \nMsg: {exp.decode()}")

            elif tag == b'See':
                """
                视觉信息标签，包含机器人看到的场景信息，如球、标志、队友和对手等。
                """
                self.world.vision_is_up_to_date = True  # 标记视觉信息已更新
                self.world.vision_last_update = self.world.time_local_ms  # 更新视觉信息的最后更新时间

                while True:
                    tag, end, min_depth = self.get_next_tag(end)
                    if min_depth == 0:  # 如果嵌套深度为 0，退出当前标签解析
                        break

                    tag_bytes = bytes(tag)  # 将标签转换为字节序列，用于字典查找

                    if tag in [b'G1R', b'G2R', b'G1L', b'G2L']:
                        # 解析球门标志
                        _, end, _ = self.get_next_tag(end)
                        c1, end = self.read_float(end + 1)
                        c2, end = self.read_float(end + 1)
                        c3, end = self.read_float(end + 1)

                        # 根据球队在左半场还是右半场，选择对应的标志位置
                        aux = self.LEFT_SIDE_FLAGS[tag_bytes] if self.world.team_side_is_left else self.RIGHT_SIDE_FLAGS[tag_bytes]
                        self.world.flags_posts[aux] = (c1, c2, c3)

                    elif tag in [b'F1R', b'F2R', b'F1L', b'F2L']:
                        # 解析角旗
                        _, end, _ = self.get_next_tag(end)
                        c1, end = self.read_float(end + 1)
                        c2, end = self.read_float(end + 1)
                        c3, end = self.read_float(end + 1)

                        aux = self.LEFT_SIDE_FLAGS[tag_bytes] if self.world.team_side_is_left else self.RIGHT_SIDE_FLAGS[tag_bytes]
                        self.world.flags_corners[aux] = (c1, c2, c3)

                    elif tag == b'B':
                        # 解析球的相对位置
                        _, end, _ = self.get_next_tag(end)
                        self.world.ball_rel_head_sph_pos[0], end = self.read_float(end + 1)
                        self.world.ball_rel_head_sph_pos[1], end = self.read_float(end + 1)
                        self.world.ball_rel_head_sph_pos[2], end = self.read_float(end + 1)

                        # 将球的球坐标转换为笛卡尔坐标
                        self.world.ball_rel_head_cart_pos = M.deg_sph2cart(self.world.ball_rel_head_sph_pos)
                        self.world.ball_is_visible = True  # 标记球可见
                        self.world.ball_last_seen = self.world.time_local_ms  # 更新球的最后可见时间

                    elif tag == b'mypos':
                        # 解析机器人自身绝对位置（作弊信息）
                        self.world.robot.cheat_abs_pos[0], end = self.read_float(end + 1)
                        self.world.robot.cheat_abs_pos[1], end = self.read_float(end + 1)
                        self.world.robot.cheat_abs_pos[2], end = self.read_float(end + 1)

                    elif tag == b'myorien':
                        # 解析机器人自身绝对朝向（作弊信息）
                        self.world.robot.cheat_ori, end = self.read_float(end + 1)

                    elif tag == b'ballpos':
                        # 解析球的绝对位置和速度（作弊信息）
                        c1, end = self.read_float(end + 1)
                        c2, end = self.read_float(end + 1)
                        c3, end = self.read_float(end + 1)

                        # 计算球的速度
                        self.world.ball_cheat_abs_vel[0] = (c1 - self.world.ball_cheat_abs_pos[0]) / World.VISUALSTEP
                        self.world.ball_cheat_abs_vel[1] = (c2 - self.world.ball_cheat_abs_pos[1]) / World.VISUALSTEP
                        self.world.ball_cheat_abs_vel[2] = (c3 - self.world.ball_cheat_abs_pos[2]) / World.VISUALSTEP

                        # 更新球的绝对位置
                        self.world.ball_cheat_abs_pos[0] = c1
                        self.world.ball_cheat_abs_pos[1] = c2
                        self.world.ball_cheat_abs_pos[2] = c3

                    elif tag == b'P':
                        # 解析球员信息
                        while True:
                            previous_depth = self.depth
                            previous_end = end
                            tag, end, min_depth = self.get_next_tag(end)
                            if min_depth < 2:  # 如果嵌套深度小于 2，退出当前标签解析
                                end = previous_end
                                self.depth = previous_depth
                                break

                            if tag == b'team':
                                # 解析球员所属球队
                                player_team, end = self.read_str(end + 1)
                                is_teammate = bool(player_team == self.world.team_name)
                                if self.world.team_name_opponent is None and not is_teammate:
                                    self.world.team_name_opponent = player_team  # 记录对手球队名称
                            elif tag == b'id':
                                # 解析球员编号
                                player_id, end = self.read_int(end + 1)
                                player = self.world.teammates[player_id - 1] if is_teammate else self.world.opponents[player_id - 1]
                                player.body_parts_cart_rel_pos = dict()  # 重置球员可见身体部位
                                player.is_visible = True  # 标记球员可见
                            elif tag in [b'llowerarm', b'rlowerarm', b'lfoot', b'rfoot', b'head']:
                                # 解析球员身体部位的位置
                                tag_str = tag.decode()
                                _, end, _ = self.get_next_tag(end)

                                c1, end = self.read_float(end + 1)
                                c2, end = self.read_float(end + 1)
                                c3, end = self.read_float(end + 1)

                                if is_teammate:
                                    self.world.teammates[player_id - 1].body_parts_sph_rel_pos[tag_str] = (c1, c2, c3)
                                    self.world.teammates[player_id - 1].body_parts_cart_rel_pos[tag_str] = M.deg_sph2cart((c1, c2, c3))
                                else:
                                    self.world.opponents[player_id - 1].body_parts_sph_rel_pos[tag_str] = (c1, c2, c3)
                                    self.world.opponents[player_id - 1].body_parts_cart_rel_pos[tag_str] = M.deg_sph2cart((c1, c2, c3))
                            else:
                                # 记录未知标签
                                self.world.log(f"{self.LOG_PREFIX}Unknown tag inside 'P': {tag} at {end}, \nMsg: {exp.decode()}")

                    elif tag == b'L':
                        # 解析场地线条信息
                        l = self.world.lines[self.world.line_count]

                        _, end, _ = self.get_next_tag(end)
                        l[0], end = self.read_float(end + 1)
                        l[1], end = self.read_float(end + 1)
                        l[2], end = self.read_float(end + 1)
                        _, end, _ = self.get_next_tag(end)
                        l[3], end = self.read_float(end + 1)
                        l[4], end = self.read_float(end + 1)
                        l[5], end = self.read_float(end + 1)

                        if np.isnan(l).any():
                            # 如果线条信息包含 NaN 值，记录日志
                            self.world.log(f"{self.LOG_PREFIX}Received field line with NaNs {l}")
                        else:
                            self.world.line_count += 1  # 接受场地线条信息

                    else:
                        # 记录未知标签
                        self.world.log(f"{self.LOG_PREFIX}Unknown tag inside 'see': {tag} at {end}, \nMsg: {exp.decode()}")

            elif tag == b'hear':
                """
                听觉信息标签，包含从队友接收到的消息。
                """
                team_name, end = self.read_str(end + 1)

                if team_name == self.world.team_name:  # 仅处理来自本队的消息
                    timestamp, end =                    timestamp, end = self.read_float(end + 1)

                    if self.exp[end + 1] == ord('s'):  # 判断消息是否来自自己
                        direction, end = "self", end + 5  # 如果是自己发送的消息，方向标记为 "self"
                    else:
                        direction, end = self.read_float(end + 1)  # 如果是队友发送的消息，解析方向信息

                    msg, end = self.read_bytes(end + 1)  # 解析消息内容
                    self.hear_callback(msg, direction, timestamp)  # 调用回调函数处理消息

                # 获取下一个标签
                tag, end, _ = self.get_next_tag(end)

            else:
                # 如果遇到未知的根标签，记录日志
                self.world.log(f"{self.LOG_PREFIX}Unknown root tag: {tag} at {end}, \nMsg: {exp.decode()}")
                # 继续解析下一个标签
                tag, end, min_depth = self.get_next_tag(end)

"""
1. World_Parser 类的作用
World_Parser 是一个解析器类，用于解析机器人足球比赛中的环境信息。它的主要作用是将从仿真环境或传感器获取的原始数据（通常是字节流或字符串）解析为结构化的信息，并更新全局状态对象 World。
主要功能：
数据解析：从输入的字节流中提取各种信息，包括时间、比赛状态、传感器数据、视觉信息和听觉信息。
状态更新：将解析后的信息更新到 World 对象中，以便机器人控制系统可以基于这些信息做出决策。
错误处理：在解析过程中，记录未知标签或解析失败的情况，便于调试和优化。
回调支持：支持对特定事件（如队友消息）的回调处理，增强系统的灵活性。
2. 类中各个函数的作用
2.1 初始化函数：__init__
Python
复制
def __init__(self, world: World, hear_callback):
作用：初始化 World_Parser 对象。
参数：
world：World 类型的对象，表示比赛的全局状态。
hear_callback：一个回调函数，用于处理从队友接收到的消息。
功能：
设置日志前缀 LOG_PREFIX，用于标识日志来源。
初始化 self.exp 和 self.depth，分别用于存储当前解析的字节流和嵌套深度。
定义标志位置信息（LEFT_SIDE_FLAGS 和 RIGHT_SIDE_FLAGS），用于根据比赛场地侧别调整标志位置。
定义比赛模式到 ID 的映射（LEFT_PLAY_MODE_TO_ID 和 RIGHT_PLAY_MODE_TO_ID），用于快速查找比赛模式对应的 ID。
2.2 辅助函数
这些函数用于解析基本数据类型（如浮点数、整数、字符串等）或查找特定字符。
2.2.1 find_non_digit
Python
复制
def find_non_digit(self, start):
作用：从指定位置开始，查找第一个非数字且非小数点的字符。
参数：start，起始位置。
返回值：非数字字符的位置。
功能：用于辅助解析浮点数或整数时，确定数字的结束位置。
2.2.2 find_char
Python
复制
def find_char(self, start, char):
作用：从指定位置开始，查找第一个等于指定字符的位置。
参数：
start：起始位置。
char：目标字符。
返回值：目标字符的位置。
功能：用于查找特定分隔符（如空格或右括号）。
2.2.3 read_float
Python
复制
def read_float(self, start):
作用：从指定位置开始解析一个浮点数。
参数：start，起始位置。
返回值：解析的浮点数和结束位置。
功能：解析浮点数，支持处理 NaN 值。如果解析失败，记录日志并返回默认值 0。
2.2.4 read_int
Python
复制
def read_int(self, start):
作用：从指定位置开始解析一个整数。
参数：start，起始位置。
返回值：解析的整数和结束位置。
功能：解析整数，直到遇到非数字字符。
2.2.5 read_bytes
Python
复制
def read_bytes(self, start):
作用：从指定位置开始解析一个字节序列，直到遇到空格或右括号。
参数：start，起始位置。
返回值：解析的字节序列和结束位置。
功能：用于提取标签名称或消息内容。
2.2.6 read_str
Python
复制
def read_str(self, start):
作用：从指定位置开始解析一个字符串（字节序列解码为 UTF-8）。
参数：start，起始位置。
返回值：解析的字符串和结束位置。
功能：调用 read_bytes 获取字节序列，并将其解码为字符串。
2.2.7 get_next_tag
Python
复制
def get_next_tag(self, start):
作用：从指定位置开始解析下一个标签（如 XML 格式的 <tag>）。
参数：start，起始位置。
返回值：标签名称、结束位置和最小嵌套深度。
功能：用于解析嵌套结构（如 XML 或括号表达式），并跟踪嵌套深度。
2.3 主解析函数：parse
Python
复制
def parse(self, exp):
作用：解析输入的字节流，并更新 World 对象的状态。
参数：exp，输入的字节流，包含环境信息。
功能：
初始化解析状态（如重置计数器、更新时间步）。
遍历输入字节流，逐个解析标签。
根据标签类型调用对应的处理逻辑（如时间信息、比赛状态、传感器数据等）。
在解析过程中，记录未知标签或解析失败的情况。
2.4 标签处理逻辑
parse 函数中根据不同的标签类型调用了不同的处理逻辑。以下是主要标签及其处理逻辑的总结：
2.4.1 时间信息（time 标签）
解析服务器时间和本地时间，更新 World 对象中的时间信息。
2.4.2 比赛状态（GS 标签）
解析比赛模式、比分、球队信息等，更新 World 对象中的比赛状态。
2.4.3 陀螺仪数据（GYR 标签）
解析机器人旋转速度，更新 World 对象中的陀螺仪数据。
2.4.4 加速度计数据（ACC 标签）
解析机器人加速度，更新 World 对象中的加速度计数据。
2.4.5 关节信息（HJ 标签）
解析关节角度和速度，更新 World 对象中的关节状态。
2.4.6 脚部接触点（FRP 标签）
解析脚部接触点的位置和力，更新 World 对象中的脚部接触信息。
2.4.7 视觉信息（See 标签）
解析机器人看到的球、标志、队友和对手的位置，更新 World 对象中的视觉信息。
2.4.8 听觉信息（hear 标签）
解析从队友接收到的消息，并通过回调函数处理。
3. 类的作用总结
World_Parser 类的作用是将复杂的环境信息（通常是字节流）解析为结构化的数据，并更新到 World 对象中。它通过一系列辅助函数和标签处理逻辑，实现了对不同数据类型的解析和状态更新。这种解析器是机器人控制系统的核心部分，确保机器人能够实时感知环境并做出决策。
4. 函数的作用总结
辅助函数：用于解析基本数据类型（如浮点数、整数、字符串）或查找特定字符，为解析器提供底层支持。
主解析函数（parse）：负责遍历输入字节流，逐个解析标签，并调用对应的处理逻辑。
标签处理逻辑：根据不同的标签类型，解析具体的数据内容，并更新 World 对象的状态。
"""