# 导入抽象方法装饰器，用于定义抽象基类中的抽象方法
from abc import abstractmethod

# 导入其他模块，这些模块提供了机器人代理所需的功能
from behaviors.Behavior import Behavior
from communication.Radio import Radio
from communication.Server_Comm import Server_Comm
from communication.World_Parser import World_Parser
from logs.Logger import Logger
from math_ops.Inverse_Kinematics import Inverse_Kinematics
from world.commons.Path_Manager import Path_Manager
from world.World import World

# 定义 Base_Agent 类，这是一个抽象基类，用于创建机器人代理
class Base_Agent():
    # 类变量，用于存储所有创建的 Base_Agent 实例，便于全局管理
    all_agents = []

    # 构造函数，初始化机器人代理的各个组件
    def __init__(self, host: str, agent_port: int, monitor_port: int, unum: int, robot_type: int, team_name: str, 
                 enable_log: bool = True, enable_draw: bool = True, apply_play_mode_correction: bool = True, 
                 wait_for_server: bool = True, hear_callback = None) -> None:
        """
        初始化机器人代理。

        参数:
            host (str): 服务器主机地址。
            agent_port (int): 代理通信端口。
            monitor_port (int): 监控端口。
            unum (int): 机器人编号（unique number）。
            robot_type (int): 机器人类型。
            team_name (str): 机器人所属队伍名称。
            enable_log (bool): 是否启用日志记录，默认为 True。
            enable_draw (bool): 是否启用绘图功能，默认为 True。
            apply_play_mode_correction (bool): 是否应用比赛模式修正，默认为 True。
            wait_for_server (bool): 是否等待服务器连接，默认为 True。
            hear_callback (callable): 可选的回调函数，用于处理接收到的消息。
        """
        # 初始化通信模块（Radio）为 None，因为 hear_message 方法可能在 Server_Comm 初始化期间被调用
        self.radio = None

        # 初始化日志记录器，记录代理的日志信息
        self.logger = Logger(enable_log, f"{team_name}_{unum}")

        # 初始化世界模型，包含机器人类型、队伍名称、编号等信息
        self.world = World(robot_type, team_name, unum, apply_play_mode_correction, enable_draw, self.logger, host)

        # 初始化世界解析器，用于解析世界模型中的信息
        self.world_parser = World_Parser(self.world, self.hear_message if hear_callback is None else hear_callback)

        # 初始化服务器通信模块，用于与服务器进行通信
        self.scom = Server_Comm(host, agent_port, monitor_port, unum, robot_type, team_name, self.world_parser, self.world, 
                                Base_Agent.all_agents, wait_for_server)

        # 初始化逆运动学模块，用于计算机器人的运动控制
        self.inv_kinematics = Inverse_Kinematics(self.world.robot)

        # 初始化行为管理模块，用于管理机器人的行为逻辑
        self.behavior = Behavior(self)

        # 初始化路径管理模块，用于规划机器人路径
        self.path_manager = Path_Manager(self.world)

        # 初始化通信模块（Radio），用于机器人之间的通信
        self.radio = Radio(self.world, self.scom.commit_announcement)

        # 创建行为逻辑（具体行为由子类实现）
        self.behavior.create_behaviors()

        # 将当前代理实例添加到全局管理列表中
        Base_Agent.all_agents.append(self)

    # 定义抽象方法，子类必须实现该方法
    @abstractmethod
    def think_and_send(self):
        """
        抽象方法，用于实现机器人的决策逻辑和消息发送。
        子类需要根据具体需求实现该方法。
        """
        pass

    # 处理接收到的消息
    def hear_message(self, msg: bytearray, direction, timestamp: float) -> None:
        """
        处理接收到的消息。

        参数:
            msg (bytearray): 接收到的消息内容。
            direction: 消息来源方向。
            timestamp (float): 消息的时间戳。
        """
        # 如果消息不是来自自身，并且通信模块（Radio）已初始化，则调用 Radio 的接收方法处理消息
        if direction != "self" and self.radio is not None:
            self.radio.receive(msg)

    # 终止代理实例
    def terminate(self):
        """
        终止代理实例。
        如果这是当前线程中最后一个代理实例，则关闭共享的监控套接字。
        """
        # 关闭服务器通信模块，如果这是最后一个代理实例，则关闭共享的监控套接字
        self.scom.close(close_monitor_socket=(len(Base_Agent.all_agents) == 1))

        # 从全局管理列表中移除当前代理实例
        Base_Agent.all_agents.remove(self)

    # 静态方法，终止所有代理实例
    @staticmethod
    def terminate_all():
        """
        终止所有代理实例。
        关闭所有代理的服务器通信模块，并清空全局管理列表。
        """
        # 遍历所有代理实例，关闭其服务器通信模块
        for o in Base_Agent.all_agents:
            o.scom.close(True)  # 关闭共享的监控套接字（如果存在）

        # 清空全局管理列表
        Base_Agent.all_agents = []