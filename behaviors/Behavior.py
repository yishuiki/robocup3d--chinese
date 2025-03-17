import numpy as np


class Behavior:
    """
    行为管理器类，用于管理和执行机器人的各种行为（如行走、踢球、起身等）。
    通过整合多种行为模块，提供统一的接口来执行和管理行为。
    """

    def __init__(self, base_agent):
        """
        初始化行为管理器。

        :param base_agent: Base_Agent 类型的对象，提供对全局状态（world）的访问。
        """
        from agent.Base_Agent import Base_Agent  # 用于类型提示
        self.base_agent: Base_Agent = base_agent  # 当前机器人的代理对象
        self.world = self.base_agent.world  # 全局状态对象

        # 初始化行为状态相关变量
        self.state_behavior_name = None  # 当前行为的名称
        self.state_behavior_init_ms = 0  # 当前行为的开始时间（毫秒）
        self.previous_behavior = None  # 上一个行为的名称
        self.previous_behavior_duration = None  # 上一个行为的持续时间（秒）

        # 初始化标准行为模块
        from behaviors.Poses import Poses  # 姿势行为模块
        from behaviors.Slot_Engine import Slot_Engine  # 槽引擎行为模块
        from behaviors.Head import Head  # 头部控制行为模块

        self.poses = Poses(self.world)  # 初始化姿势模块
        self.slot_engine = Slot_Engine(self.world)  # 初始化槽引擎模块
        self.head = Head(self.world)  # 初始化头部控制模块

    def create_behaviors(self):
        """
        创建行为字典，整合所有行为模块的回调函数。
        行为字典的格式：
            key: 行为名称
            value: (描述, 是否自动控制头部, 执行函数, 准备状态检查函数)
        """
        # 从姿势模块获取行为回调
        self.behaviors = self.poses.get_behaviors_callbacks()
        # 从槽引擎模块获取行为回调
        self.behaviors.update(self.slot_engine.get_behaviors_callbacks())
        # 从自定义行为模块获取行为回调
        self.behaviors.update(self.get_custom_callbacks())

    def get_custom_callbacks(self):
        """
        加载自定义行为模块，并生成回调函数。
        目前需要手动添加自定义行为的导入和注册。
        """
        # 手动声明自定义行为模块
        from behaviors.custom.Basic_Kick.Basic_Kick import Basic_Kick
        from behaviors.custom.Dribble.Dribble import Dribble
        from behaviors.custom.Fall.Fall import Fall
        from behaviors.custom.Get_Up.Get_Up import Get_Up
        from behaviors.custom.Step.Step import Step
        from behaviors.custom.Walk.Walk import Walk

        classes = [Basic_Kick, Dribble, Fall, Get_Up, Step, Walk]  # 自定义行为类列表

        # 创建自定义行为对象，并生成回调函数
        self.objects = {cls.__name__: cls(self.base_agent) for cls in classes}

        # 生成行为字典
        return {
            name: (
                o.description,  # 行为描述
                o.auto_head,  # 是否自动控制头部
                lambda reset, *args, o=o: o.execute(reset, *args),  # 执行函数
                lambda *args, o=o: o.is_ready(*args)  # 准备状态检查函数
            )
            for name, o in self.objects.items()
        }

    def get_custom_behavior_object(self, name):
        """
        根据名称获取自定义行为对象。

        :param name: 行为的名称
        :return: 自定义行为对象
        """
        assert name in self.objects, f"There is no custom behavior called {name}"
        return self.objects[name]

    def get_all_behaviors(self):
        """
        获取所有行为的名称和描述。

        :return: 行为名称列表和描述列表
        """
        names = [key for key in self.behaviors]
        descriptions = [val[0] for val in self.behaviors.values()]
        return names, descriptions

    def get_current(self):
        """
        获取当前行为的名称和持续时间（秒）。

        :return: 当前行为的名称和持续时间
        """
        duration = (self.world.time_local_ms - self.state_behavior_init_ms) / 1000.0
        return self.state_behavior_name, duration

    def get_previous(self):
        """
        获取上一个行为的名称和持续时间（秒）。

        :return: 上一个行为的名称和持续时间
        """
        return self.previous_behavior, self.previous_behavior_duration

    def force_reset(self):
        """
        强制重置当前行为，使得下一次执行时重新初始化。
        """
        self.state_behavior_name = None

    def execute(self, name, *args):
        """
        执行指定行为的一步。

        :param name: 行为的名称
        :param *args: 行为的参数
        :return: 布尔值，表示行为是否完成
        """
        assert name in self.behaviors, f"Behavior {name} does not exist!"

        # 检查是否需要重置行为（切换行为时自动重置）
        reset = bool(self.state_behavior_name != name)
        if reset:
            # 如果切换行为，记录上一个行为的信息
            if self.state_behavior_name is not None:
                self.previous_behavior = self.state_behavior_name
            self.previous_behavior_duration = (
                self.world.time_local_ms - self.state_behavior_init_ms
            ) / 1000.0
            self.state_behavior_name = name
            self.state_behavior_init_ms = self.world.time_local_ms

        # 如果行为允许，控制头部方向
        if self.behaviors[name][1]:
            self.head.execute()

        # 执行行为
        if not self.behaviors[name][2](reset, *args):
            return False  # 行为未完成

        # 行为完成
        self.previous_behavior = self.state_behavior_name
        self.state_behavior_name = None
        return True

    def execute_sub_behavior(self, name, reset, *args):
        """
        执行子行为的一步（手动控制重置）。

        :param name: 行为的名称
        :param reset: 是否重置行为
        :param *args: 行为的参数
        :return: 布尔值，表示行为是否完成
        """
        assert name in self.behaviors, f"Behavior {name} does not exist!"

        # 如果行为允许，控制头部方向
        if self.behaviors[name][1]:
            self.head.execute()

        # 执行行为
        return self.behaviors[name][2](reset, *args)

    def execute_to_completion(self, name, *args):
        """
        执行行为，直到完成或被中断。

        :param name: 行为的名称
        :param *args: 行为的参数
        """
        r = self.world.robot
        skip_last = name not in self.slot_engine.behaviors  # 是否忽略最后一个命令

        while True:
            done = self.execute(name, *args)
            if done and skip_last:
                break  # 如果最后一个命令无关紧要，则提前退出
            self.base_agent.scom.commit_and_send(r.get_command())  # 将命令发送到服务器
            self.base_agent.scom.receive()  # 接收服务器反馈
            if done:
                break  # 如果行为完成，则退出

        # 重置关节速度，避免污染下一个行为
        r.joints_target_speed = np.zeros_like(r.joints_target_speed)

    def is_ready(self, name, *args):
        """
        检查行为是否可以在当前条件下执行。

        :param name: 行为的名称
        :param *args: 行为的参数
        :return: 布尔值，表示行为是否准备好
        """
        assert name in self.behaviors, f"Behavior {name} does not exist!"
        return self.behaviors[name][3](*args)