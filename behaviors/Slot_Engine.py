from math_ops.Math_Ops import Math_Ops as M
from os import listdir
from os.path import isfile, join
from world.World import World
import numpy as np
import xml.etree.ElementTree as xmlp

class Slot_Engine():
    """
    Slot_Engine 类用于管理机器人的槽行为（slot behaviors），这些行为通过 XML 文件定义。
    """

    def __init__(self, world: World) -> None:
        """
        初始化 Slot_Engine。

        参数：
        - world: World 类型，表示机器人的世界状态。
        """
        self.world = world
        self.state_slot_number = 0  # 当前槽行为的索引
        self.state_slot_start_time = 0  # 槽行为的开始时间
        self.state_slot_start_angles = None  # 槽行为开始时的关节角度
        self.state_init_zero = True  # 是否初始化为零

        # ------------- 解析槽行为

        dir = M.get_active_directory("/behaviors/slot/")  # 获取行为文件目录

        common_dir = f"{dir}common/"  # 公共行为目录
        files = [(f, join(common_dir, f)) for f in listdir(common_dir) if isfile(join(common_dir, f)) and f.endswith(".xml")]
        robot_dir = f"{dir}r{world.robot.type}"  # 机器人特定行为目录
        files += [(f, join(robot_dir, f)) for f in listdir(robot_dir) if isfile(join(robot_dir, f)) and f.endswith(".xml")]

        self.behaviors = dict()  # 存储槽行为
        self.descriptions = dict()  # 存储行为描述
        self.auto_head_flags = dict()  # 存储是否自动控制头部的标志

        for fname, file in files:
            robot_xml_root = xmlp.parse(file).getroot()  # 解析 XML 文件
            slots = []
            bname = fname[:-4]  # 去掉文件扩展名 ".xml"

            for xml_slot in robot_xml_root:
                assert xml_slot.tag == 'slot', f"在槽行为 {fname} 中发现意外的 XML 元素：'{xml_slot.tag}'"
                indices, angles = [], []

                for action in xml_slot:
                    indices.append(int(action.attrib['id']))  # 获取关节索引
                    angles.append(float(action.attrib['angle']))  # 获取关节角度

                delta_ms = float(xml_slot.attrib['delta']) * 1000  # 时间间隔（毫秒）
                assert delta_ms > 0, f"在槽行为 {fname} 中发现无效的 delta <= 0"
                slots.append((delta_ms, indices, angles))

            assert bname not in self.behaviors, f"发现至少两个槽行为具有相同的名称：{fname}"

            self.descriptions[bname] = robot_xml_root.attrib["description"] if "description" in robot_xml_root.attrib else bname
            self.auto_head_flags[bname] = (robot_xml_root.attrib["auto_head"] == "1")
            self.behaviors[bname] = slots


    def get_behaviors_callbacks(self):
        """
        返回每个槽行为的回调函数（内部使用）。

        实现说明：
        ---------------
        使用默认参数，因为 lambda 表达式会记住作用域和变量名。
        在循环中，作用域不会改变，变量名也不会改变。
        然而，默认参数在定义 lambda 时会被评估。
        """
        return {key: (self.descriptions[key], self.auto_head_flags[key],
                      lambda reset, key=key: self.execute(key, reset), lambda key=key: self.is_ready(key)) for key in self.behaviors}


    def is_ready(self, name) -> bool:
        """
        检查槽行为是否准备好在当前条件下开始。

        参数：
        - name: 槽行为的名称。

        返回：
        - bool 类型，如果行为准备好，则返回 True。
        """
        return True


    def reset(self, name):
        """
        初始化/重置槽行为。

        参数：
        - name: 槽行为的名称。
        """
        self.state_slot_number = 0  # 重置槽行为索引
        self.state_slot_start_time_ms = self.world.time_local_ms  # 更新开始时间
        self.state_slot_start_angles = np.copy(self.world.robot.joints_position)  # 复制当前关节角度
        assert name in self.behaviors, f"请求的槽行为不存在：{name}"


    def execute(self, name, reset) -> bool:
        """
        执行一步槽行为。

        参数：
        - name: 槽行为的名称。
        - reset: 是否重置行为。

        返回：
        - bool 类型，如果行为完成，则返回 True。
        """
        if reset:
            self.reset(name)  # 如果需要重置，则调用 reset 方法

        elapsed_ms = self.world.time_local_ms - self.state_slot_start_time_ms  # 计算经过的时间
        delta_ms, indices, angles = self.behaviors[name][self.state_slot_number]  # 获取当前槽的行为参数

        # 检查槽行为的进度
        if elapsed_ms >= delta_ms:
            self.state_slot_start_angles[indices] = angles  # 更新起始角度为上一个目标角度

            # 防止两种罕见情况：
            # 1 - 在行为完成且 reset=False 的情况下调用此函数
            # 2 - 我们在最后一个槽中，同步模式未激活，且丢失了最后一步
            if self.state_slot_number + 1 == len(self.behaviors[name]):
                return True  # 表示行为完成，直到通过参数发送重置信号

            self.state_slot_number += 1  # 移动到下一个槽
            elapsed_ms = 0
            self.state_slot_start_time_ms = self.world.time_local_ms  # 更新开始时间
            delta_ms, indices, angles = self.behaviors[name][self.state_slot_number]  # 获取下一个槽的行为参数

        # 执行行为
        progress = (elapsed_ms + 20) / delta_ms  # 计算进度
        target = (angles - self.state_slot_start_angles[indices]) * progress + self.state_slot_start_angles[indices]  # 计算目标角度
        self.world.robot.set_joints_target_position_direct(indices, target, False)  # 设置关节目标位置

        # 如果下一步（现在+20ms）超出范围，则返回 True，表示行为完成
        return bool(elapsed_ms + 20 >= delta_ms and self.state_slot_number + 1 == len(self.behaviors[name]))