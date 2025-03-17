'''
Pose - 角度（以度为单位）指定的关节姿势。
注意：对于没有脚趾的机器人，脚趾位置会被忽略。

姿势可以控制所有关节，也可以只控制由 "indices" 变量定义的子集。
'''

import numpy as np
from world.World import World


class Poses:
    """
    Poses 类用于定义和管理机器人的各种姿势。
    每个姿势由一组关节角度（以度为单位）定义，可以控制所有关节或指定的关节子集。
    """

    def __init__(self, world: World) -> None:
        """
        初始化 Poses 类。

        :param world: World 类型的对象，表示比赛的全局状态。
        """
        self.world = world  # 全局状态对象
        self.tolerance = 0.05  # 容忍的角度误差（用于判断行为是否完成）

        '''
        添加新姿势的说明：
        1. 在以下字典中添加一个新条目，使用唯一的姿势名称。
        2. 完成。
        '''
        self.poses = {
            "Zero": (
                "中性姿势，包括头部",  # 描述
                False,  # 禁用自动头部朝向
                np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]),  # 关节索引
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -90, -90, 0, 0, 90, 90, 0, 0, 0, 0])  # 关节角度值
            ),
            "Zero_Legacy": (
                "中性姿势，包括头部，肘部可能导致碰撞（遗留姿势）",  # 描述
                False,  # 禁用自动头部朝向
                np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]),  # 关节索引
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -90, -90, 0, 0, 0, 0, 0, 0, 0, 0])  # 关节角度值
            ),
            "Zero_Bent_Knees": (
                "中性姿势，包括头部，膝盖弯曲",  # 描述
                False,  # 禁用自动头部朝向
                np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]),  # 关节索引
                np.array([0, 0, 0, 0, 0, 0, 30, 30, -60, -60, 30, 30, 0, 0, -90, -90, 0, 0, 90, 90, 0, 0, 0, 0])  # 关节角度值
            ),
            "Zero_Bent_Knees_Auto_Head": (
                "中性姿势，自动头部，膝盖弯曲",  # 描述
                True,  # 启用自动头部朝向
                np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]),  # 关节索引
                np.array([0, 0, 0, 0, 30, 30, -60, -60, 30, 30, 0, 0, -90, -90, 0, 0, 90, 90, 0, 0, 0, 0])  # 关节角度值
            ),
            "Fall_Back": (
                "向后倾斜双脚以摔倒",  # 描述
                True,  # 启用自动头部朝向
                np.array([10, 11]),  # 关节索引
                np.array([-20, -20])  # 关节角度值
            ),
            "Fall_Front": (
                "向前倾斜双脚以摔倒",  # 描述
                True,  # 启用自动头部朝向
                np.array([10, 11]),  # 关节索引
                np.array([45, 45])  # 关节角度值
            ),
            "Fall_Left": (
                "向左倾斜双腿以摔倒",  # 描述
                True,  # 启用自动头部朝向
                np.array([4, 5]),  # 关节索引
                np.array([-20, 20])  # 关节角度值
            ),
            "Fall_Right": (
                "向右倾斜双腿以摔倒",  # 描述
                True,  # 启用自动头部朝向
                np.array([4, 5]),  # 关节索引
                np.array([20, -20])  # 关节角度值
            ),
        }

        # 如果机器人类型不是 4，则移除脚趾关节（机器人 4 有脚趾关节）
        if world.robot.type != 4:
            for key, val in self.poses.items():
                idxs = np.where(val[2] >= 22)[0]  # 查找关节 22 和 23
                if len(idxs) > 0:
                    # 移除这些关节
                    self.poses[key] = (val[0], val[1], np.delete(val[2], idxs), np.delete(val[3], idxs))

    def get_behaviors_callbacks(self):
        """
        返回每个姿势行为的回调函数（内部使用）。

        实现说明：
        -----------
        使用默认参数的原因是因为 lambda 表达式会记住作用域和变量名。
        在循环中，作用域不会改变，变量名也不会改变。
        然而，当 lambda 被定义时，会评估默认参数。
        """
        return {
            key: (
                val[0],  # 描述
                val[1],  # 是否启用自动头部朝向
                lambda reset, key=key: self.execute(key),  # 执行函数
                lambda: True  # 准备状态检查函数（始终返回 True）
            )
            for key, val in self.poses.items()
        }

    def execute(self, name) -> bool:
        """
        执行指定姿势。

        :param name: 姿势的名称。
        :return: 布尔值，表示姿势是否完成。
        """
        _, _, indices, values = self.poses[name]  # 获取姿势的关节索引和角度值
        # 设置关节目标位置，并返回剩余步骤数
        remaining_steps = self.world.robot.set_joints_target_position_direct(indices, values, True, tolerance=self.tolerance)
        # 如果剩余步骤数为 -1，表示姿势已完成
        return bool(remaining_steps == -1)