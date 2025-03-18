import numpy as np

# 注意：当看到其他机器人时，之前的所有身体部位位置信息将被删除。
# 例如：
#   - 在 0 秒时看到 5 个身体部位 -> body_parts_cart_rel_pos 包含 5 个元素
#   - 在 1 秒时看到 1 个身体部位 -> body_parts_cart_rel_pos 包含 1 个元素
class Other_Robot():
    """
    表示机器人足球比赛中其他机器人（队友或对手）的状态和信息。
    """
    def __init__(self, unum, is_teammate) -> None:
        """
        初始化 Other_Robot 类。
        :param unum: 球衣号码（方便标识机器人，等于其他机器人的索引 + 1）。
        :param is_teammate: 是否为队友。
        """
        self.unum = unum  # 球衣号码
        self.is_self = False  # 标志，表示该机器人是否为自身
        self.is_teammate = is_teammate  # 标志，表示该机器人是否为队友
        self.is_visible = False  # 如果该机器人在最近一次从服务器收到的消息中被看到，则为 True（但这并不意味着我们知道它的绝对位置）
        self.body_parts_cart_rel_pos = dict()  # 机器人可见身体部位的笛卡尔相对位置
        self.body_parts_sph_rel_pos = dict()  # 机器人可见身体部位的球面相对位置
        self.vel_filter = 0.3  # 应用于 self.state_filtered_velocity 的 EMA 滤波系数
        self.vel_decay = 0.95  # 每个视觉周期的速度衰减系数（如果速度被更新，则中和）

        # 状态变量：当该机器人可见且原始机器人能够自我定位时，这些变量会被计算
        self.state_fallen = False  # 如果机器人躺下，则为 True（当头部可见时更新）
        self.state_last_update = 0  # 状态最后一次更新时的世界时间（单位：毫秒）
        self.state_horizontal_dist = 0  # 如果头部可见，则为头部的水平距离；否则，为可见身体部位的平均水平距离（该距离通过视觉或无线电更新，即使其他机器人不可见，也会假设其最后位置）
        self.state_abs_pos = None  # 如果头部可见，则为头部的 3D 位置；否则，为可见身体部位的 2D 平均位置，或者为 2D 无线电头部位置
        self.state_orientation = 0  # 基于一对下臂或脚的朝向，或者两者的平均值（警告：可能比 state_last_update 更旧）
        self.state_ground_area = None  # 玩家区域在地面上的投影（圆形），如果距离超过 3 米则不精确（为了性能考虑），在机器人倒地时用于障碍物避让
        self.state_body_parts_abs_pos = dict()  # 每个身体部位的 3D 绝对位置
        self.state_filtered_velocity = np.zeros(3)  # 3D 滤波速度（单位：米/秒）（如果头部不可见，则更新 2D 部分，v.z 衰减）
