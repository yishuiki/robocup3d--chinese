from agent.Base_Agent import Base_Agent as Agent  # 导入基础代理类
from cpp.a_star import a_star  # 导入 A* 算法模块
from scripts.commons.Script import Script  # 导入脚本类
import numpy as np  # 导入 NumPy 库用于数学计算
import time  # 导入时间模块用于性能测试
'''
::::::::::::::::::::::::::::::::::::::::::
::::::::a_star.compute(param_vec):::::::::
::::::::::::::::::::::::::::::::::::::::::

param_vec (numpy array, float32)
param_vec[0] - 起点 x 坐标
param_vec[1] - 起点 y 坐标
param_vec[2] - 是否允许路径超出场地边界？（当球员没有球时很有用）
param_vec[3] - 是否前往对方球门？（路径会前往球门的最高效区域）
param_vec[4] - 目标 x 坐标（仅当 param_vec[3]==0 时使用）
param_vec[5] - 目标 y 坐标（仅当 param_vec[3]==0 时使用）
param_vec[6] - 超时时间（微秒，最大执行时间）
-------------- [可选] ----------------
param_vec[7-11]  - 障碍物 1: x, y, 硬半径（最大 5 米），软半径（最大 5 米），软半径的排斥力（最小值为 0）
param_vec[12-16] - 障碍物 2: x, y, 硬半径（最大 5 米），软半径（最大 5 米），软半径的排斥力（最小值为 0）
...               - 障碍物 n: x, y, 硬半径（最大 5 米），软半径（最大 5 米），软半径的排斥力（最小值为 0）
---------------- 返回值 ------------------
path_ret : numpy array (float32)
    path_ret[:-2]
        包含从起点到目标点的路径（最多 1024 个位置）
        每个位置由 x, y 坐标组成（因此最多有 2048 个坐标）
        返回向量是扁平的（一维）（例如：[x1,y1,x2,y2,x3,y3,...]）
        路径可能无法到达目标的原因：
            - 路径长度超过 1024 个位置（至少 102 米！）
            - 无法到达目标或超时（在这种情况下，路径结束于离目标最近的位置）
    path_ret[-2]
        数字表示路径状态
        0 - 成功
        1 - 在到达目标之前超时（可能无法到达）
        2 - 无法到达目标（所有选项都已尝试）
        3 - 起点和目标之间没有障碍物（path_ret[:-2] 只包含 2 个点：起点和目标）
    path_ret[-1]
        A* 路径成本
::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::注意事项:::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::

场地地图：
    - 该算法有一个 32 米 x 22 米的地图，精度为 10 厘米（与场地尺寸相同，外加 1 米边界）
    - 地图包含场地线、球门柱和球网的信息
    - 如果用户允许，路径可以超出场地边界（越界），但不能穿过球门柱或球网（这些被视为静态不可达障碍物）
    - 用户必须通过参数指定动态障碍物

排斥力：
    - 排斥力作为 A* 算法的额外成本实现
    - 横向移动 10 厘米的成本为 1，对角线移动的成本为 sqrt(2)
    - 在具有排斥力 f=1 的位置上移动的额外成本为 1
    - 对于场地上的任何给定位置，多个对象的排斥力通过 max 函数组合，而不是相加！
    - 如果路径从不可达位置开始，它可以移动到相邻的不可达位置，但成本为 100（以避免不可达路径）
    示例：
        地图 1   地图 2   地图 3
        ..x..   ..o..   ..o..
        ..1..   ..o..   .o1..
        ..o..   ..o..   ..o..
    考虑“地图 1”，其中“x”是目标，“o”是球员，“1”是排斥力为 1 的位置
    在“地图 2”中，球员选择向前移动，总成本为：1+(额外=1)+1 = 3
    在“地图 3”中，球员避开排斥力，总成本为：sqrt(2)+sqrt(2) = 2.83（最优解）
         地图 1     地图 2     地图 3     地图 4
        ...x...   ..oo...   ...o...   ...o...
        ..123..   .o123..   ..o23..   ..1o3..
        ...o...   ..oo...   ...o...   ...o...
    考虑“地图 1”，其中 3 个位置具有 3 种不同的排斥力，从 1 到 3。
    在“地图 2”中，球员避开所有排斥力，总成本为：1+sqrt(2)+sqrt(2)+1 = 4.83
    在“地图 3”中，球员通过最小的排斥力，总成本为：sqrt(2)+(额外=1)+sqrt(2) = 3.83（最优解）
    在“地图 4”中，球员选择向前移动，总成本为：1+(额外=2)+1 = 4.00

障碍物：
    硬半径：不可达障碍物半径（无限排斥力）
    软半径：可达障碍物半径，具有用户定义的排斥力（距离越远，排斥力越小）（如果 <= 硬半径，则禁用）
    示例：
        obstacle(0,0,1,3,5) -> 障碍物位于 (0,0)，硬半径为 1 米，软半径为 3 米，排斥力为 5
            - 路径不能距离此障碍物 <=1 米，除非路径从该半径内开始
            - 软半径的力在中心最大（5），随着距离增加而逐渐减弱，直到在距离障碍物 3 米处为 0
            - 总结：在 [0,1] 米范围内，力为无穷大；在 [1,3] 米范围内，力从 3.333 逐渐减弱到 0
        obstacle(-2.1,3,0,0,0) -> 障碍物位于 (-2.1,3)，硬半径为 0 米，软半径为 0 米，排斥力为 0
            - 路径不能穿过 (-2.1,3)
        obstacle(-2.16,3,0,0,8) -> 障碍物位于 (-2.2,3)，硬半径为 0 米，软半径为 0 米，排斥力为 8
            - 路径不能穿过 (-2.2,3)，地图精度为 10 厘米，因此障碍物放置在最近的有效位置
            - 排斥力被忽略，因为（软半径 <= 硬半径）
'''
class Pathfinding():
    def __init__(self, script: Script) -> None:
        """
        初始化路径规划类。
        :param script: 脚本对象，用于获取相关参数。
        """
        self.script = script
        # 初始化 A* 算法（虽然不是必须的，但首次运行会花费更多时间）
        a_star.compute(np.zeros(6, np.float32))

    def draw_grid(self):
        """
        绘制网格，用于可视化路径规划的成本和状态。
        """
        d = self.player.world.draw  # 获取绘图对象
        MAX_RAW_COST = 0.6  # 控球缓冲区的最大成本

        # 遍历场地网格
        for x in np.arange(-16, 16.01, 0.1):
            for y in np.arange(-11, 11.01, 0.1):
                # 计算路径成本（不允许越界）
                s_in, cost_in = a_star.compute(np.array([x, y, 0, 0, x, y, 5000], np.float32))[-2:]
                # 计算路径成本（允许越界）
                s_out, cost_out = a_star.compute(np.array([x, y, 1, 0, x, y, 5000], np.float32))[-2:]

                # 根据路径状态和成本绘制不同颜色的点
                if s_out != 3:  # 如果允许越界时路径状态不是“无障碍”
                    d.point((x, y), 5, d.Color.red, "grid", False)  # 绘制红色点
                elif s_in != 3:  # 如果不允许越界时路径状态不是“无障碍”
                    d.point((x, y), 4, d.Color.blue_pale, "grid", False)  # 绘制浅蓝色点
                elif 0 < cost_in < MAX_RAW_COST + 1e-6:  # 如果路径成本在合理范围内
                    # 根据成本值绘制渐变颜色的点
                    d.point((x, y), 4, d.Color.get(255, (1 - cost_in / MAX_RAW_COST) * 255, 0), "grid", False)
                elif cost_in > MAX_RAW_COST:  # 如果路径成本过高
                    d.point((x, y), 4, d.Color.black, "grid", False)  # 绘制黑色点

        # 刷新网格绘制
        d.flush("grid")

    def sync(self):
        """
        同步机器人状态，提交命令并接收反馈。
        """
        r = self.player.world.robot  # 获取机器人对象
        self.player.behavior.head.execute()  # 执行头部行为
        self.player.scom.commit_and_send(r.get_command())  # 提交并发送机器人命令
        self.player.scom.receive()  # 接收仿真环境的反馈

    def draw_path_and_obstacles(self, obst, path_ret_pb, path_ret_bp):
        """
        绘制路径和障碍物。
        :param obst: 障碍物信息。
        :param path_ret_pb: 球员到球的路径返回值。
        :param path_ret_bp: 球到球员的路径返回值。
        """
        w = self.player.world  # 获取世界对象

        # 绘制障碍物
        for i in range(0, len(obst[0]), 5):
            # 绘制硬半径区域（红色）
            w.draw.circle(obst[0][i:i + 2], obst[0][i + 2], 2, w.draw.Color.red, "obstacles", False)
            # 绘制软半径区域（橙色）
            w.draw.circle(obst[0][i:i + 2], obst[0][i + 3], 2, w.draw.Color.orange, "obstacles", False)

        # 提取路径信息
        path_pb = path_ret_pb[:-2]  # 球员到球的路径（去掉状态和成本）
        path_status_pb = path_ret_pb[-2]  # 路径状态
        path_cost_pb = path_ret_pb[-1]  # A* 成本

        path_bp = path_ret_bp[:-2]  # 球到球员的路径（去掉状态和成本）
        path_status_bp = path_ret_bp[-2]  # 路径状态
        path_cost_bp = path_ret_bp[-1]  # A* 成本

        # 根据路径状态选择颜色
        c_pb = {0: w.draw.Color.green_lime, 1: w.draw.Color.yellow, 2: w.draw.Color.red, 3: w.draw.Color.blue_light}[path_status_pb]
        c_bp = {0: w.draw.Color.green_pale, 1: w.draw.Color.yellow_light, 2: w.draw.Color.red_salmon, 3: w.draw.Color.blue_pale}[path_status_bp]

        # 绘制球员到球的路径
        for i in range(2, len(path_pb) - 2, 2):
            w.draw.line(path_pb[i - 2:i], path_pb[i:i + 2], 5, c_pb, "path_player_ball", False)
        if len(path_pb) >= 4:
            w.draw.arrow(path_pb[-4:-2], path_pb[-2:], 0.4, 5, c_pb, "path_player_ball", False)

        # 绘制球到球员的路径
        for i in range(2, len(path_bp) - 2, 2):
            w.draw.line(path_bp[i - 2:i], path_bp[i:i + 2], 5, c_bp, "path_ball_player", False)
        if len(path_bp) >= 4:
            w.draw.arrow(path_bp[-4:-2], path_bp[-2:], 0.4, 5, c_bp, "path_ball_player", False)

        # 刷新障碍物和路径绘制
        w.draw.flush("obstacles")
        w.draw.flush("path_player_ball")
        w.draw.flush("path_ball_player")

    def move_obstacles(self, obst):
        """
        移动障碍物。
        :param obst: 障碍物信息。
        """
        # 遍历每个障碍物
        for i in range(len(obst[0]) // 5):
            # 更新障碍物位置
            obst[0][i * 5] += obst[1][i, 0]
            obst[0][i * 5 + 1] += obst[1][i, 1]

            # 如果障碍物超出场地边界，反向移动
            if not -16 < obst[0][i * 5] < 16:
                obst[1][i, 0] *= -1
            if not -11 < obst[0][i * 5 + 1] < 11:
                obst[1][i, 1] *= -1

    def execute(self):
        """
        执行路径规划。
        """
        a = self.script.args  # 获取脚本参数
        self.player = Agent(a.i, a.p, a.m, a.u, a.r, a.t)  # 创建机器人代理
        w = self.player.world  # 获取世界对象
        r = self.player.world.robot  # 获取机器人对象
        timeout = 5000  # 设置路径规划超时时间（微秒）

        go_to_goal = 0  # 是否前往对方球门（0 表示不前往）
        obst_no = 50  # 障碍物数量
        # 初始化障碍物信息（位置、硬半径、软半径、排斥力）和随机速度
        obst = [[0, 0, 0.5, 1, 1] * obst_no, np.random.uniform(-0.01, 0.01, (obst_no, 2))]

        print("\nMove player/ball around using RoboViz!")
        print("Press ctrl+c to return.")
        print("\nPathfinding timeout set to", timeout, "us.")
        print("Pathfinding execution time:")

        # 绘制网格
        self.draw_grid()

        while True:
            # 获取球和球员的位置
            ball = w.ball_abs_pos[:2]
            rpos = r.loc_head_position[:2]

            # 移动障碍物
            self.move_obstacles(obst)

            # 构建路径规划参数（球员到球）
            param_vec_pb = np.array([*rpos, 1, go_to_goal, *ball, timeout, *obst[0]], np.float32)
            # 构建路径规划参数（球到球员）
            param_vec_bp = np.array([*ball, 0, go_to_goal, *rpos, timeout, *obst[0]], np.float32)

            # 计时并执行路径规划
            t1 = time.time()
            path_ret_pb = a_star.compute(param_vec_pb)
            t2 = time.time()
            path_ret_bp = a_star.compute(param_vec_bp)
            t3 = time.time()

            # 打印路径规划的执行时间和路径长度
            print(end=f"\rplayer->ball {int((t2-t1)*1000000):5}us (len:{len(path_ret_pb[:-2])//2:4})      ball->player {int((t3-t2)*1000000):5}us  (len:{len(path_ret_bp[:-2])//2:4}) ")

            # 绘制路径和障碍物
            self.draw_path_and_obstacles(obst, path_ret_pb, path_ret_bp)
            # 同步机器人状态
            self.sync()

