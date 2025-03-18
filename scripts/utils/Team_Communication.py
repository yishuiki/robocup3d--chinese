from agent.Base_Agent import Base_Agent as Agent
from itertools import count
from scripts.commons.Script import Script

'''
通信是如何工作的？
    say 命令允许球员向场上的每个人广播消息
    消息范围：50 米（场地对角线为 36 米，因此忽略此限制）
    hear 感知器指示 3 件事：
        - 消息内容
        - 消息来源队伍
        - 消息来源的绝对角度（如果是自己发送的消息，则设置为“self”）

    消息在下一步被听到。
    消息每 2 步（0.04 秒）发送一次。
    在静音步骤中发送的消息只能被自己听到。
    在一个时间步中，球员只能听到除自己之外的另一个球员的消息。
    如果两个其他球员同时说话，只有第一个消息会被听到。
    这种能力独立于双方队伍的消息。
    从理论上讲，球员可以听到自己的消息 + 第一个说话的队友的消息 + 第一个说话的对手的消息。
    实际上，对手的消息并不重要，因为我们的队伍解析器会忽略其他队伍的消息。

    消息特点：
        最多 20 个字符，ASCII 在 0x20 和 0x7E 之间，但不包括 ' '、'(' 和 ')'
        接受的字符：字母+数字+符号："!#$%&'*+,-./:;<=>?@[$$ ^_`{|}~"
        然而，由于服务器的一个错误，发送 ' 或 " 会使消息提前结束
'''

class Team_Communication():
    """
    团队通信类，用于演示球员之间的通信功能。
    """

    def __init__(self, script: Script) -> None:
        """
        初始化团队通信类。
        :param script: 脚本对象，用于获取相关参数。
        """
        self.script = script

    def player1_hear(self, msg: bytes, direction, timestamp: float) -> None:
        """
        球员 1 的 hear 回调函数，用于处理听到的消息。
        :param msg: 消息内容（字节类型）。
        :param direction: 消息来源方向。
        :param timestamp: 时间戳。
        """
        print(f"Player 1 heard: {msg.decode():20}  from:{direction:7}  timestamp:{timestamp}")

    def player2_hear(self, msg: bytes, direction, timestamp: float) -> None:
        """
        球员 2 的 hear 回调函数，用于处理听到的消息。
        :param msg: 消息内容（字节类型）。
        :param direction: 消息来源方向。
        :param timestamp: 时间戳。
        """
        print(f"Player 2 heard: {msg.decode():20}  from:{direction:7}  timestamp:{timestamp}")

    def player3_hear(self, msg: bytes, direction, timestamp: float) -> None:
        """
        球员 3 的 hear 回调函数，用于处理听到的消息。
        :param msg: 消息内容（字节类型）。
        :param direction: 消息来源方向。
        :param timestamp: 时间戳。
        """
        print(f"Player 3 heard: {msg.decode():20}  from:{direction:7}  timestamp:{timestamp}")

    def execute(self):
        """
        执行团队通信演示。
        """
        a = self.script.args  # 获取脚本参数

        # 定义每个球员的 hear 回调函数
        hear_callbacks = (self.player1_hear, self.player2_hear, self.player3_hear)

        # 创建球员代理
        self.script.batch_create(Agent, (
            (a.i, a.p, a.m, i + 1, 0, a.t, True, True, False, True, clbk) for i, clbk in enumerate(hear_callbacks)))
        p1: Agent = self.script.players[0]  # 球员 1
        p2: Agent = self.script.players[1]  # 球员 2
        p3: Agent = self.script.players[2]  # 球员 3

        # 将球员移动到指定位置
        self.script.batch_commit_beam([(-2, i, 45) for i in range(3)])

        # 主循环
        for i in count():
            # 构建消息内容
            msg1 = b"I_am_p1!_no:" + str(i).encode()
            msg2 = b"I_am_p2!_no:" + str(i).encode()
            msg3 = b"I_am_p3!_no:" + str(i).encode()
            # 提交消息
            p1.scom.commit_announcement(msg1)
            p2.scom.commit_announcement(msg2)
            p3.scom.commit_announcement(msg3)
            # 发送消息
            self.script.batch_commit_and_send()
            # 打印发送的消息内容
            print(f"Player 1 sent:  {msg1.decode()}      HEX: {' '.join([f'{m:02X}' for m in msg1])}")
            print(f"Player 2 sent:  {msg2.decode()}      HEX: {' '.join([f'{m:02X}' for m in msg2])}")
            print(f"Player 3 sent:  {msg3.decode()}      HEX: {' '.join([f'{m:02X}' for m in msg3])}")
            # 接收消息并更新世界状态
            self.script.batch_receive()
            # 等待用户按下回车键继续，或按 Ctrl+C 返回
            input("Press enter to continue or ctrl+c to return.")
