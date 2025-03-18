from communication.World_Parser import World_Parser
from itertools import count
from select import select
from sys import exit
from world.World import World
import socket
import time

class Server_Comm():
    monitor_socket = None  # 监控套接字，由所有在同一线程上运行的代理共享

    def __init__(self, host: str, agent_port: int, monitor_port: int, unum: int, robot_type: int, team_name: str,
                 world_parser: World_Parser, world: World, other_players, wait_for_server=True) -> None:
        """
        初始化 Server_Comm 类
        :param host: 服务器 IP
        :param agent_port: Agent 端口
        :param monitor_port: 监控端口
        :param unum: 球衣号码
        :param robot_type: 机器人类型
        :param team_name: 队伍名称
        :param world_parser: 世界解析器
        :param world: 世界状态
        :param other_players: 其他球员
        :param wait_for_server: 是否等待服务器
        """
        self.BUFFER_SIZE = 8192  # 缓冲区大小
        self.rcv_buff = bytearray(self.BUFFER_SIZE)  # 接收缓冲区
        self.send_buff = []  # 发送缓冲区
        self.world_parser = world_parser  # 世界解析器
        self.unum = unum  # 球衣号码

        # 初始化时，不清楚我们是在左侧还是右侧
        self._unofficial_beam_msg_left = "(agent (unum " + str(unum) + ") (team Left) (move "
        self._unofficial_beam_msg_right = "(agent (unum " + str(unum) + ") (team Right) (move "
        self.world = world  # 世界状态

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建套接字

        if wait_for_server:  # 如果需要等待服务器
            print("Waiting for server at ", host, ":", agent_port, sep="", end=".", flush=True)
        while True:
            try:
                self.socket.connect((host, agent_port))  # 连接到服务器
                print(end=" ")
                break
            except ConnectionRefusedError:  # 如果连接被拒绝
                if not wait_for_server:
                    print("Server is down. Closing...")
                    exit()
                time.sleep(1)
                print(".", end="", flush=True)
        print("Connected agent", unum, self.socket.getsockname())

        self.send_immediate(b'(scene rsg/agent/nao/nao_hetero.rsg ' + str(robot_type).encode() + b')')
        self._receive_async(other_players, True)

        self.send_immediate(b'(init (unum ' + str(unum).encode() + b') (teamname ' + team_name.encode() + b'))')
        self._receive_async(other_players, False)

        # 重复以确保收到队伍信息
        for _ in range(3):
            # 通过改变同步顺序消除高级步骤（rcssserver3d 协议错误，通常出现在球员 11）
            self.send_immediate(b'(syn)')  # 如果这个同步不需要，服务器会丢弃它
            for p in other_players:
                p.scom.send_immediate(b'(syn)')
            for p in other_players:
                p.scom.receive()
            self.receive()

        if world.team_side_is_left is None:
            print("\nError: server did not return a team side! Check server terminal!")
            exit()

        # 如果监控套接字为空且监控端口不为空，则连接到监控端口
        if Server_Comm.monitor_socket is None and monitor_port is not None:
            print("Connecting to server's monitor port at ", host, ":", monitor_port, sep="", end=".", flush=True)
            Server_Comm.monitor_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            Server_Comm.monitor_socket.connect((host, monitor_port))
            print("Done!")

    def _receive_async(self, other_players, first_pass) -> None:
        """
        在初始化期间接收异步信息
        :param other_players: 其他球员
        :param first_pass: 是否是第一次传递
        """
        if not other_players:
            self.receive()
            return

        self.socket.setblocking(0)  # 设置为非阻塞模式
        if first_pass: print("Async agent", self.unum, "initialization", end="", flush=True)

        while True:
            try:
                print(".", end="", flush=True)
                self.receive()
                break
            except:
                pass
            for p in other_players:
                p.scom.send_immediate(b'(syn)')
            for p in other_players:
                p.scom.receive()

        self.socket.setblocking(1)  # 恢复为阻塞模式
        if not first_pass: print("Done!")

    def receive(self, update=True):
        """
        接收消息并解析
        :param update: 是否更新世界状态
        """
        for i in count():  # 解析所有消息并执行值更新，但只在最后执行一次重计算
            try:
                if self.socket.recv_into(self.rcv_buff, nbytes=4) != 4: raise ConnectionResetError()
                msg_size = int.from_bytes(self.rcv_buff[:4], byteorder='big', signed=False)
                if self.socket.recv_into(self.rcv_buff, nbytes=msg_size, flags=socket.MSG_WAITALL) != msg_size: raise ConnectionResetError()
            except ConnectionResetError:
                print("\nError: socket was closed by rcssserver3d!")
                exit()

            self.world_parser.parse(self.rcv_buff[:msg_size])  # 解析消息
            if len(select([self.socket], [], [], 0.0)[0]) == 0: break

        if update:
            if i == 1: self.world.log("Server_Comm.py: The agent lost 1 packet! Is syncmode enabled?")
            if i > 1: self.world.log(f"Server_Comm.py: The agent lost {i} consecutive packets! Is syncmode disabled?")
            self.world.update()

            if len(select([self.socket], [], [], 0.0)[0]) != 0:
                self.world.log("Server_Comm.py: Received a new packet while on world.update()!")
                self.receive()

    def send_immediate(self, msg: bytes) -> None:
        """
        立即发送消息
        :param msg: 消息内容
        """
        try:
            self.socket.send((len(msg)).to_bytes(4, byteorder='big') + msg)  # 添加消息长度到前 4 个字节
        except BrokenPipeError:
            print("\nError: socket was closed by rcssserver3d!")
            exit()

    def send(self) -> None:
        """
        发送所有已提交的消息
        """
        if len(select([self.socket], [], [], 0.0)[0]) == 0:
            self.send_buff.append(b'(syn)')
            self.send_immediate(b''.join(self.send_buff))
        else:
            self.world.log("Server_Comm.py: Received a new packet while thinking!")
        self.send_buff = []  # 清空缓冲区

    def commit(self, msg: bytes) -> None:
        """
        提交消息到发送缓冲区
        :param msg: 消息内容
        """
        assert type(msg) == bytes, "Message must be of type Bytes!"
        self.send_buff.append(msg)

    def commit_and_send(self, msg: bytes=b'') -> None:
        """
        提交并发送消息
        :param msg: 消息内容
        """
        self.commit(msg)
        self.send()

    def clear_buffer(self) -> None:
        """
        清空发送缓冲区
        """
        self.send_buff = []

    def commit_announcement(self, msg: bytes) -> None:
        """
        向场上的所有球员发送消息
        :param msg: 消息内容
        """
        assert len(msg) <= 20 and type(msg) == bytes
        self.commit(b'(say ' + msg + b')')

    def commit_pass_command(self) -> None:
        """
        发送传球命令
        条件：
        - 当前比赛模式为 PlayOn
        - 球员靠近球（默认 0.5 米）
        - 没有对手靠近球（默认 1 米）
        - 球处于静止状态（默认速度 <0.05 米/秒）
        - 自上次传球命令以来已过了一定时间
        """
        self.commit(b'(pass)')

    def commit_beam(self, pos2d, rot) -> None:
        """
        官方传送命令，可在比赛中使用
        此传送会受到噪声影响（除非在服务器配置中禁用）

        参数
        ----------
        pos2d : array_like
            绝对 2D 位置（负 X 始终是我们的半场，无论我们是哪边）
        rot : `int`/`float`
            球员角度（以度为单位，0 表示向前）
        """
        assert len(pos2d) == 2, "官方传送命令只接受 2D 位置!"
        self.commit(f"(beam {pos2d[0]} {pos2d[1]} {rot})".encode())

    def unofficial_beam(self, pos3d, rot) -> None:
        ''' 
        非官方传送命令 - 不能在正式比赛中使用
        
        参数
        ----------
        pos3d : array_like
            绝对 3D 位置（负 X 始终是我们的半场，无论我们是哪边）
        rot : `int`/`float`
            球员角度（以度为单位，0 表示向前）
        '''
        assert len(pos3d) == 3, "非官方传送命令只接受 3D 位置!"

        # 无需对角度进行归一化，服务器接受任意角度
        if self.world.team_side_is_left:
            msg = f"{self._unofficial_beam_msg_left}{pos3d[0]} {pos3d[1]} {pos3d[2]} {rot - 90}))".encode()
        else:
            msg = f"{self._unofficial_beam_msg_right}{-pos3d[0]} {-pos3d[1]} {pos3d[2]} {rot + 90}))".encode()

        self.monitor_socket.send((len(msg)).to_bytes(4, byteorder='big') + msg)

    def unofficial_kill_sim(self) -> None:
        ''' 非官方命令：终止模拟器 '''
        msg = b'(killsim)'
        self.monitor_socket.send((len(msg)).to_bytes(4, byteorder='big') + msg)

    def unofficial_move_ball(self, pos3d, vel3d=(0, 0, 0)) -> None:
        ''' 
        非官方命令：移动球
        信息：球的半径为 0.042 米

        参数
        ----------
        pos3d : array_like
            绝对 3D 位置（负 X 始终是我们的半场，无论我们是哪边）
        vel3d : array_like
            绝对 3D 速度（负 X 始终是我们的半场，无论我们是哪边）
        '''
        assert len(pos3d) == 3 and len(vel3d) == 3, "移动球需要 3D 位置和速度"

        if self.world.team_side_is_left:
            msg = f"(ball (pos {pos3d[0]} {pos3d[1]} {pos3d[2]}) (vel {vel3d[0]} {vel3d[1]} {vel3d[2]}))".encode()
        else:
            msg = f"(ball (pos {-pos3d[0]} {-pos3d[1]} {pos3d[2]}) (vel {-vel3d[0]} {-vel3d[1]} {vel3d[2]}))".encode()

        self.monitor_socket.send((len(msg)).to_bytes(4, byteorder='big') + msg)

    def unofficial_set_game_time(self, time_in_s: float) -> None:
        '''
        非官方命令：设置比赛时间
        示例：unofficial_set_game_time(68.78)

        参数
        ----------
        time_in_s : float
            比赛时间（以秒为单位）
        '''
        msg = f"(time {time_in_s})".encode()
        self.monitor_socket.send((len(msg)).to_bytes(4, byteorder='big') + msg)

    def unofficial_set_play_mode(self, play_mode: str) -> None:
        '''
        非官方命令：设置比赛模式
        示例：unofficial_set_play_mode("PlayOn")

        参数
        ----------
        play_mode : str
            比赛模式
        '''
        msg = f"(playMode {play_mode})".encode()
        self.monitor_socket.send((len(msg)).to_bytes(4, byteorder='big') + msg)

    def unofficial_kill_player(self, unum: int, team_side_is_left: bool) -> None:
        '''
        非官方命令：终止特定球员

        参数
        ----------
        unum : int
            球衣号码
        team_side_is_left : bool
            如果要终止的球员属于左队，则为 True
        '''
        msg = f"(kill (unum {unum}) (team {'Left' if team_side_is_left else 'Right'}))".encode()
        self.monitor_socket.send((len(msg)).to_bytes(4, byteorder='big') + msg)

    def close(self, close_monitor_socket=False):
        ''' 关闭代理套接字，可选地关闭监控套接字（由在同一线程上运行的球员共享） '''
        self.socket.close()
        if close_monitor_socket and Server_Comm.monitor_socket is not None:
            Server_Comm.monitor_socket.close()
            Server_Comm.monitor_socket = None