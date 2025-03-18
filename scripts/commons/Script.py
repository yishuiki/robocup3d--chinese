from os import path, listdir, getcwd, cpu_count
from os.path import join, realpath, dirname, isfile, isdir, getmtime
from scripts.commons.UI import UI
import __main__
import argparse, json, sys
import pickle
import subprocess


class Script():
    ROOT_DIR = path.dirname(path.dirname(realpath(join(getcwd(), dirname(__file__)))))  # 项目根目录

    def __init__(self, cpp_builder_unum=0) -> None:
        '''
        参数说明
        -----------------------
        - 要添加新参数，请编辑下面的信息
        - 修改下面的信息后，必须手动删除 config.json 文件
        - 在其他模块中，可以通过它们的 1 字母 ID 访问这些参数
        '''
        # 参数列表：1 字母 ID，描述，硬编码默认值
        self.options = {'i': ('Server Hostname/IP', 'localhost'),
                        'p': ('Agent Port',         '3100'),
                        'm': ('Monitor Port',       '3200'),
                        't': ('Team Name',          'FCPortugal'),
                        'u': ('Uniform Number',     '1'),
                        'r': ('Robot Type',         '1'),
                        'P': ('Penalty Shootout',   '0'),
                        'F': ('magmaFatProxy',      '0'),
                        'D': ('Debug Mode',         '1')}

        # 参数列表：1 字母 ID，数据类型，选择范围
        self.op_types = {'i': (str, None),
                         'p': (int, None),
                         'm': (int, None),
                         't': (str, None),
                         'u': (int, range(1, 12)),
                         'r': (int, [0, 1, 2, 3, 4]),
                         'P': (int, [0, 1]),
                         'F': (int, [0, 1]),
                         'D': (int, [0, 1])}

        '''
        参数说明结束
        '''

        self.read_or_create_config()

        # 推进帮助文本位置
        formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=52)
        parser = argparse.ArgumentParser(formatter_class=formatter)

        o = self.options
        t = self.op_types

        for id in self.options:  # 较短的 metavar 用于美观原因
            parser.add_argument(f"-{id}", help=f"{o[id][0]:30}[{o[id][1]:20}]", type=t[id][0], nargs='?', default=o[id][1], metavar='X', choices=t[id][1])

        self.args = parser.parse_args()

        if getattr(sys, 'frozen', False):  # 在二进制文件运行时禁用调试模式
            self.args.D = 0

        self.players = []  # 创建的球员列表

        Script.build_cpp_modules(exit_on_build=(cpp_builder_unum != 0 and cpp_builder_unum != self.args.u))

        if self.args.D:
            try:
                print(f"\nNOTE: for help run \"python {__main__.__file__} -h\"")
            except:
                pass

            columns = [[], [], []]
            for key, value in vars(self.args).items():
                columns[0].append(o[key][0])
                columns[1].append(o[key][1])
                columns[2].append(value)

            UI.print_table(columns, ["Argument","Default at /config.json","Active"], alignment=["<","^","^"])

    def read_or_create_config(self) -> None:
        '''
        读取或创建配置文件
        '''
        if not path.isfile('config.json'):  # 如果文件不存在，则保存硬编码的默认值
            with open("config.json", "w") as f:
                json.dump(self.options, f, indent=4)
        else:  # 加载用户定义的值（可能会被命令行参数覆盖）
            if path.getsize("config.json") == 0:  # 等待可能的写操作，当启动多个代理时
                from time import sleep
                sleep(1)
            if path.getsize("config.json") == 0:  # 1 秒后仍未写入则终止
                print("Aborting: 'config.json' is empty. Manually verify and delete if still empty.")
                exit()

            with open("config.json", "r") as f:
                self.options = json.loads(f.read())

    @staticmethod
    def build_cpp_modules(special_environment_prefix=[], exit_on_build=False):
        '''
        使用 Pybind11 在 /cpp 文件夹中构建 C++ 模块

        参数
        ----------
        special_environment_prefix : `list`
            运行给定命令的环境前缀
            用于为不同版本的 Python 解释器编译 C++ 模块（而不是默认版本）
            Conda 环境示例：['conda', 'run', '-n', 'myEnv']
            如果为空列表，则使用默认的 Python 解释器作为编译目标
        exit_on_build : bool
            如果需要构建的内容，则退出（以便每个队伍只有一个代理构建 C++ 模块）
        '''
        cpp_path = Script.ROOT_DIR + "/cpp/"
        exclusions = ["__pycache__"]

        cpp_modules = [d for d in listdir(cpp_path) if isdir(join(cpp_path, d)) and d not in exclusions]

        if not cpp_modules: return  # 没有模块需要构建

        python_cmd = f"python{sys.version_info.major}.{sys.version_info.minor}"  # "python3" 可能会选择错误的版本，这可以防止这种情况

        def init():
            print("--------------------------\nC++ modules:",cpp_modules)

            try:
                process = subprocess.Popen(special_environment_prefix + [python_cmd, "-m", "pybind11", "--includes"], stdout=subprocess.PIPE)
                (includes, err) = process.communicate()
                process.wait()
            except:
                print(f"Error while executing child program: '{python_cmd} -m pybind11 --includes'")
                exit()

            includes = includes.decode().rstrip()  # 去除尾部换行符（和其他空白字符）
            print("Using Pybind11 includes: '",includes,"'",sep="")
            return includes

        nproc = str(cpu_count())
        zero_modules = True

        for module in cpp_modules:
            module_path = join(cpp_path, module)

            # 如果没有 Makefile（典型分发情况），则跳过模块
            if not isfile(join(module_path, "Makefile")):
                continue

            # 在某些条件下跳过模块
            if isfile(join(module_path, module + ".so")) and isfile(join(module_path, module + ".c_info")):
                with open(join(module_path, module + ".c_info"), 'rb') as f:
                    info = pickle.load(f)
                if info == python_cmd:
                    code_mod_time = max(getmtime(join(module_path, f)) for f in listdir(module_path) if f.endswith(".cpp") or f.endswith(".h"))
                    bin_mod_time = getmtime(join(module_path, module + ".so"))
                    if bin_mod_time + 30 > code_mod_time:  # 为避免构建，设置 30 秒的余量（场景：我们解压 fcpy 项目，包括二进制文件，修改时间都相似）
                        continue

            # 初始化：打印信息并获取 Pybind11 包含文件
            if zero_modules:
                if exit_on_build:
                    print("There are C++ modules to build. This player is not allowed to build. Aborting.")
                    exit()
                zero_modules = False
                includes = init()

            # 构建模块
            print(f'{f"Building: {module}... ":40}',end='',flush=True)
            process = subprocess.Popen(['make', '-j' + nproc, 'PYBIND_INCLUDES=' + includes], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=module_path)
            (output, err) = process.communicate()
            exit_code = process.wait()
            if exit_code == 0:
                print("success!")
                with open(join(module_path, module + ".c_info"), "wb") as f:  # 保存 Python 版本
                    pickle.dump(python_cmd, f, protocol=4)  # 协议 4 与 Python 3.4 向后兼容
            else:
                print("Aborting! Building errors:")
                print(output.decode(), err.decode())
                exit()

        if not zero_modules:
            print("All modules were built successfully!\n--------------------------")

    def batch_create(self, agent_cls, args_per_player):
        ''' 批量创建代理 '''

        for a in args_per_player:
            self.players.append(agent_cls(*a))

    def batch_execute_agent(self, index: slice = slice(None)):
        ''' 
        正常执行代理（包括提交和发送）

        参数
        ----------
        index : slice
            代理的子集
            （例如：index=slice(1,2) 将选择第二个代理）
            （例如：index=slice(1,3) 将选择第二个和第三个代理）
            默认情况下，选择所有代理
        '''   
        for p in self.players[index]:
            p.think_and_send()

    def batch_execute_behavior(self, behavior, index: slice = slice(None)):
        '''
        执行行为

        参数
        ----------
        behavior : str
            要执行的行为名称
        index : slice
            代理的子集
            （例如：index=slice(1,2) 将选择第二个代理）
            （例如：index=slice(1,3) 将选择第二个和第三个代理）
            默认情况下，选择所有代理
        '''
        for p in self.players[index]:
            p.behavior.execute(behavior)

    def batch_commit_and_send(self, index: slice = slice(None)):
        '''
        提交并发送数据到服务器

        参数
        ----------
        index : slice
            代理的子集
            （例如：index=slice(1,2) 将选择第二个代理）
            （例如：index=slice(1,3) 将选择第二个和第三个代理）
            默认情况下，选择所有代理
        '''
        for p in self.players[index]:
            p.scom.commit_and_send(p.world.robot.get_command())

    def batch_receive(self, index: slice = slice(None), update=True):
        ''' 
        等待服务器消息

        参数
        ----------
        index : slice
            代理的子集
            （例如：index=slice(1,2) 将选择第二个代理）
            （例如：index=slice(1,3) 将选择第二个和第三个代理）
            默认情况下，选择所有代理
        update : bool
            根据从服务器收到的信息更新世界状态
            如果为 False，则代理将对其自身及其周围环境一无所知
            这在演示中用于减少虚拟代理的 CPU 资源时非常有用
        '''
        for p in self.players[index]:
            p.scom.receive(update)

    def batch_commit_beam(self, pos2d_and_rotation, index: slice = slice(None)):
        '''
        将所有球员传送到 2D 位置并设置给定的旋转

        参数
        ----------
        pos2d_and_rotation : `list`
            2D 位置和旋转的可迭代列表，例如 [(0,0,45),(-5,0,90)]
        index : slice
            代理的子集
            （例如：index=slice(1,2) 将选择第二个代理）
            （例如：index=slice(1,3) 将选择第二个和第三个代理）
            默认情况下，选择所有代理
        '''        
        for p, pos_rot in zip(self.players[index], pos2d_and_rotation): 
            p.scom.commit_beam(pos_rot[0:2], pos_rot[2])

    def batch_unofficial_beam(self, pos3d_and_rotation, index: slice = slice(None)):
        '''
        将所有球员传送到 3D 位置并设置给定的旋转

        参数
        ----------
        pos3d_and_rotation : `list`
            3D 位置和旋转的可迭代列表，例如 [(0,0,0.5,45),(-5,0,0.5,90)]
        index : slice
            代理的子集
            （例如：index=slice(1,2) 将选择第二个代理）
            （例如：index=slice(1,3) 将选择第二个和第三个代理）
            默认情况下，选择所有代理
        '''        
        for p, pos_rot in zip(self.players[index], pos3d_and_rotation): 
            p.scom.unofficial_beam(pos_rot[0:3], pos_rot[3])

    def batch_terminate(self, index: slice = slice(None)):
        '''
        关闭连接到代理端口的所有套接字
        对于代理在应用程序结束前一直存活的脚本，此操作不是必需的

        参数
        ----------
        index : slice
            代理的子集
            （例如：index=slice(1,2) 将选择第二个代理）
            （例如：index=slice(1,3) 将选择第二个和第三个代理）
            默认情况下，选择所有代理
        '''
        for p in self.players[index]:
            p.terminate()
        del self.players[index]  # 删除选定的代理
