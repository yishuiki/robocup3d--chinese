import subprocess

class Server():
    def __init__(self, first_server_p, first_monitor_p, n_servers) -> None:
        try:
            import psutil
            self.check_running_servers(psutil, first_server_p, first_monitor_p, n_servers)
        except ModuleNotFoundError:
            print("Info: Cannot check if the server is already running, because the psutil module was not found")
            
        self.first_server_p = first_server_p  # 第一个服务器端口
        self.n_servers = n_servers  # 服务器数量
        self.rcss_processes = []  # 用于存储启动的服务器进程

        # 根据服务器数量选择启动命令
        cmd = "simspark" if n_servers == 1 else "rcssserver3d"
        for i in range(n_servers):
            # 启动服务器进程，并将标准输出和标准错误重定向到null，避免输出干扰
            self.rcss_processes.append(
                subprocess.Popen((f"{cmd} --agent-port {first_server_p+i} --server-port {first_monitor_p+i}").split(),
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=True)
            )

    def check_running_servers(self, psutil, first_server_p, first_monitor_p, n_servers):
        ''' 检查选定端口上是否已有服务器运行 '''
        found = False  # 标记是否发现冲突
        # 获取所有可能的服务器进程
        p_list = [p for p in psutil.process_iter() if p.cmdline() and p.name() in ["rcssserver3d","simspark"]]
        # 计算服务器端口范围
        range1 = (first_server_p, first_server_p  + n_servers)
        range2 = (first_monitor_p,first_monitor_p + n_servers)
        bad_processes = []  # 用于存储冲突的进程

        for p in p_list:  
            # 当前忽略只指定了一个端口的情况（不太常见）
            ports = [int(arg) for arg in p.cmdline()[1:] if arg.isdigit()]
            if len(ports) == 0:
                ports = [3100,3200]  # 默认服务器端口（不太可能更改）

            # 检查端口是否冲突
            conflicts = [str(port) for port in ports if (
                (range1[0] <= port < range1[1]) or (range2[0] <= port < range2[1]) )]

            if len(conflicts)>0:
                if not found:
                    print("\nThere are already servers running on the same port(s)!")
                    found = True
                bad_processes.append(p)
                print(f"Port(s) {','.join(conflicts)} already in use by \"{p.name()}\" (PID:{p.pid})")

        if found:
            print()
            while True:
                inp = input("Enter 'kill' to kill these processes or ctrl+c to abort: ")
                if inp == "kill":
                    for p in bad_processes:
                        p.kill()  # 终止冲突进程
                    return
            

    def kill(self):
        for p in self.rcss_processes:
            p.kill()  # 终止所有启动的服务器进程
        print(f"Killed {self.n_servers} rcssserver3d processes starting at port {self.first_server_p}")
