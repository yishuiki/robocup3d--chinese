import os

class Server():
    """
    用于管理机器人足球比赛服务器配置的类。
    """
    def __init__(self, script) -> None:
        """
        初始化 Server 类。
        :param script: 脚本对象（在此代码中未使用）。
        """
        # 检查服务器配置文件的路径
        if os.path.isdir("/usr/local/share/rcssserver3d/"):
            self.source = "/usr/local/share/rcssserver3d/"
        elif os.path.isdir("/usr/share/rcssserver3d/"):
            self.source = "/usr/share/rcssserver3d/"
        else:
            raise FileNotFoundError("The server configuration files were not found!")

        # 定义可配置选项及其描述
        self.options = ["Official Config", "Penalty Shootout", "Soccer Rules", "Sync Mode", "Real Time", "Cheats", "Full Vision", "Add Noise", "25Hz Monitor"]
        self.descriptions = [
            "Configuration used in official matches",
            "Server's Penalty Shootout mode",
            "Play modes, automatic referee, etc.",
            "Synchronous communication between agents and server",
            "Real Time (or maximum server speed)",
            "Agent position & orientation, ball position",
            "See 360 deg instead of 120 deg (vertically & horizontally)",
            "Noise added to the position of visible objects",
            "25Hz Monitor (or 50Hz but RoboViz will show 2x the actual speed)"
        ]

        # 定义配置文件路径
        spark_f = os.path.expanduser("~/.simspark/spark.rb")
        naoneckhead_f = self.source + "rsg/agent/nao/naoneckhead.rsg"

        self.files = {
            "Penalty Shootout": self.source + "naosoccersim.rb",
            "Soccer Rules": self.source + "naosoccersim.rb",
            "Sync Mode": spark_f,
            "Real Time": self.source + "rcssserver3d.rb",
            "Cheats": naoneckhead_f,
            "Full Vision": naoneckhead_f,
            "Add Noise": naoneckhead_f,
            "25Hz Monitor": spark_f
        }
    def label(self, setting_name, t_on, t_off):
        """
        检查配置文件中的某个设置是否启用或禁用。
        :param setting_name: 设置名称。
        :param t_on: 启用时的文本。
        :param t_off: 禁用时的文本。
        """
        with open(self.files[setting_name], "r") as sources:
            content = sources.read()
            
        if t_on in content:
            self.values[setting_name] = "On"
        elif t_off in content:
            self.values[setting_name] = "Off"
        else:
            self.values[setting_name] = "Error"
    def read_config(self):
        """
        读取服务器配置文件并更新当前配置状态。
        """
        v = self.values = dict()  # 初始化配置状态字典

        print("Reading server configuration files...")

        # 检查每个配置选项的状态
        self.label("Penalty Shootout", "addSoccerVar('PenaltyShootout', true)", "addSoccerVar('PenaltyShootout', false)")
        self.label("Soccer Rules", "gameControlServer.initControlAspect('SoccerRuleAspect')", "#gameControlServer.initControlAspect('SoccerRuleAspect')")
        self.label("Real Time", "enableRealTimeMode = true", "enableRealTimeMode = false")
        self.label("Cheats", "setSenseMyPos true", "setSenseMyPos false")
        self.label("Full Vision", "setViewCones 360 360", "setViewCones 120 120")
        self.label("Add Noise", "addNoise true", "addNoise false")
        self.label("Sync Mode", "agentSyncMode = true", "agentSyncMode = false")
        self.label("25Hz Monitor", "monitorStep = 0.04", "monitorStep = 0.02")

        # 检查是否为官方配置
        is_official_config = (
            v["Penalty Shootout"] == "Off" and
            v["Soccer Rules"] == "On" and
            v["Real Time"] == "On" and
            v["Cheats"] == "Off" and
            v["Full Vision"] == "Off" and
            v["Add Noise"] == "On" and
            v["Sync Mode"] == "Off" and
            v["25Hz Monitor"] == "On"
        )
        v["Official Config"] = "On" if is_official_config else "Off"
    def change_config(self, setting_name, t_on, t_off, current_value=None, file=None):
        """
        修改配置文件中的某个设置。
        :param setting_name: 设置名称。
        :param t_on: 启用时的文本。
        :param t_off: 禁用时的文本。
        :param current_value: 当前值（默认从 self.values 中获取）。
        :param file: 配置文件路径（默认从 self.files 中获取）。
        """
        if current_value is None:
            current_value = self.values[setting_name]

        if file is None:
            file = self.files[setting_name]

        with open(file, "r") as sources:
            t = sources.read()

        if current_value == "On":
            t = t.replace(t_on, t_off, 1)
            print(f"Replacing  '{t_on}'  with  '{t_off}'  in  '{file}'")
        elif current_value == "Off":
            t = t.replace(t_off, t_on, 1)
            print(f"Replacing  '{t_off}'  with  '{t_on}'  in  '{file}'")
        else:
            print(setting_name, "was not changed because the value is unknown!")
        with open(file, "w") as sources:
            sources.write(t)
    def execute(self):
        """
        执行服务器配置管理。
        """
        while True:
            self.read_config()  # 读取当前配置

            # 将配置状态字典转换为列表
            values_list = [self.values[o] for o in self.options]
            
            print()
            UI.print_table([self.options, values_list, self.descriptions], ["Setting", "Value", "Description"], numbering=[True, False, False])
            choice = UI.read_int('Choose setting (ctrl+c to return): ', 0, len(self.options))
            opt = self.options[choice]

            prefix = ['sudo', 'python3', 'scripts/utils/Server.py', opt]

            if opt in self.files:
                suffix = [self.values[opt], self.files[opt]]

            # 根据用户选择修改配置
            if opt == "Penalty Shootout":
                subprocess.call([*prefix, "addSoccerVar('PenaltyShootout', true)", "addSoccerVar('PenaltyShootout', false)", *suffix])
            elif opt == "Soccer Rules":
                subprocess.call([*prefix, "gameControlServer.initControlAspect('SoccerRuleAspect')", "#gameControlServer.initControlAspect('SoccerRuleAspect')", *suffix])
            elif opt == "Sync Mode":
                self.change_config(opt, "agentSyncMode = true", "agentSyncMode = false")  # 不需要 sudo 权限
            elif opt == "Real Time":
                subprocess.call([*prefix, "enableRealTimeMode = true", "enableRealTimeMode = false", *suffix])
            elif opt == "Cheats":
                subprocess.call([*prefix, "setSenseMyPos true", "setSenseMyPos false", *suffix,
                                opt, "setSenseMyOrien true", "setSenseMyOrien false", *suffix,
                                opt, "setSenseBallPos true", "setSenseBallPos false", *suffix])
            elif opt == "Full Vision":
                subprocess.call([*prefix, "setViewCones 360 360", "setViewCones 120 120", *suffix])
            elif opt == "Add Noise":
                subprocess.call([*prefix, "addNoise true", "addNoise false", *suffix])
            elif opt == "25Hz Monitor":
                self.change_config(opt, "monitorStep = 0.04", "monitorStep = 0.02")  # 不需要 sudo 权限
            elif opt == "Official Config": 
                if self.values[opt] == "On":
                    print("The official configuration is already On!")
                else:  # 将所有选项设置为官方配置
                    subprocess.call([*prefix[:3],
                "Penalty Shootout", "addSoccerVar('PenaltyShootout', false)", "addSoccerVar('PenaltyShootout', true)", "Off", self.files["Penalty Shootout"],
                "Soccer Rules", "gameControlServer.initControlAspect('SoccerRuleAspect')", "#gameControlServer.initControlAspect('SoccerRuleAspect')", "Off", self.files["Soccer Rules"],
                "Sync Mode", "agentSyncMode = false", "agentSyncMode = true", "Off", self.files["Sync Mode"],
                "Real Time", "enableRealTimeMode = true", "enableRealTimeMode = false", "Off", self.files["Real Time"],
                "Cheats", "setSenseMyPos false", "setSenseMyPos true", "Off", self.files["Cheats"],
                "Cheats", "setSenseMyOrien false", "setSenseMyOrien true", "Off", self.files["Cheats"],
                "Cheats", "setSenseBallPos false", "setSenseBallPos true", "Off", self.files["Cheats"],
                "Full Vision", "setViewCones 120 120", "setViewCones 360 360", "Off", self.files["Full Vision"],
                "Add Noise", "addNoise true", "addNoise false", "Off", self.files["Add Noise"],
                "25Hz Monitor", "monitorStep = 0.04", "monitorStep = 0.02", "Off", self.files["25Hz Monitor"]])

# process with sudo privileges to change the configuration files
if __name__ == "__main__":
    import sys
    s = Server(None)

    # 如果直接运行此脚本，尝试从命令行参数中读取配置修改指令
    for i in range(1, len(sys.argv), 5):
        s.change_config(*sys.argv[i:i + 5])
else:
    import subprocess
    from scripts.commons.UI import UI
