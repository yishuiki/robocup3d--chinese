from scripts.commons.Script import Script
script = Script()  # Initialize: load config file, parse arguments, build cpp modules
# 初始化：加载配置文件，解析参数，构建 C++ 模块
a = script.args

from agent.Agent import Agent

# Args: Server IP, Agent Port, Monitor Port, Uniform No., Team name, Enable Log, Enable Draw
# 参数：服务器 IP、Agent 端口、监控端口、球衣号码、队伍名称、启用日志、启用绘图
script.batch_create(Agent, ((a.i, a.p, a.m, a.u, a.t, True, True),))  # one player for home team
# 为本队创建一个球员
script.batch_create(Agent, ((a.i, a.p, a.m, a.u, "Opponent", True, True),))  # one player for away team
# 为对方队伍创建一个球员

while True:
    script.batch_execute_agent()
    # 批量执行 Agent 的操作
    script.batch_receive()
    # 批量接收 Agent 的反馈
