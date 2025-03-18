from scripts.commons.Script import Script
script = Script()  # Initialize: load config file, parse arguments, build cpp modules
# 初始化：加载配置文件，解析参数，构建 C++ 模块
a = script.args

from agent.Agent import Agent

# Args: Server IP, Agent Port, Monitor Port, Uniform No., Team name, Enable Log, Enable Draw
# 参数：服务器 IP、Agent 端口、监控端口、球衣号码、队伍名称、启用日志、启用绘图
team_args = ((a.i, a.p, a.m, u, a.t, True, True) for u in range(1, 12))
# 生成每个球员的参数元组，球衣号码从 1 到 11
script.batch_create(Agent, team_args)
# 批量创建 Agent 实例

while True:
    script.batch_execute_agent()
    # 批量执行 Agent 的操作
    script.batch_receive()
    # 批量接收 Agent 的反馈
