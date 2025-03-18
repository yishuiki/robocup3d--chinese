from scripts.commons.Script import Script
script = Script(cpp_builder_unum=1)  # Initialize: load config file, parse arguments, build cpp modules
# 初始化：加载配置文件，解析参数，构建 C++ 模块，设置 cpp_builder_unum 参数为 1
a = script.args

if a.P:  # penalty shootout
    # 如果是点球大战模式
    from agent.Agent_Penalty import Agent
    # 导入点球大战模式的 Agent 类
else:  # normal agent
    # 如果是普通模式
    from agent.Agent import Agent
    # 导入普通模式的 Agent 类

# Args: Server IP, Agent Port, Monitor Port, Uniform No., Team name, Enable Log, Enable Draw, Wait for Server, is magmaFatProxy
# 参数：服务器 IP、Agent 端口、监控端口、球衣号码、队伍名称、启用日志、启用绘图、等待服务器、是否为 magmaFatProxy
if a.D:  # debug mode
    # 如果是调试模式
    player = Agent(a.i, a.p, a.m, a.u, a.t, True, True, False, a.F)
    # 创建 Agent 实例，启用日志和绘图
else:
    player = Agent(a.i, a.p, None, a.u, a.t, False, False, False, a.F)
    # 创建 Agent 实例，禁用日志和绘图，监控端口设置为 None

while True:
    player.think_and_send()
    # Agent 思考并发送指令
    player.scom.receive()
    # Agent 接收服务器的反馈
