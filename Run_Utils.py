# 主函数入口
def main():
    # 导入自定义的 Script 类，用于初始化程序
    from scripts.commons.Script import Script
    script = Script()  # 初始化：加载配置文件，解析参数，构建 C++ 模块，并检查不一致情况

    # 使用本地版本的 StableBaselines3（如果需要）
    # 将 stable-baselines3 文件夹添加到 Python 的模块搜索路径中
    import sys
    from os.path import dirname, abspath, join
    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), "stable-baselines3"))

    # 导入其他需要的模块和工具
    from scripts.commons.UI import UI  # 自定义的用户界面工具
    from os.path import isfile, join, realpath, dirname
    from os import listdir, getcwd
    from importlib import import_module  # 动态导入模块

    # 获取当前工作目录的绝对路径
    _cwd = realpath(join(getcwd(), dirname(__file__)))

    # 定义 gyms 和 utils 模块的路径
    gyms_path = _cwd + "/scripts/gyms/"
    utils_path = _cwd + "/scripts/utils/"
    exclusions = ["__init__.py"]  # 排除不需要加载的文件

    # 动态加载 gyms 和 utils 文件夹中的 Python 文件
    # 过滤出以 .py 结尾的文件，并去掉后缀，同时排除指定的文件
    utils = sorted(
        [f[:-3] for f in listdir(utils_path) if isfile(join(utils_path, f)) and f.endswith(".py") and f not in exclusions],
        key=lambda x: (x != "Server", x)  # 确保 "Server" 模块优先
    )
    gyms = sorted(
        [f[:-3] for f in listdir(gyms_path) if isfile(join(gyms_path, f)) and f.endswith(".py") and f not in exclusions]
    )

    # 主循环：让用户选择要运行的脚本
    while True:
        # 使用 UI.print_table 显示一个表格，让用户选择脚本
        # 表格分为两列：utils 和 gyms
        _, col_idx, col = UI.print_table(
            [utils, gyms], 
            ["Demos & Tests & Utils", "Gyms"], 
            cols_per_title=[2, 1], 
            numbering=[True] * 2, 
            prompt='Choose script (ctrl+c to exit): '
        )

        # 根据用户选择的列，确定是 utils 还是 gyms 模块
        is_gym = False
        if col == 0:
            chosen = ("scripts.utils.", utils[col_idx])  # 选择的是 utils 模块
        elif col == 1:
            chosen = ("scripts.gyms.", gyms[col_idx])  # 选择的是 gyms 模块
            is_gym = True

        # 获取模块名并动态加载模块
        cls_name = chosen[1]
        mod = import_module(chosen[0] + chosen[1])

        # 如果选择的是 utils 模块
        if not is_gym:
            # 从 world.commons.Draw 和 agent.Base_Agent 导入工具
            from world.commons.Draw import Draw
            from agent.Base_Agent import Base_Agent

            # 创建模块的实例，并传入 script 对象
            obj = getattr(mod, cls_name)(script)

            try:
                obj.execute()  # 调用模块的 execute 方法
            except KeyboardInterrupt:
                print("\nctrl+c pressed, returning...\n")  # 捕获 Ctrl+C，返回主菜单

            # 清理资源
            Draw.clear_all()  # 清除所有绘图
            Base_Agent.terminate_all()  # 关闭所有服务器套接字
            script.players = []  # 清空玩家列表
            del obj  # 删除实例

        # 如果选择的是 gyms 模块
        else:
            # 导入 Train_Base 模块，用于处理模型加载和训练
            from scripts.commons.Train_Base import Train_Base

            # 提示用户检查服务器参数
            print("\nBefore using GYMS, make sure all server parameters are set correctly")
            print("(sync mode should be 'On', real time should be 'Off', cheats should be 'On', ...)")
            print("To change these parameters go to the previous menu, and select Server\n")
            print("Also, GYMS start their own servers, so don't run any server manually")

            # 内部循环：让用户选择 Train、Test 或 Retrain 操作
            while True:
                try:
                    # 显示选项表格
                    idx = UI.print_table([["Train", "Test", "Retrain"]], numbering=[True], prompt='Choose option (ctrl+c to return): ')[0]
                except KeyboardInterrupt:
                    print()  # 捕获 Ctrl+C，返回上一级菜单
                    break

                # 根据用户选择执行操作
                if idx == 0:
                    mod.Train(script).train(dict())  # 调用 Train 类的 train 方法
                else:
                    model_info = Train_Base.prompt_user_for_model()  # 提示用户输入模型信息
                    if model_info is not None and idx == 1:
                        mod.Train(script).test(model_info)  # 调用 Train 类的 test 方法
                    elif model_info is not None:
                        mod.Train(script).train(model_info)  # 调用 Train 类的 train 方法（重新训练）


# 程序入口
if __name__ == "__main__":
    try:
        main()  # 调用主函数
    except KeyboardInterrupt:
        print("\nctrl+c pressed, exiting...")  # 捕获 Ctrl+C，退出程序
        exit()