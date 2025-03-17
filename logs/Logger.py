# 导入必要的模块
from pathlib import Path  # 用于处理文件路径和目录操作
from datetime import datetime  # 用于获取当前时间和日期
import random  # 用于生成随机数
from string import ascii_uppercase  # 包含所有大写英文字母（A-Z）

class Logger():
    _folder = None  # 类变量，用于存储日志文件夹路径，所有实例共享

    def __init__(self, is_enabled: bool, topic: str) -> None:
        """
        初始化 Logger 实例。

        参数:
            is_enabled (bool): 是否启用日志记录。
            topic (str): 日志的主题，通常用于区分不同的日志文件。
        """
        self.no_of_entries = 0  # 初始化日志条目数量为 0
        self.enabled = is_enabled  # 设置是否启用日志记录
        self.topic = topic  # 设置日志主题（文件名）

    def write(self, msg: str, timestamp: bool = True, step: int = None) -> None:
        """
        将日志信息写入文件。

        参数:
            msg (str): 要写入的日志信息。
            timestamp (bool): 是否在日志前添加时间戳，默认为 True。
            step (int): 模拟步骤信息，如果提供，会在日志前添加，默认为 None。
        """
        # 如果日志未启用，直接返回
        if not self.enabled:
            return

        # 如果日志文件夹尚未创建，则动态创建
        if Logger._folder is None:
            # 随机生成一个 6 位的大写字母字符串，用于防止多个进程写入同一个文件夹
            rnd = ''.join(random.choices(ascii_uppercase, k=6))
            # 构造日志文件夹路径，包含当前时间和随机字符串
            Logger._folder = "./logs/" + datetime.now().strftime("%Y-%m-%d_%H.%M.%S__") + rnd + "/"
            print("\nLogger Info: 日志文件夹已创建，路径为：", Logger._folder)
            # 创建日志文件夹，parents=True 表示创建父目录，exist_ok=True 表示如果目录已存在则不报错
            Path(Logger._folder).mkdir(parents=True, exist_ok=True)

        # 增加日志条目计数
        self.no_of_entries += 1

        # 构建日志前缀（包含时间戳和模拟步骤信息）
        prefix = ""
        write_step = step is not None  # 检查是否需要写入模拟步骤信息
        if timestamp or write_step:  # 如果需要时间戳或模拟步骤信息
            prefix = "{"  # 开始构建前缀
            if timestamp:  # 如果需要时间戳
                prefix += datetime.now().strftime("%a %H:%M:%S")  # 添加当前时间
                if write_step:  # 如果需要模拟步骤信息，添加一个空格
                    prefix += " "
            if write_step:  # 如果需要模拟步骤信息
                prefix += f'Step:{step}'  # 添加模拟步骤信息
            prefix += "} "  # 结束前缀构建

        # 打开日志文件并写入日志信息
        with open(Logger._folder + self.topic + ".log", 'a+') as f:  # 以追加模式打开日志文件
            f.write(prefix + msg + "\n")  # 写入日志内容