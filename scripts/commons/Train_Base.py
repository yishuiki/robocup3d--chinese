from datetime import datetime, timedelta
from itertools import count
from os import listdir
from os.path import isdir, join, isfile
from scripts.commons.UI import UI
from datetime import datetime, timedelta  # 用于时间戳和时间间隔计算
from itertools import count  # 用于生成无限计数器
from os import listdir  # 用于列出目录中的文件和文件夹
from os.path import isdir, join, isfile  # 用于路径操作和检查
from scripts.commons.UI import UI  # 自定义用户界面工具
from shutil import copy  # 用于复制文件
from stable_baselines3 import PPO  # 稳定基线库中的 PPO 算法
from stable_baselines3.common.base_class import BaseAlgorithm  # 稳定基线库的基础算法类
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback  # 稳定基线库的回调工具
from typing import Callable  # 用于类型注解
from world.World import World  # 自定义的 World 类，用于模拟环境
from xml.dom import minidom  # 用于 XML 文件操作
import numpy as np  # 用于数值计算
import os, time, math, csv, select, sys  # 标准库，用于文件操作、时间、数学计算等
import pickle  # 用于序列化和反序列化 Python 对象
import xml.etree.ElementTree as ET  # 用于解析和生成 XML 文件

class Train_Base():
    def __init__(self, script) -> None:
        '''
        初始化训练环境的基本参数。
        当使用多环境训练时，服务器端口和监控端口会自动递增。
        '''
        '''
        When training with multiple environments (multiprocessing):
            The server port is incremented as follows:
                self.server_p, self.server_p+1, self.server_p+2, ...
            We add +1000 to the initial monitor port, so than we can have more than 100 environments:
                self.monitor_p+1000, self.monitor_p+1001, self.monitor_p+1002, ...
        When testing we use self.server_p and self.monitor_p
        '''
        args = script.args  # 获取脚本的命令行参数
        self.script = script  # 保存脚本对象
        self.ip = args.i  # 服务器 IP 地址
        self.server_p = args.p  # 初始服务器端口
        self.monitor_p = args.m  # 测试时的监控端口
        self.monitor_p_1000 = args.m + 1000  # 训练时的初始监控端口
        self.robot_type = args.r  # 机器人类型
        self.team = args.t  # 团队名称
        self.uniform = args.u  # 是否使用统一配置
        self.cf_last_time = 0  # 上一次控制帧率的时间戳
        self.cf_delay = 0  # 控制帧率的延迟
        self.cf_target_period = World.STEPTIME  # 目标模拟周期（默认实时）
        
    @staticmethod
    def prompt_user_for_model():
        '''
        提示用户选择一个模型文件夹和模型文件。
        返回模型文件的路径和相关信息。
        '''
        gyms_logs_path = "./scripts/gyms/logs/"  # 日志文件夹路径
        folders = [f for f in listdir(gyms_logs_path) if isdir(join(gyms_logs_path, f))]  # 列出所有文件夹
        folders.sort(key=lambda f: os.path.getmtime(join(gyms_logs_path, f)), reverse=True)  # 按修改时间排序

        while True:
            try:
                folder_name = UI.print_list(folders, prompt="Choose folder (ctrl+c to return): ")[1]  # 用户选择文件夹
            except KeyboardInterrupt:
                print()
                return None  # 如果用户按下 Ctrl+C，返回 None

            folder_dir = os.path.join(gyms_logs_path, folder_name)  # 文件夹路径
            models = [m[:-4] for m in listdir(folder_dir) if isfile(join(folder_dir, m)) and m.endswith(".zip")]  # 列出所有 .zip 模型文件

            if not models:
                print("The chosen folder does not contain any .zip file!")  # 如果文件夹中没有 .zip 文件，提示用户
                continue

            models.sort(key=lambda m: os.path.getmtime(join(folder_dir, m + ".zip")), reverse=True)  # 按修改时间排序

            try:
                model_name = UI.print_list(models, prompt="Choose model (ctrl+c to return): ")[1]  # 用户选择模型文件
                break
            except KeyboardInterrupt:
                print()

        return {
            "folder_dir": folder_dir,  # 模型文件夹路径
            "folder_name": folder_name,  # 文件夹名称
            "model_file": os.path.join(folder_dir, model_name + ".zip")  # 模型文件路径
        }

    def control_fps(self, read_input = False):
        ''' Add delay to control simulation speed '''

        if read_input:
            speed = input()
            if speed == '':
                self.cf_target_period = 0
                print(f"Changed simulation speed to MAX")
            else:
                if speed == '0':
                    inp = input("Paused. Set new speed or '' to use previous speed:")
                    if inp != '':
                        speed = inp   

                try:
                    speed = int(speed)
                    assert speed >= 0
                    self.cf_target_period = World.STEPTIME * 100 / speed
                    print(f"Changed simulation speed to {speed}%")
                except:
                    print("""Train_Base.py: 
    Error: To control the simulation speed, enter a non-negative integer.
    To disable this control module, use test_model(..., enable_FPS_control=False) in your gym environment.""")

        now = time.time()
        period = now - self.cf_last_time
        self.cf_last_time = now
        self.cf_delay += (self.cf_target_period - period)*0.9
        if self.cf_delay > 0:
            time.sleep(self.cf_delay)
        else:
            self.cf_delay = 0


    def test_model(self, model:BaseAlgorithm, env, log_path:str=None, model_path:str=None, max_episodes=0, enable_FPS_control=True, verbose=1):
        '''
        Test model and log results

        Parameters
        ----------
        model : BaseAlgorithm
            Trained model 
        env : Env
            Gym-like environment
        log_path : str
            Folder where statistics file is saved, default is `None` (no file is saved)
        model_path : str
            Folder where it reads evaluations.npz to plot it and create evaluations.csv, default is `None` (no plot, no csv)
        max_episodes : int
            Run tests for this number of episodes
            Default is 0 (run until user aborts)
        verbose : int
            0 - no output (except if enable_FPS_control=True)
            1 - print episode statistics
        '''

        if model_path is not None:
            assert os.path.isdir(model_path), f"{model_path} is not a valid path"
            self.display_evaluations(model_path)

        if log_path is not None:
            assert os.path.isdir(log_path), f"{log_path} is not a valid path"

            # If file already exists, don't overwrite
            if os.path.isfile(log_path + "/test.csv"):
                for i in range(1000):
                    p = f"{log_path}/test_{i:03}.csv"
                    if not os.path.isfile(p):
                        log_path = p
                        break
            else:
                log_path += "/test.csv"
            
            with open(log_path, 'w') as f:
                f.write("reward,ep. length,rew. cumulative avg., ep. len. cumulative avg.\n")
            print("Train statistics are saved to:", log_path)

        if enable_FPS_control: # control simulation speed (using non blocking user input)
            print("\nThe simulation speed can be changed by sending a non-negative integer\n"
                  "(e.g. '50' sets speed to 50%, '0' pauses the simulation, '' sets speed to MAX)\n")

        ep_reward = 0
        ep_length = 0
        rewards_sum = 0
        reward_min = math.inf
        reward_max = -math.inf
        ep_lengths_sum = 0
        ep_no = 0

        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_length += 1

            if enable_FPS_control: # control simulation speed (using non blocking user input)
                self.control_fps(select.select([sys.stdin], [], [], 0)[0]) 

            if done:
                obs = env.reset()
                rewards_sum += ep_reward
                ep_lengths_sum += ep_length
                reward_max = max(ep_reward, reward_max)
                reward_min = min(ep_reward, reward_min)
                ep_no += 1
                avg_ep_lengths = ep_lengths_sum/ep_no
                avg_rewards = rewards_sum/ep_no

                if verbose > 0:
                    print(  f"\rEpisode: {ep_no:<3}  Ep.Length: {ep_length:<4.0f}  Reward: {ep_reward:<6.2f}                                                             \n",
                        end=f"--AVERAGE--   Ep.Length: {avg_ep_lengths:<4.0f}  Reward: {avg_rewards:<6.2f}  (Min: {reward_min:<6.2f}  Max: {reward_max:<6.2f})", flush=True)
                
                if log_path is not None:
                    with open(log_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([ep_reward, ep_length, avg_rewards, avg_ep_lengths])
                
                if ep_no == max_episodes:
                    return

                ep_reward = 0
                ep_length = 0

    def learn_model(self, model:BaseAlgorithm, total_steps:int, path:str, eval_env=None, eval_freq=None, eval_eps=5, save_freq=None, backup_env_file=None, export_name=None):
        '''
        Learn Model for a specific number of time steps

        Parameters
        ----------
        model : BaseAlgorithm
            Model to train
        total_steps : int
            The total number of samples (env steps) to train on
        path : str
            Path where the trained model is saved
            If the path already exists, an incrementing number suffix is added
        eval_env : Env
            Environment to periodically test the model
            Default is None (no periodical evaluation)
        eval_freq : int
            Evaluate the agent every X steps
            Default is None (no periodical evaluation)
        eval_eps : int
            Evaluate the agent for X episodes (both eval_env and eval_freq must be defined)
            Default is 5
        save_freq : int
            Saves model at every X steps
            Default is None (no periodical checkpoint)
        backup_gym_file : str
            Generates backup of environment file in model's folder
            Default is None (no backup)
        export_name : str
            If export_name and save_freq are defined, a model is exported every X steps
            Default is None (no export)

        Returns
        -------
        model_path : str
            Directory where model was actually saved (considering incremental suffix)

        Notes
        -----
        If `eval_env` and `eval_freq` were specified:
            - The policy will be evaluated in `eval_env` every `eval_freq` steps
            - Evaluation results will be saved in `path` and shown at the end of training
            - Every time the results improve, the model is saved
        '''
        """
        训练模型指定时间步数
        参数
        model：BaseAlgorithm 类型
        要训练的模型。
        total_steps：int 类型
        训练的总样本数（环境步数）。
        path：str 类型
        训练完成的模型保存路径。如果路径已存在，则会自动添加递增的编号后缀。
        eval_env：Env 类型
        定期测试模型的环境。默认为 None（不进行定期评估）。
        eval_freq：int 类型
        每隔多少步评估一次智能体。默认为 None（不进行定期评估）。
        eval_eps：int 类型
        每次评估时运行的测试集数量（需要同时定义 eval_env 和 eval_freq）。默认为 5。
        save_freq：int 类型
        每隔多少步保存一次模型。默认为 None（不进行定期保存）。
        backup_gym_file：str 类型
        在模型文件夹中备份环境文件。默认为 None（不备份）。
        export_name：str 类型
        如果定义了 export_name 和 save_freq，则每隔 X 步导出一次模型。默认为 None（不导出）。
        返回值
        model_path：str 类型
        模型实际保存的目录（考虑递增后缀）。
        说明
        如果定义了 eval_env 和 eval_freq：
        策略将在每 eval_freq 步时在 eval_env 中进行评估。
        评估结果将保存在 path 中，并在训练结束时显示。
        每当结果有所改进时，模型将被保存。
        """
        start = time.time()
        start_date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        # 如果路径已存在，添加递增的编号后缀
        if os.path.isdir(path):
            for i in count():
                p = path.rstrip("/") + f"_{i:03}/"
                if not os.path.isdir(p):
                    path = p
                    break
        os.makedirs(path)

        # 备份环境文件
        if backup_env_file is not None:
            backup_file = os.path.join(path, os.path.basename(backup_env_file))
            copy(backup_env_file, backup_file)

        evaluate = bool(eval_env is not None and eval_freq is not None)

        # 创建评估回调
        eval_callback = (
            None
            if not evaluate
            else EvalCallback(
                eval_env,
                n_eval_episodes=eval_eps,
                eval_freq=eval_freq,
                log_path=path,
                best_model_save_path=path,
                deterministic=True,
                render=False,
            )
        )

        # 创建自定义回调
        custom_callback = (
            None if not evaluate else Cyclic_Callback(eval_freq, lambda: self.display_evaluations(path, True))
        )

        # 创建检查点回调
        checkpoint_callback = (
            None if save_freq is None else CheckpointCallback(save_freq=save_freq, save_path=path, name_prefix="model", verbose=1)
        )

        # 创建导出回调
        export_callback = (
            None if save_freq is None or export_name is None else Export_Callback(save_freq, path, export_name)
        )

        callbacks = CallbackList([c for c in [eval_callback, custom_callback, checkpoint_callback, export_callback] if c is not None])

        model.learn(total_timesteps=total_steps, callback=callbacks)
        model.save(os.path.join(path, "last_model"))

        # 显示评估结果
        if evaluate:
            self.display_evaluations(path)

        # 显示时间戳和模型路径
        end_date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        duration = timedelta(seconds=int(time.time() - start))
        print(f"Train start:     {start_date}")
        print(f"Train end:       {end_date}")
        print(f"Train duration:  {duration}")
        print(f"Model path:      {path}")

        # 将时间戳写入备份文件
        if backup_env_file is not None:
            with open(backup_file, "a") as f:
                f.write(f"\n# Train start:    {start_date}\n")
                f.write(f"# Train end:      {end_date}\n")
                f.write(f"# Train duration: {duration}")

        return path

    def display_evaluations(self, path, save_csv=False):

        eval_npz = os.path.join(path, "evaluations.npz")

        if not os.path.isfile(eval_npz):
            return

        console_width = 80
        console_height = 18
        symb_x = "\u2022"
        symb_o = "\u007c"
        symb_xo = "\u237f"

        with np.load(eval_npz) as data:
            time_steps = data["timesteps"]
            results_raw = np.mean(data["results"],axis=1)
            ep_lengths_raw = np.mean(data["ep_lengths"],axis=1)
        sample_no = len(results_raw)

        xvals = np.linspace(0, sample_no-1, 80)
        results    = np.interp(xvals, range(sample_no), results_raw)
        ep_lengths = np.interp(xvals, range(sample_no), ep_lengths_raw)

        results_limits    = np.min(results),    np.max(results)
        ep_lengths_limits = np.min(ep_lengths), np.max(ep_lengths)

        results_discrete    = np.digitize(results,    np.linspace(results_limits[0]-1e-5, results_limits[1]+1e-5,    console_height+1))-1
        ep_lengths_discrete = np.digitize(ep_lengths, np.linspace(0,                      ep_lengths_limits[1]+1e-5, console_height+1))-1

        matrix = np.zeros((console_height, console_width, 2), int)
        matrix[results_discrete[0]   ][0][0] = 1    # draw 1st column
        matrix[ep_lengths_discrete[0]][0][1] = 1    # draw 1st column
        rng = [[results_discrete[0], results_discrete[0]], [ep_lengths_discrete[0], ep_lengths_discrete[0]]]

        # Create continuous line for both plots
        for k in range(2):
            for i in range(1,console_width):
                x = [results_discrete, ep_lengths_discrete][k][i]
                if x > rng[k][1]:
                    rng[k] = [rng[k][1]+1, x]
                elif x < rng[k][0]:
                    rng[k] = [x, rng[k][0]-1]
                else:
                    rng[k] = [x,x]
                for j in range(rng[k][0],rng[k][1]+1):
                    matrix[j][i][k] = 1

        print(f'{"-"*console_width}')
        for l in reversed(range(console_height)):
            for c in range(console_width):
                if   np.all(matrix[l][c] == 0): print(end=" ")
                elif np.all(matrix[l][c] == 1): print(end=symb_xo)
                elif matrix[l][c][0] == 1:      print(end=symb_x)
                else:                           print(end=symb_o)
            print()
        print(f'{"-"*console_width}')
        print(f"({symb_x})-reward          min:{results_limits[0]:11.2f}    max:{results_limits[1]:11.2f}")
        print(f"({symb_o})-ep. length      min:{ep_lengths_limits[0]:11.0f}    max:{ep_lengths_limits[1]:11.0f}    {time_steps[-1]/1000:15.0f}k steps")
        print(f'{"-"*console_width}')

        # save CSV
        if save_csv:
            eval_csv = os.path.join(path, "evaluations.csv")
            with open(eval_csv, 'a+') as f:
                writer = csv.writer(f)
                if sample_no == 1:
                    writer.writerow(["time_steps", "reward ep.", "length"])
                writer.writerow([time_steps[-1],results_raw[-1],ep_lengths_raw[-1]])


    def generate_slot_behavior(self, path, slots, auto_head:bool, XML_name):
        '''
        Function that generates the XML file for the optimized slot behavior, overwriting previous files
        '''

        file = os.path.join( path, XML_name )

        # create the file structure
        auto_head = '1' if auto_head else '0'
        EL_behavior = ET.Element('behavior',{'description':'Add description to XML file', "auto_head":auto_head})

        for i,s in enumerate(slots):
            EL_slot = ET.SubElement(EL_behavior, 'slot', {'delta':str(s[0]/1000)})
            for j in s[1]: # go through all joint indices
                ET.SubElement(EL_slot, 'move', {'id':str(j), 'angle':str(s[2][j])})

        # create XML file
        xml_rough = ET.tostring( EL_behavior, 'utf-8' )
        xml_pretty = minidom.parseString(xml_rough).toprettyxml(indent="    ")
        with open(file, "w") as x:
            x.write(xml_pretty)
        
        print(file, "was created!")

    @staticmethod
    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        '''
        Linear learning rate schedule

        Parameters
        ----------
        initial_value : float
            Initial learning rate
        
        Returns
        -------
        schedule : Callable[[float], float]
            schedule that computes current learning rate depending on remaining progress
        '''
        def func(progress_remaining: float) -> float:
            '''
            Compute learning rate according to current progress

            Parameters
            ----------
            progress_remaining : float
                Progress will decrease from 1 (beginning) to 0
            
            Returns
            -------
            learning_rate : float
                Learning rate according to current progress
            '''
            return progress_remaining * initial_value

        return func

    @staticmethod
    def export_model(input_file, output_file, add_sufix=True):
        '''
        Export model weights to binary file

        Parameters
        ----------
        input_file : str
            Input file, compatible with algorithm
        output_file : str
            Output file, including directory
        add_sufix : bool
            If true, a suffix is appended to the file name: output_file + "_{index}.pkl"
        '''

        # If file already exists, don't overwrite
        if add_sufix:
            for i in count():
                f = f"{output_file}_{i:03}.pkl"
                if not os.path.isfile(f):
                    output_file = f
                    break
        
        model = PPO.load(input_file)
        weights = model.policy.state_dict() # dictionary containing network layers

        w = lambda name : weights[name].detach().cpu().numpy() # extract weights from policy

        var_list = []
        for i in count(0,2): # add hidden layers (step=2 because that's how SB3 works)
            if f"mlp_extractor.policy_net.{i}.bias" not in weights:
                break
            var_list.append([w(f"mlp_extractor.policy_net.{i}.bias"), w(f"mlp_extractor.policy_net.{i}.weight"), "tanh"])

        var_list.append( [w("action_net.bias"), w("action_net.weight"), "none"] ) # add final layer
        
        with open(output_file,"wb") as f:
            pickle.dump(var_list, f, protocol=4) # protocol 4 is backward compatible with Python 3.4



class Cyclic_Callback(BaseCallback):
    ''' Stable baselines custom callback '''
    def __init__(self, freq, function):
        super(Cyclic_Callback, self).__init__(1)
        self.freq = freq
        self.function = function

    def _on_step(self) -> bool:
        if self.n_calls % self.freq == 0:
            self.function()
        return True # If the callback returns False, training is aborted early

class Export_Callback(BaseCallback):
    ''' Stable baselines custom callback '''
    def __init__(self, freq, load_path, export_name):
        super(Export_Callback, self).__init__(1)
        self.freq = freq
        self.load_path = load_path
        self.export_name = export_name

    def _on_step(self) -> bool:
        if self.n_calls % self.freq == 0:
            path = os.path.join(self.load_path, f"model_{self.num_timesteps}_steps.zip")
            Train_Base.export_model(path, f"./scripts/gyms/export/{self.export_name}")
        return True # If the callback returns False, training is aborted early