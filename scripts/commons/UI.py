from itertools import zip_longest
from math import inf
import math
import numpy as np
import shutil

class UI():
    console_width = 80
    console_height = 24
    
    @staticmethod
    def read_particle(prompt, str_options, dtype=str, interval=[-inf,inf]):
        ''' 
        从用户处读取一个值，该值可以是给定的 dtype 或 str_options 列表中的一个选项

        参数
        ----------
        prompt : `str`
            在读取输入之前显示给用户的提示
        str_options : `list`
            字符串选项列表（除了 dtype 之外）
        dtype : `class`
            如果 dtype 是 str，则用户必须从 str_options 中选择一个值，否则也可以发送一个 dtype 值
        interval : `list`
            [>=min,<max] 区间，用于数值类型的 dtype
        
        返回值
        -------
        choice : `int` 或 dtype
            str_options 的索引（int）或值（dtype）
        is_str_option : `bool`
            如果 `choice` 是 str_options 的索引，则为 True
        '''
        # 检查用户是否有选择
        if dtype is str and len(str_options) == 1:
            print(prompt, str_options[0], sep="")
            return 0, True
        elif dtype is int and interval[0] == interval[1]-1:
            print(prompt, interval[0], sep="")
            return interval[0], False

        while True:
            inp = input(prompt)
            if inp in str_options: 
                return str_options.index(inp), True

            if dtype is not str:
                try:
                    inp = dtype(inp)
                    if inp >= interval[0] and inp < interval[1]:
                        return inp, False
                except:
                    pass
            
            print("Error: illegal input! Options:", str_options, f" or  {dtype}" if dtype != str else "")

    @staticmethod
    def read_int(prompt, min, max):
        ''' 
        从用户处读取一个整数，该整数在给定区间内
        :param prompt: 在读取输入之前显示给用户的提示
        :param min: 最小输入值（包含）
        :param max: 最大输入值（不包含）
        :return: 用户选择的整数
        '''
        while True:
            inp = input(prompt)
            try:
                inp = int(inp)
                assert inp >= min and inp < max
                return inp
            except:
                print(f"Error: illegal input! Choose number between {min} and {max-1}")

    @staticmethod
    def print_table(data, titles=None, alignment=None, cols_width=None, cols_per_title=None, margins=None, numbering=None, prompt=None):
        '''
        打印表格
        
        参数
        ----------
        data : `list`
            列表的列表，每一列是一个项目列表
        titles : `list`
            每列的标题列表，默认为 `None`（不显示标题）
        alignment : `list`
            每列的对齐方式（不包括标题），默认为 `None`（所有列左对齐）
        cols_width : `list`
            每列的宽度列表，默认为 `None`（根据内容调整宽度）
            正值表示固定的列宽
            零表示该列将根据其内容调整宽度
        cols_per_title : `list`
            每个标题的最大子列数，默认为 `None`（每个标题一个子列）
        margins : `list`
            每列的前后空白字符数，默认为 `None`（每列的空白字符数为2）
        numbering : `list`
            每列的布尔值列表，指示是否为每个选项分配编号
        prompt : `str`
            如果提供，将在表格打印后提示用户输入

        返回值
        -------
        index : `int`
            返回所选项目的全局索引（相对于表格）
        col_index : `int`
            返回所选项目的局部索引（相对于列）
        column : `int`
            返回所选项目的列号（从0开始）
        * 如果 `numbering` 或 `prompt` 为 `None`，则返回 `None`
        
        示例
        -------
        titles = ["Name","Age"]
        data = [["John","Graciete"], [30,50]]
        alignment = ["<","^"]               # 第一列左对齐，第二列居中对齐
        cols_width = [10,5]                # 第一列宽度为10，第二列宽度为5
        margins = [3,3]                    
        numbering = [True,False]           # 打印：[0-John,1-Graciete][30,50]
        prompt = "Choose a person:"
        '''
        
        #--------------------------------------------- 参数
        cols_no = len(data)

        if alignment is None:
            alignment = ["<"]*cols_no

        if cols_width is None:
            cols_width = [0]*cols_no

        if numbering is None:
            numbering = [False]*cols_no
            any_numbering = False
        else:
            any_numbering = True

        if margins is None:
            margins = [2]*cols_no

        # 根据内容调整列宽（如果需要）
        subcol = [] # 子列长度和宽度
        for i in range(cols_no):
            subcol.append([[],[]])
            if cols_width[i] == 0:
                numbering_width = 4 if numbering[i] else 0
                if cols_per_title is None or cols_per_title[i] < 2:
                    cols_width[i] = max([len(str(item))+numbering_width for item in data[i]]) + margins[i]*2
                else:
                    subcol[i][0] = math.ceil(len(data[i])/cols_per_title[i]) # 子列最大长度
                    cols_per_title[i] = math.ceil(len(data[i])/subcol[i][0]) # 根据需要减少列数
                    cols_width[i] = margins[i]*(1+cols_per_title[i]) - (1 if numbering[i] else 0) # 如果有编号，则移除一个，与打印时相同
                    for j in range(cols_per_title[i]):
                        subcol_data_width = max([len(str(item))+numbering_width for item in data[i][j*subcol[i][0]:j*subcol[i][0]+subcol[i][0]]])
                        cols_width[i] += subcol_data_width     # 将子列数据宽度添加到列宽
                        subcol[i][1].append(subcol_data_width) # 保存子列数据宽度
                        
                if titles is not None: # 如果需要，扩展以容纳标题
                    cols_width[i] = max(cols_width[i], len(titles[i]) + margins[i]*2  )

        if any_numbering:
            no_of_items=0
            cumulative_item_per_col=[0] # 用于获取局部索引
            for i in range(cols_no):
                assert type(data[i]) == list, "In function 'print_table', 'data' must be a list of lists!"

                if numbering[i]:
                    data[i] = [f"{n+no_of_items:3}-{d}" for n,d in enumerate(data[i])]
                    no_of_items+=len(data[i])
                cumulative_item_per_col.append(no_of_items)

        table_width = sum(cols_width)+cols_no-1

        #--------------------------------------------- 列标题
        print(f'{"="*table_width}')
        if titles is not None:
            for i in range(cols_no):
                print(f'{titles[i]:^{cols_width[i]}}', end='|' if i < cols_no - 1 else '')
            print()
            for i in range(cols_no):
                print(f'{"-"*cols_width[i]}', end='+' if i < cols_no - 1 else '')
            print()

        #--------------------------------------------- 合并子列
        if cols_per_title is not None:
            for i,col in enumerate(data):
                if cols_per_title[i] < 2:
                    continue
                for k in range(subcol[i][0]): # 创建合并后的项目
                    col[k] = (" "*margins[i]).join( f'{col[item]:{alignment[i]}{subcol[i][1][subcol_idx]}}' 
                                                    for subcol_idx, item in enumerate(range(k,len(col),subcol[i][0])) )
                del col[subcol[i][0]:] # 删除重复的项目
        
        #--------------------------------------------- 列项目
        for line in zip_longest(*data):       
            for i,item in enumerate(line):
                l_margin = margins[i]-1 if numbering[i] else margins[i] # 如果有编号，则调整边距
                item = "" if item is None else f'{" "*l_margin}{item}{" "*margins[i]}' # 添加边距
                print(f'{item:{alignment[i]}{cols_width[i]}}', end='')
                if i < cols_no - 1:
                    print(end='|')
            print(end="\n")
        print(f'{"="*table_width}')

        #--------------------------------------------- 提示
        if prompt is None:
            return None

        if not any_numbering:
            print(prompt)
            return None

        index = UI.read_int(prompt, 0, no_of_items)

        for i,n in enumerate(cumulative_item_per_col):
            if index < n:
                return index, index-cumulative_item_per_col[i-1], i-1

        raise ValueError('Failed to catch illegal input')


    @staticmethod
    def print_list(data, numbering=True, prompt=None, divider=" | ", alignment="<", min_per_col=6):
        '''
        打印列表 - 使用尽可能多的列打印列表
        
        参数
        ----------
        data : `list`
            项目列表
        numbering : `bool`
            为每个选项分配编号
        prompt : `str`
            如果提供，将在表格打印后提示用户输入
        divider : `str`
            分隔列的字符串
        alignment : `str`
            f-string 风格的对齐方式 ( '<', '>', '^' )
        min_per_col : int
            避免拆分项目数较少的列
        
        返回值
        -------
        item : `int`, item
            返回所选项目的全局索引和项目对象，
            或 `None`（如果 `numbering` 或 `prompt` 为 `None`）

        '''
        
        WIDTH = shutil.get_terminal_size()[0]

        data_size = len(data)   
        items = []
        items_len = []

        #--------------------------------------------- 添加编号、边距和分隔符
        for i in range(data_size):
            number = f"{i}-" if numbering else ""
            items.append( f"{divider}{number}{data[i]}" )
            items_len.append( len(items[-1]) )

        max_cols = np.clip((WIDTH+len(divider)) // min(items_len),1,math.ceil(data_size/max(min_per_col,1))) # 宽度 + len(divider)，因为最后一列不需要分隔符

        #--------------------------------------------- 检查最大列数，考虑内容宽度（最小值为1）
        for i in range(max_cols,0,-1):
            cols_width = []
            cols_items = []
            table_width = 0
            a,b = divmod(data_size,i)
            for col in range(i):
                start = a*col + min(b,col)
                end = start+a+(1 if col<b else 0)
                cols_items.append( items[start:end] )
                col_width = max(items_len[start:end])
                cols_width.append( col_width )
                table_width += col_width
            if table_width <= WIDTH+len(divider):
                break
        table_width -= len(divider)
        
        #--------------------------------------------- 打印列
        print("="*table_width)
        for row in range(math.ceil(data_size / i)):
            for col in range(i):
                content = cols_items[col][row] if len(cols_items[col]) > row else divider # 如果没有项目，则打印分隔符
                if col == 0:
                    l = len(divider)
                    print(end=f"{content[l:]:{alignment}{cols_width[col]-l}}")  # 从第一列中移除分隔符
                else:
                    print(end=f"{content    :{alignment}{cols_width[col]  }}") 
            print()  
        print("="*table_width)

        #--------------------------------------------- 提示
        if prompt is None:
            return None

        if numbering is None:
            return None
        else:
            idx = UI.read_int( prompt, 0, data_size )
            return idx, data[idx]
