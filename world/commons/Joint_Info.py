import numpy as np

class Joint_Info():
    """
    从 XML 元素中提取关节信息的类。
    """
    def __init__(self, xml_element) -> None:
        """
        初始化关节信息。
        :param xml_element: 包含关节信息的 XML 元素。
        """
        # 提取感知器名称
        self.perceptor = xml_element.attrib['perceptor']
        # 提取执行器名称
        self.effector = xml_element.attrib['effector']
        # 提取旋转轴向量
        self.axes = np.array([
            float(xml_element.attrib['xaxis']),  # X 轴分量
            float(xml_element.attrib['yaxis']),  # Y 轴分量
            float(xml_element.attrib['zaxis'])   # Z 轴分量
        ])
        # 提取关节的最小角度限制
        self.min = int(xml_element.attrib['min'])
        # 提取关节的最大角度限制
        self.max = int(xml_element.attrib['max'])

        # 提取第一个锚点信息
        self.anchor0_part = xml_element[0].attrib['part']  # 第一个锚点连接的身体部分名称
        self.anchor0_axes = np.array([
            float(xml_element[0].attrib['y']),  # Y 轴分量（注意：这里与 XML 中的顺序相反）
            float(xml_element[0].attrib['x']),  # X 轴分量
            float(xml_element[0].attrib['z'])   # Z 轴分量
        ])  # 第一个锚点的旋转轴向量（X 和 Y 轴被交换）

        # 提取第二个锚点信息
        self.anchor1_part = xml_element[1].attrib['part']  # 第二个锚点连接的身体部分名称
        self.anchor1_axes_neg = np.array([
            -float(xml_element[1].attrib['y']),  # Y 轴分量（取负值）
            -float(xml_element[1].attrib['x']),  # X 轴分量（取负值）
            -float(xml_element[1].attrib['z'])   # Z 轴分量（取负值）
        ])  # 第二个锚点的旋转轴向量（X 和 Y 轴被交换，并取负值）
