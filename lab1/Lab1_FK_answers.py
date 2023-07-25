import numpy
import numpy as np
from scipy.spatial.transform import Rotation as R


def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i + 1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1, -1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def part1_calculate_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = []
    joint_stack = []

    file = open(bvh_file_path)
    datas = file.readlines()
    # 通过\n 分割 datas

    datas = [data.strip() for data in datas]

    def build(bvh_data):
        if len(bvh_data) == 0:
            return

        # pop
        data = bvh_data[0]

        # 三个空格合成一个
        data = data.replace('   ', ' ')
        data = data.replace('  ', ' ')
        split_data = data.split(' ')
        if data == "MOTION":
            return

        if data == 'HIERARCHY':
            pass

        elif data.startswith("ROOT"):
            joint_name.append(split_data[1])
            joint_stack.append(split_data[1])
            joint_parent.append(-1)

        elif data.startswith("OFFSET"):
            joint_offset.append(np.array([split_data[1], split_data[2], split_data[3]], float))

        elif data.startswith("JOINT"):
            joint_parent.append(joint_name.index(joint_stack[-1]))
            joint_name.append(split_data[1])
            joint_stack.append(split_data[1])

        elif data.startswith("{"):
            return build(bvh_data[1:], )

        elif data.startswith("}"):
            joint_stack.pop()
            return build(bvh_data[1:])

        elif data.startswith("End"):
            joint_parent.append(joint_name.index(joint_stack[-1]))
            parent_name = joint_name[-1]
            joint_name.append(parent_name + "_end")
            joint_stack.append(split_data[1])

        elif data.startswith("CHANNELS"):
            pass

        return build(bvh_data[1:])

    build(datas)

    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_names, joint_parents, joint_offsets, motion_datas, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    motion_data = motion_datas[frame_id]
    joint_positions = []  # 全局位置
    joint_orientations = []  # 全局朝向
    joint_rotation = []  # 局部旋转 motion_data信息

    joint_position_temp = [motion_data[0], motion_data[1], motion_data[2]]
    motion_data = motion_data[3:]

    index_rotate = 0
    for i in range(len(joint_names)):
        if joint_names[i].endswith("_end"):
            joint_rotation.append([0, 0, 0])
        else:
            num = index_rotate * 3
            joint_rotation.append([motion_data[num], motion_data[num + 1], motion_data[num + 2]])
            index_rotate += 1

    for i in range(len(joint_names)):
        parent_index = joint_parents[i]
        if parent_index == -1:
            Q1 = R.from_euler('XYZ', joint_rotation[0], degrees=True)  # 全局朝向
            P1 = joint_position_temp
        else:
            Q0 = R.from_quat(joint_orientations[parent_index])
            R1 = joint_rotation[i]
            P0 = joint_positions[parent_index]
            L0 = joint_offsets[i]

            # 计算当前关节的全局位置和朝向
            Q1 = Q0 * R.from_euler('XYZ', R1, degrees=True)
            P1 = P0 + Q0.apply(L0)

        joint_positions.append(P1)
        joint_orientations.append(Q1.as_quat())

    return np.array(joint_positions), np.array(joint_orientations)


def get_rotation_matrix(z_angle_degrees):
    # 将角度转换为弧度
    angle_radians = np.radians(z_angle_degrees)
    # 旋转轴（这里选择绕Z轴旋转）
    rotation_axis = [0, 0, 1]
    # 使用Rotation类构建旋转对象
    rotation = R.from_rotvec(angle_radians * rotation_axis)
    # 得到旋转矩阵
    rotation_matrix = rotation.as_matrix()
    return rotation_matrix

def get_rotation_matrix(vector_A, vector_B):
    # Normalize the input vectors to ensure they have unit length
    vector_A = vector_A / np.linalg.norm(vector_A)
    vector_B = vector_B / np.linalg.norm(vector_B)

    # Compute the rotation quaternion between the two unit vectors
    rotation_quaternion = R.from_quat(np.cross(vector_A, vector_B))

    # Convert the rotation quaternion to a rotation matrix
    rotation_matrix = rotation_quaternion.as_matrix()

    return rotation_matrix

# A的旋转 T的骨骼，但是因为基础姿势不一样，所以不能直接使用，需要把基础姿势的偏移扭回来
def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """

    # 关节index
    A_joint_name, A_joint_parent,A_offset = part1_calculate_pose(A_pose_bvh_path)
    T_joint_name, T_joint_parent,T_offset = part1_calculate_pose(T_pose_bvh_path)

    # 建立骨骼映射
    A_bone_index = {}
    for i in range(len(A_joint_name)):
        A_bone_index[A_joint_name[i]] = i

    A_motion_data = load_motion_data(A_pose_bvh_path)

    res = []
    for frame in range(len(A_motion_data)):
        data = []
        A_motion_data_frame = A_motion_data[frame]
        # init 硬去掉position参数
        motion_data_node = A_motion_data_frame[0:3]
        A_motion_data_frame = A_motion_data_frame[3:]

        # 骨骼motion旋转
        index_rotate = 0
        A_joint_rotation = []
        for i in range(len(A_joint_name)):
            if A_joint_name[i].endswith("_end"):
                A_joint_rotation.append([0, 0, 0])
            else:
                num = index_rotate * 3
                A_joint_rotation.append([A_motion_data_frame[num], A_motion_data_frame[num + 1], A_motion_data_frame[num + 2]])
                index_rotate += 1

        for i in range(len(T_joint_name)):


            join_name = T_joint_name[i]
            A_index = A_bone_index[join_name]

            if join_name.endswith("_end"):
                continue

            if join_name == "RootJoint":
                data.append(motion_data_node)

            elif join_name == "lShoulder":
                A_joint_rotation[A_index][2] -= 45
            elif join_name == "rShoulder":
                A_joint_rotation[A_index][2] += 45
            data.append(A_joint_rotation[A_index])

        temp = []
        for u in data:
            temp.append(u[0])
            temp.append(u[1])
            temp.append(u[2])
        res.append(temp)

    return np.array(res)
