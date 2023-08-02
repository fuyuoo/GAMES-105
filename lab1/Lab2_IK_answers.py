import numbers

import numpy as np
from scipy.spatial.transform import Rotation as R

def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入:
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """

    # CCD IK根节点到末端节点，返回Path_ori,path_pos
    # FK 剩余节点，返回所有的ori,pos
    joint_parents = meta_data.joint_parent
    joint_names = meta_data.joint_name
    joint_initial_position = meta_data.joint_initial_position
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()

    joint_offsets = []
    for i in range(len(joint_parents)):
        if i in path:
            parent_index = path[path.index(i) - 1]
        else:
            parent_index = joint_parents[i]
        cur_index = i
        if parent_index == -1:
            offset = np.array([0, 0, 0])
        else:
            position_A = joint_initial_position[parent_index]
            position_B = joint_initial_position[cur_index]
            offset = position_A - position_B
        joint_offsets.append(-offset)

    root_joint_index = joint_names.index(meta_data.root_joint)
    # 初始rotation
    joint_rotation = {0: R.from_quat(joint_orientations[0])}
    for i in range(len(joint_parents)):
        parent_index = joint_parents[i]
        if parent_index == -1:
            continue
        joint_rotation[i] = R.inv(R.from_quat(joint_orientations[parent_index])) * R.from_quat(joint_orientations[i])

    # begin ccd
    cnt = 0
    end_joint_index = joint_names.index(meta_data.end_joint)
    min_distance = np.linalg.norm(target_pose - joint_positions[end_joint_index])
    while min_distance > 0.01 and cnt <= 10:
        # 逆序CCD
        for i in range(len(path) - 2, -1, -1):
            cur_joint_index = i
            index = path[cur_joint_index]
            cur_position = joint_positions[index]
            cur2target_vector = (target_pose - cur_position) / np.linalg.norm(target_pose - cur_position)
            cur2end_vector = (joint_positions[end_joint_index] - cur_position) / np.linalg.norm(joint_positions[end_joint_index] - cur_position)

            # 计算轴角
            rotation_radius = np.arccos(np.clip(np.dot(cur2end_vector, cur2target_vector), -1, 1))
            rotation_axis = np.cross(cur2end_vector, cur2target_vector)
            rotation_axis_noml = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_vector = R.from_rotvec(rotation_radius * rotation_axis_noml)

            # CCD 单次结果
            joint_orientations[index] = (rotation_vector * R.from_quat(joint_orientations[index])).as_quat()

            # 计算local rotation
            joint_rotation_temp = {path[0]: joint_orientations[path[0]]}
            for j in range(i + 1, len(path)):
                joint_index = path[j]
                parent_index = path[j - 1]
                joint_rotation_temp[joint_index] = R.inv(R.from_quat(joint_orientations[parent_index])) * R.from_quat(joint_orientations[joint_index])
            # 局部FK
            for j in range(i + 1, len(path)):
                j_parent_index = path[j - 1]
                j_index = path[j]
                joint_positions[j_index] = joint_positions[j_parent_index] + R.from_quat(joint_orientations[j_parent_index]).apply(joint_offsets[j_index])

                if j < len(path) - 1:
                    joint_orientations[j_index] = (R.from_quat(joint_orientations[j_parent_index]) * joint_rotation_temp[j_index]).as_quat()
                else:
                    joint_orientations[j_index] = joint_orientations[j_parent_index]

            min_distance = min(min_distance, np.linalg.norm(target_pose - joint_positions[end_joint_index]))
            if min_distance <= 0.01:
                break
        cnt += 1

    for i in range(len(joint_parents)):
        if i in path:
            continue
        parent_index = joint_parents[i]
        if parent_index == -1:  # 根节点
            Q1 = joint_rotation[i]# 全局朝向
            P1 = joint_positions[i]
        else:
            Q0 = R.from_quat(joint_orientations[parent_index])
            R1 = joint_rotation[i]
            P0 = joint_positions[parent_index]
            L0 = joint_offsets[i]

            # 计算当前关节的全局位置和朝向
            Q1 = Q0 * R1
            P1 = P0 + Q0.apply(L0)

        joint_positions[i] = P1
        joint_orientations[i] = Q1.as_quat()

    # 保持根节点旋转不变，其他节点从根节点从脚开始映射回从原本的root开始的朝向
    # 因为旋转本身是子节点在父节点的朝向方向上，对调了之后，需要把旋转也对调，所有把节点ori往后挪一位，让现在的父节点得到子节点的朝向，并不需要逆旋转
    temp_joint_ori = joint_orientations.copy()
    for i in range(len(path2)-1):
        joint_orientations[path2[i+1]] = temp_joint_ori[path2[i]]

    return joint_positions, joint_orientations


def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """

    return joint_positions, joint_orientations


def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """

    return joint_positions, joint_orientations