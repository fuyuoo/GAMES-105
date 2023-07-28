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

    joint_offsets = []
    joint_parents = meta_data.joint_parent
    joint_names = meta_data.joint_name
    joint_initial_position = meta_data.joint_initial_position
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()

    for i in range(len(joint_names)):
        if i == 0:
            offset = np.array([0, 0, 0])
        else:
            parent_index = joint_parents[i]
            position_A = joint_initial_position[parent_index]
            position_B = joint_initial_position[i]
            offset = position_A - position_B
        joint_offsets.append(-offset)

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
            cur2end_vector = (joint_positions[end_joint_index] - cur_position) / np.linalg.norm(
                joint_positions[end_joint_index] - cur_position)

            # 计算轴角
            rotation_radius = np.arccos(np.clip(np.dot(cur2end_vector, cur2target_vector), -1, 1))
            rotation_axis = np.cross(cur2end_vector, cur2target_vector)
            rotation_axis_noml = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_vector = R.from_rotvec(rotation_radius * rotation_axis_noml)

            # CCD 单次结果
            joint_orientations[index] = (rotation_vector * R.from_quat(joint_orientations[index])).as_quat()

            # 计算local rotation
            joint_rotation = {path[0]: joint_orientations[path[0]]}
            for j in range(1, len(path)):
                joint_index = path[j]
                parent_index = path[j - 1]
                rotation_save = R.inv(R.from_quat(joint_orientations[parent_index])) * R.from_quat(
                    joint_orientations[joint_index])
                joint_rotation[joint_index] = rotation_save
            # 局部FK
            for j in range(i + 1, len(path)):
                j_parent_index = path[j - 1]
                j_index = path[j]
                joint_positions[j_index] = joint_positions[j_parent_index] + R.from_quat(
                    joint_orientations[j_parent_index]).apply(joint_offsets[j_index])

                if j < end_joint_index - 1:
                    joint_orientations[j_index] = (
                                R.from_quat(joint_orientations[j_parent_index]) * joint_rotation[j_index]).as_quat()
                else:
                    joint_orientations[j_index] = joint_orientations[j_parent_index]

            min_distance = min(min_distance, np.linalg.norm(target_pose - joint_positions[end_joint_index]))
            if min_distance <= 0.01:
                break
        cnt += 1

    root_joint_index = joint_names.index(meta_data.root_joint)
    joint_rotation = {root_joint_index: R.from_quat(joint_orientations[root_joint_index])}
    for i in range(len(joint_parents)):
        parent_index = joint_parents[i]
        if parent_index == -1:
            continue
        rotation_save = R.inv(R.from_quat(joint_orientations[parent_index])) * R.from_quat(joint_orientations[i])
        joint_rotation[i] = rotation_save

    # fk but not in path
    for i in range(len(joint_parents)):
        if i in path:
            continue
        parent_index = joint_parents[i]
        if parent_index == -1:  # 根节点
            Q1 = R.from_euler('XYZ', joint_rotation[i], degrees=True)  # 全局朝向
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
