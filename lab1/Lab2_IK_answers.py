import numbers

import numpy as np
from scipy.spatial.transform import Rotation as R


def FK(path, joint_names,joint_parents, joint_offsets, joint_rotation):

    joint_positions = []
    joint_orientations = []
    for i in range(len(joint_names)):
        if i in path:
            continue
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

    # CCD 从叶子节点反向迭代
    path, _, _, _ = meta_data.get_path_from_root_to_end()

    # init rotation
    joint_names = meta_data.joint_name
    joint_parents = meta_data.joint_parent

    joint_rotations = [R.from_quat(joint_orientations[0])]
    for i in range(1, len(path) - 1):
        orientation_A = joint_orientations[path[i]]
        orientation_B = joint_orientations[path[i + 1]]
        rotation_offset = np.multiply(orientation_A, np.conj(orientation_B))
        joint_rotations.append(R.from_quat(rotation_offset))
    joint_rotations.append(R.from_quat([0, 0, 0, 1]))

    for i in range(len(joint_names)):
        if joint_names[i].endswith("_end"):
            joint_rotations.append([0, 0, 0])
        else:
            joint_rotation.append([motion_data[num], motion_data[num + 1], motion_data[num + 2]])

    # init offset
    joint_initial_position = meta_data.joint_initial_position
    joint_offsets = [np.array([0, 0, 0])]
    for i in range(1, len(joint_names)):
        parent_index = joint_parents[i]
        position_A = joint_initial_position[parent_index]
        position_B = joint_initial_position[i]
        offset = position_A - position_B
        joint_offsets.append(offset)

    min_distance = np.linalg.norm(target_pose - joint_positions[path[-1]])
    cnt = 0
    while min_distance > 0.01 and cnt <= 10:
        for i in range(len(path) - 2, -1, -1):
            cur_position = joint_positions[path[i]]
            target_vector = target_pose - cur_position
            end_vector = joint_positions[path[-1]] - cur_position
            rotate_quat = R.align_vectors([target_vector],[end_vector])[0]
            joint_rotations[path[i]] = rotate_quat * joint_rotations[path[i]]
            joint_orientations[path[i]] = (rotate_quat * R.from_quat(joint_orientations[path[i]])).as_quat()
            for j in range(i,len(path) - 1):
                joint_positions[path[j+1]] = joint_positions[path[j]] + R.from_quat(joint_orientations[path[j]]).apply(joint_offsets[path[j + 1]])
            min_distance = min(min_distance, np.linalg.norm(target_pose - joint_positions[path[-1]]))
            if min_distance <= 0.01:
                break
        cnt += 1
    FK(path, joint_names,joint_parents, joint_positions, joint_orientations)

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
