# Lab1 小结

## FK 

* FK公式
  * $Q_1 $ = $Q_0$ * $R_1$
    * `*` 代表旋转叠加，而不是点乘，点乘的结果是标量
    * 也不是叉乘，叉乘求的是两个向量之间的垂直向量
  * $P_1$ = $P_0$+ $Q_0$$L_0$
    * 这里就是把旋转应用到$L_0$上
* 叶子节点
  * 在**Bvh**文件中的叶子节点是没有旋转的，只有**Offset**
  * **Offset**代表的是**自己和父节点**之间的向量差
* 局部和全局坐标系
  * Q，世界坐标系，朝向
  * R，局部坐标系，旋转 



## Retarget

* 没有实现任意姿势，而且更多的就不是同一个骨骼名字，或者更多或更少骨骼的问题
  * 实现任意姿势Retarget就是找到相同的骨骼，把差异化旋转叠加到需要的旋转上
  * todo 公式
* 因为T-Pose和A-Pose只在`lShoulder` 和`rShoulder`上旋转了45°，所以只是简单的把A-Pose的动画资源在响应位置的局部rotation进行旋转就可以了
  * 因为骨骼的父子节点关系不一致，需要重新映射
    * 关系不一致会导致在MotionData的index不一致

## IK

* 使用方式：指定关节接近某个点，然后逆向求的其他关节的变化

* CCD方法

  * 从节点开始遍历，让`cur2end_vector`旋转到 `cur2target_vector`上去
    * 因为两点之间线段最短，所以一直遍历就能得到想要得结果，但是迭代速度较慢
  * 方案
    1. 确定joint_end和joint_root
    2. 开始CDD
       1. 一般是从末端节点逆序遍历，因为最开始迭代得关节旋转变化是最大的，更自然，当然也看需求
    3. 最后把其余不受IK路径的关节也进行一遍FK
       1. 因为IK有可能会改变其余节点的父关节，所以需要从新FK
       2. 再IK的迭代过程中会一直FK，所以不需要重新FK

* 细节点
  * 局部旋转可以由父节点朝向的逆叠加上当前的朝向求得

    * ```
      joint_rotation[i] = R.inv(R.from_quat(joint_orientations[parent_index])) * R.from_quat(joint_orientations[i])
      ```

    * 且需要区分

  * 计算旋转向量，通过叉乘得到两个向量得垂直向量的归一向量作为旋转轴，再通过两个向量点乘的反三角函数（arccos）求得角度，再把旋转轴乘以旋转角度得到旋转向量

    * ```
      rotation_radius = np.arccos(np.clip(np.dot(cur2end_vector, cur2target_vector), -1, 1))
      rotation_axis = np.cross(cur2end_vector, cur2target_vector)
      rotation_axis_noml = rotation_axis / np.linalg.norm(rotation_axis)
      rotation_vector = R.from_rotvec(rotation_radius * rotation_axis_noml)
      ```

  * 如果路径经过了root，offset和rotation需要注意，因为父子关系变化了

    * 因为ori作用是当前节点的下一段的方向，且是全局朝向，但是父子对调了，所以ori需要重新映射一下

    * ```
      temp_joint_ori = joint_orientations.copy()
      for i in range(len(path2)-1):
      	joint_orientations[path2[i+1]] = temp_joint_ori[path2[i]]
      ```

    * 

    

