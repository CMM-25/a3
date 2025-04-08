import numpy as np
from scipy.spatial.transform import Rotation as R

def joint_transform(joint, angle):
    transform = np.eye(4)
    transform[:3, :3] = R.from_rotvec(joint.axis * angle).as_matrix()
    transform[:3, 3] = joint.origin[:3, 3]
    return transform

def forward_kinematics(robot, base_pos, base_quat, joint_angles):
    base_tf = np.eye(4)
    base_tf[:3, 3] = base_pos
    base_tf[:3, :3] = R.from_quat(base_quat).as_matrix()
    frames = {robot.base_link.name: base_tf}

    for _ in range(3):  # do a few times to make sure all joints are covered
        for joint in robot.joints:
            if joint.parent in frames and joint.child not in frames:
                transform = frames[joint.parent] @ joint.origin
                if joint.name in joint_angles['names']:
                    id = joint_angles['names'].index(joint.name)
                    transform = transform @ joint_transform(joint, joint_angles['values'][id])
                frames[joint.child] = transform
    return frames

def set_axes_equal(ax):
    """Set 3D plot axes to equal scale."""
    limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    mean_limits = limits.mean(axis=1)
    max_range = (limits[:, 1] - limits[:, 0]).max() / 2

    # ax.set_xlim(mean_limits[0] - max_range, mean_limits[0] + max_range)
    # ax.set_ylim(mean_limits[1] - max_range, mean_limits[1] + max_range)
    # ax.set_zlim(mean_limits[2] - max_range, mean_limits[2] + max_range)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 2)

def plot_robot(ax, robot, frames, color='bo-', label='policy'):
    ax.clear()

    policy_plotted = False
    for joint in robot.joints:
        if joint.parent in frames and joint.child in frames:
            p1 = frames[joint.parent][:3, 3]
            p2 = frames[joint.child][:3, 3]
            if not policy_plotted:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color, label=label)
                policy_plotted = True
            else:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    set_axes_equal(ax)