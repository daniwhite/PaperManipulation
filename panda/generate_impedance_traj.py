import sys
sys.path.append("../")

from ctrl.impedance_generators.setpoint_generators.circle import CircularSetpointGenerator
import constants

import numpy as np

from pydrake.all import RollPitchYaw
import quaternion

def get_traj(end_time, desired_radius):
    setpoint_gen = CircularSetpointGenerator(
        desired_radius=desired_radius, 
        end_time=end_time,
        sys_consts=constants.nominal_sys_consts
    )

    times = np.arange(0, end_time, 0.1)
    x0s = np.zeros((len(times), 6))

    for i, t in enumerate(times):
        x0s[i,:] = setpoint_gen._calc_x0(t)
    
    positions = x0s[:,3:]
    rpys = x0s[:,:3]

    quaternions = np.zeros((len(times), 4))
    for i, t in enumerate(times):
        quat = RollPitchYaw(rpys[0,:]).ToQuaternion()
        quaternions[i,0] = quat.w()
        quaternions[i,1] = quat.x()
        quaternions[i,2] = quat.y()
        quaternions[i,3] = quat.z()

    poses = []
    for pos, quat in zip(positions, quaternions):
        pos_dict = {}
        pos_dict["position"]  = pos
        pos_dict["orientation"] = quaternion.quaternion(*quat)
        poses.append(pos_dict)
    
    return poses

poses=get_traj(
    desired_radius=constants.nominal_sys_consts.w_L/2,
    end_time=30
)
np.savez("poses.npz", poses=poses)

