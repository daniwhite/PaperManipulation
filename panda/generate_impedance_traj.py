import sys
sys.path.append("../")

from ctrl.impedance_generators.setpoint_generators.circle import CircularSetpointGenerator
import constants

import numpy as np
import config
import panda_config

from pydrake.all import RollPitchYaw, RigidTransform
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

    poses = []
    for i in range(len(times)):
        rpy = RollPitchYaw(rpys[i,:])
        X_SW = RigidTransform(
            rpy=rpy, p=positions[i,:]
        )
        X_HW = panda_config.X_panda_SP_hw(X_SW)

        pos_dict = {}
        pos_dict["position"]  = X_HW.translation()
        pos_dict["orientation"] = quaternion.quaternion(
            X_HW.rotation().ToQuaternion().w(),
            X_HW.rotation().ToQuaternion().x(),
            X_HW.rotation().ToQuaternion().y(),
            X_HW.rotation().ToQuaternion().z()
        )
        poses.append(pos_dict)
    return poses

poses=get_traj(
    desired_radius=constants.nominal_sys_consts.w_L/2,
    end_time=30
)
np.savez("poses.npz", poses=poses)

