import sys
sys.path.append("../")

from ctrl.impedance_generators.setpoint_generators.circle import CircularSetpointGenerator
import constants

import numpy as np
import config
import panda.panda_config as panda_config

from pydrake.all import RollPitchYaw, RigidTransform
import quaternion

def get_traj(x0s):
    poses = []
    positions = x0s[:,3:]
    rpys = x0s[:,:3]
    for i, x0 in enumerate(x0s):
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


x0s = np.load("x0s_sim.npz")["x0s"]
poses=get_traj(x0s)
np.savez("x0s_hw.npz", poses=poses)

