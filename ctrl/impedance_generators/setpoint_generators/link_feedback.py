import numpy as np

# Drake imports
import pydrake
from pydrake.all import Meshcat, RigidTransform, RollPitchYaw, RotationMatrix
import config

from constants import SystemConstants

from collections import defaultdict
import scipy.interpolate

import plant.manipulator

class LinkFeedbackSetpointGenerator(pydrake.systems.framework.LeafSystem):
    """
    Generates an impedance trajectory relative to the edge of the link.
    joint.
    """
    def __init__(self, sys_consts: SystemConstants,
            num_links: config.NumLinks):
        pydrake.systems.framework.LeafSystem.__init__(self)

        self.sys_consts = sys_consts
        self.debug = defaultdict(list)
        orientation_map = np.load(config.base_path +
            "orientation_map_{}_links.npz".format(num_links.value))
        self.get_theta_Z_func = scipy.interpolate.interp1d(
            orientation_map['theta_Ls'],
            orientation_map['theta_L_EE'],
            fill_value='extrapolate'
        )

        # If I want the contact point to moving in an arc of
        # `desired_contact_distance`, then I need the end effector to be
        # moving in an arc of radius `desired_radius`.
        # TODO: switch to end effector frame at the tip of the end effector?
        desired_contact_distance = self.sys_consts.w_L/2
        desired_radius = np.sqrt(
            self.sys_consts.r**2 + desired_contact_distance**2)
        self.offset_y = desired_radius - desired_contact_distance

        # =========================== DECLARE INPUTS ==========================
        self.DeclareVectorInputPort(
            "pose_L_rotational", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "pose_L_translational", pydrake.systems.framework.BasicVector(3))

        # ========================== DECLARE OUTPUTS ==========================
        self.DeclareVectorOutputPort(
            "x0",
            pydrake.systems.framework.BasicVector(6),
            self.calc_x0
        )
        self.DeclareVectorOutputPort(
            "dx0",
            pydrake.systems.framework.BasicVector(6),
            self.calc_dx0
        )

    def calc_x0(self, context, output):
        x0 = np.zeros(6)

        # Evaluate inputs
        pose_L_rotational = self.GetInputPort(
            "pose_L_rotational").Eval(context)
        pose_L_translational = self.GetInputPort(
            "pose_L_translational").Eval(context)
        theta_L = pose_L_rotational[0]

        # Calc X_L_SP, the transform from the link to the setpoint
        offset_Z_rot = self.get_theta_Z_func(theta_L)
        X_L_SP = RigidTransform(
            R=RotationMatrix.MakeZRotation(offset_Z_rot),
            p=[0, self.offset_y, -(self.sys_consts.h_L/2+self.sys_consts.r)]
        )

        # Calc X_W_L
        X_W_L = RigidTransform(
            p=pose_L_translational,
            R=RotationMatrix(RollPitchYaw(pose_L_rotational))
        )

        # Calc X_W_SP
        X_W_SP = X_W_L.multiply(X_L_SP)

        translation = X_W_SP.translation()
        rotation = RollPitchYaw(X_W_SP.rotation()).vector()
        rotation[0] += plant.manipulator.RotX_L_Md

        x0[:3] = rotation
        x0[3:] = translation
        output.SetFromVector(x0)

        self.debug["offset_Z_rot"].append(offset_Z_rot)
        self.debug["times"].append(context.get_time())


    def calc_dx0(self, context, output):
        output.SetFromVector(np.zeros(6))
