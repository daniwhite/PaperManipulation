import numpy as np

# Drake imports
import pydrake
from pydrake.all import MultibodyPlant, RigidTransform, RollPitchYaw, RotationMatrix, BodyIndex
import config

from constants import SystemConstants

from collections import defaultdict
import scipy.interpolate

import plant.manipulator as manipulator

class ForceFeedbackSetpointGenerator(pydrake.systems.framework.LeafSystem):
    """
    Generates an impedance trajectory relative to the edge of the link.
    joint.
    """
    def __init__(self, sys_consts: SystemConstants, contact_body_idx: int,
            num_links):
        pydrake.systems.framework.LeafSystem.__init__(self)

        self.sys_consts = sys_consts
        self.contact_body_idx = contact_body_idx
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

        # 0 is timestep, which doesn't matter
        self.manipulator_plant = MultibodyPlant(0)

        manipulator.data["add_plant_function"](
            self.manipulator_plant, self.sys_consts.m_M,
            self.sys_consts.r, self.sys_consts.mu)
        self.manipulator_plant.Finalize()
        self.manipulator_plant_context = \
            self.manipulator_plant.CreateDefaultContext()
        self.nq_manipulator = \
            self.manipulator_plant.get_actuation_input_port().size()

        # =========================== DECLARE INPUTS ==========================
        self.DeclareVectorInputPort(
            "q", pydrake.systems.framework.BasicVector(self.nq_manipulator))
        self.DeclareVectorInputPort(
            "F", pydrake.systems.framework.BasicVector(3))

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
        # Evaluate inputs
        q = self.GetInputPort("q").Eval(context)
        F = np.array(self.GetInputPort("F").Eval(context))
        F[config.hinge_rotation_axis] = 0

        # Get EE position
        self.manipulator_plant.SetPositions(
            self.manipulator_plant_context, q)
        self.manipulator_plant.SetVelocities(
            self.manipulator_plant_context, np.zeros_like(q))
        contact_body = self.manipulator_plant.GetBodyByName(
            manipulator.data["contact_body_name"])
        p_M = self.manipulator_plant.EvalBodyPoseInWorld(
            self.manipulator_plant_context, contact_body).translation()

        self.F = F
        #print(F)
        if np.linalg.norm(F) > 0:
            N_hat = F/np.linalg.norm(F)
            # TODO: double check hinge_rotation_axis dependency throughout this file
        else:
            N_hat = np.array([0,0,1])
        theta_L = np.arctan2(N_hat[2], -N_hat[0]) - np.pi/2
        #print(theta_L)

        X_W_L = RigidTransform(
            R = RotationMatrix.MakeYRotation(theta_L),
            p = p_M + (0.5 + self.sys_consts.r + self.sys_consts.h_L/2)*N_hat.flatten()
        )
        #print("X_W_L: ", X_W_L)

        # Calc X_L_SP, the transform from the link to the setpoint
        offset_Z_rot = self.get_theta_Z_func(theta_L)
        X_L_SP = RigidTransform(
            R=RotationMatrix.MakeZRotation(offset_Z_rot),
            p=[0, self.offset_y, -(self.sys_consts.h_L/2+self.sys_consts.r)]
        )
        #print("X_L_SP: ", X_L_SP)

        # Calc X_W_SP
        X_W_SP = X_W_L.multiply(X_L_SP)
        #print("X_W_SP: ", X_W_SP)

        translation = X_W_SP.translation()
        rotation = RollPitchYaw(X_W_SP.rotation()).vector()
        rotation[0] += manipulator.RotX_L_Md

        x0 = np.zeros(6)
        x0[:3] = rotation
        x0[3:] = translation
        # print("setting x0", x0)
        output.SetFromVector(x0)


    def calc_dx0(self, context, output):
        output.SetFromVector(np.zeros(6))
