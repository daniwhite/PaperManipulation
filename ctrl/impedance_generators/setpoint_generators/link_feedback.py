import numpy as np

# Drake imports
import pydrake
from pydrake.all import Meshcat, RigidTransform, RollPitchYaw, RotationMatrix

from ctrl.common import SystemConstants

from collections import defaultdict

import plant.manipulator

class LinkFeedbackSetpointGenerator(pydrake.systems.framework.LeafSystem):
    """
    Generates an impedance trajectory relative to the edge of the link.
    joint.
    """
    def __init__(self, sys_consts: SystemConstants):
        pydrake.systems.framework.LeafSystem.__init__(self)

        self.sys_consts = sys_consts
        self.debug = defaultdict(list)

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

    def smoothing_func(self, x):
        """
        Function for smoothing desired theta_Z setpoint.

        Without smoothing, I want the following behavior:
             /--
             | -pi/2    x < -pi/2
        x = <  0        x > 0
             | x        otherwise
             \--

        However, the point where the derivative because discontinuous,
        a very large torque is commanded. So I implement the following
        smoothing function to give the behavior above.

        It's a member function so that I can also evaluate while debugging.
        """
        a = 4/np.pi
        sig_term = 1/(1+np.exp(-a*x))
        return (sig_term-0.5)*(4/a)

    def calc_x0(self, context, output):
        x0 = np.zeros(6)

        # Evaluate inputs
        pose_L_rotational = self.GetInputPort(
            "pose_L_rotational").Eval(context)
        pose_L_translational = self.GetInputPort(
            "pose_L_translational").Eval(context)
        theta_L = pose_L_rotational[0]

        # Calc X_L_SP, the transform from the link to the setpoint
        offset_Z_rot = self.smoothing_func(-theta_L)
        X_L_SP = RigidTransform(
            R=RotationMatrix.MakeZRotation(offset_Z_rot),
            p=[0, 0, -(self.sys_consts.h_L/2+self.sys_consts.r)]
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
