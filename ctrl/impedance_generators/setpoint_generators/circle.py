import numpy as np

from ctrl.common import SystemConstants
import plant.manipulator
import plant.pedestal

# Drake imports
import pydrake
from pydrake.all import Meshcat, RigidTransform, RollPitchYaw, RotationMatrix

class CircularSetpointGenerator(pydrake.systems.framework.LeafSystem):
    """
    Generates an impedance trajectory that follows a circular path about the
    joint.
    """
    def __init__(self, sys_consts: SystemConstants, desired_radius: float,
            end_time=20.0):
        pydrake.systems.framework.LeafSystem.__init__(self)

        self.desired_radius = desired_radius
        # System constants/parameters

        self.joint_position = [
            0,
            sys_consts.w_L - plant.pedestal.PEDESTAL_WIDTH/2,
            plant.pedestal.PEDESTAL_HEIGHT + sys_consts.h_L/2
        ]

        # Bookkeeping terms
        self.start_time = 0
        self.end_time = end_time

        # Target values
        self.start_theta_MX = -0.25
        self.end_theta_MX = np.pi


        # =========================== DECLARE INPUTS ==========================
        # No inputs

        # ========================== DECLARE OUTPUTS ==========================
        # Impedance setpoint
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
        theta_MXd = np.interp(
            context.get_time(),
            [self.start_time, self.end_time],
            [self.start_theta_MX, self.end_theta_MX],
        )
        y_d = self.joint_position[1] + np.cos(theta_MXd)*self.desired_radius
        z_d = self.joint_position[2] + np.sin(theta_MXd)*self.desired_radius

        x0 = np.zeros(6)
        x0[0] = theta_MXd
        x0[4] = y_d
        x0[5] = z_d
        output.SetFromVector(x0)


    def calc_dx0(self, context, output):
        output.SetFromVector(np.zeros(6))
