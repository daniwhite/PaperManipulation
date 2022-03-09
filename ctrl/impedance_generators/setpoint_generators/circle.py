import numpy as np

import constants
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
    def __init__(self, sys_consts: constants.SystemConstants,
            desired_radius: float, end_time=60):
        pydrake.systems.framework.LeafSystem.__init__(self)

        self.desired_radius = desired_radius
        # System constants/parameters

        self.joint_position = [
            0,
            sys_consts.w_L - constants.PEDESTAL_X_DIM/2,
            plant.pedestal.PEDESTAL_Z_DIM + sys_consts.h_L/2
        ]

        # Bookkeeping terms
        self.start_time = 0
        self.end_time = end_time

        # Target values
        self.start_theta_MX = 0
        self.end_theta_MX = np.pi
        self.start_theta_MZ = 0
        self.end_theta_MZ = -np.pi/2


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
        x0 = self._calc_x0(context.get_time())
        output.SetFromVector(x0)




    def calc_dx0(self, context, output):
        output.SetFromVector(np.zeros(6))


    def _calc_x0(self, t):
        theta_MXd = np.interp(
           t,
            [self.start_time, self.end_time],
            [self.start_theta_MX, self.end_theta_MX],
        )
        theta_MZd = np.interp(
            t,
            [self.start_time, self.end_time/2],
            [self.start_theta_MZ, self.end_theta_MZ],
        )
        y_d = self.joint_position[1] + np.cos(theta_MXd)*self.desired_radius
        z_d = self.joint_position[2] + np.sin(theta_MXd)*self.desired_radius


        R_ = RotationMatrix.MakeXRotation(theta_MXd).multiply(
            RotationMatrix.MakeZRotation(theta_MZd)
        )
        x0 = np.zeros(6)
        x0[:3] = RollPitchYaw(R_).vector()
        x0[4] = y_d
        x0[5] = z_d
        
        return x0
