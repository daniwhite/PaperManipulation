import numpy as np

from ctrl.common import SystemConstants
import plant.manipulator
import plant.pedestal
import constants
import visualization

from collections import defaultdict

# Drake imports
import pydrake
from pydrake.all import Meshcat, RigidTransform, RollPitchYaw, RotationMatrix

class CartesianImpedanceController(pydrake.systems.framework.LeafSystem):
    def __init__(self, sys_consts: SystemConstants, meshcat: Meshcat):
        pydrake.systems.framework.LeafSystem.__init__(self)

        # System constants/parameters
        self.sys_consts = sys_consts
        self.nq = plant.manipulator.data["nq"]
        self.joint_position = [
            0,
            self.sys_consts.w_L - plant.pedestal.PEDESTAL_WIDTH/2,
            plant.pedestal.PEDESTAL_HEIGHT + self.sys_consts.h_L/2
        ]
        self.desired_radius = sys_consts.w_L/2

        self.debug = defaultdict(list)

        # Bookkeeping terms
        self.initialized = False
        self.start_time = None
        self.end_time = 20
        
        # Trajectory_params
        self.final_theta_L = np.pi

        # Target values
        self.start_theta_MX = -0.25
        self.end_theta_MX = np.pi

        # Gains
        # Order is [theta_x, theta_y, theta_z, x, y, z]
        self.K = np.diag([100, 10, 10, 100, 100, 100])
        self.D = np.diag([10, 5, 5, 10, 10, 10])
        self.x0 = np.zeros((6,1))
        self.dx0 = np.zeros((6,1))

        # Other terms
        self.debug = defaultdict(list)
        self.meshcat = meshcat


        # =========================== DECLARE INPUTS ==========================
        # Positions
        self.DeclareVectorInputPort(
            "pose_M_translational", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "pose_M_rotational", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "vel_M_translational", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "vel_M_rotational", pydrake.systems.framework.BasicVector(3))

        # Manipulator inputs
        self.DeclareVectorInputPort(
            "M",
            pydrake.systems.framework.BasicVector(
                self.nq*self.nq))
        self.DeclareVectorInputPort(
            "J",
            pydrake.systems.framework.BasicVector(6*self.nq))
        self.DeclareVectorInputPort(
            "Jdot_qdot", pydrake.systems.framework.BasicVector(6))
        self.DeclareVectorInputPort(
            "Cv",
            pydrake.systems.framework.BasicVector(self.nq))

        # ========================== DECLARE OUTPUTS ==========================
        self.DeclareVectorOutputPort(
            "tau_out", pydrake.systems.framework.BasicVector(self.nq),
            self.CalcOutput)

    def CalcOutput(self, context, output):
        # ============================ LOAD INPUTS ============================
        # Manipulator inputs
        M = np.array(self.GetInputPort("M").Eval(context)).reshape(
            (self.nq, self.nq))
        J = np.array(self.GetInputPort("J").Eval(context)).reshape(
            (6, self.nq))
        Jdot_qdot = np.expand_dims(
            np.array(self.GetInputPort("Jdot_qdot").Eval(context)), 1)
        Cv = np.expand_dims(
            np.array(self.GetInputPort("Cv").Eval(context)), 1)

        rot_vec_M = self.GetInputPort("pose_M_rotational").Eval(context)
        p_M = self.GetInputPort("pose_M_translational").Eval(context)

        omega_M = self.GetInputPort("vel_M_rotational").Eval(context)
        v_M = self.GetInputPort("vel_M_translational").Eval(context)

        # =========================== INITIALIZATION ==========================
        if not self.initialized:
            self.start_time = context.get_time()
            self.d_theta_MXd = (self.end_theta_MX - self.start_theta_MX) / (
                self.end_time - self.start_time)
            # self.dx0[0] = self.d_theta_MXd
        self.initialized = True

        # ==================== CALCULATE INTERMEDIATE TERMS ===================
        # Desired values
        ## Interpolate theta_MXd
        theta_MXd = np.interp(
            context.get_time(),
            [self.start_time, self.end_time],
            [self.start_theta_MX, self.end_theta_MX],
        )
        y_d = self.joint_position[1] + np.cos(theta_MXd)*self.sys_consts.w_L/2
        z_d = self.joint_position[2] + np.sin(theta_MXd)*self.sys_consts.w_L/2
        self.x0[0] = theta_MXd
        self.x0[4] = y_d
        self.x0[5] = z_d

        # Actual values
        d_x = np.expand_dims(np.array(list(omega_M) + list(v_M)), 1)
        x = np.expand_dims(np.array(list(rot_vec_M) + list(p_M)), 1)

        Mq = M
        # TODO: Should this be pinv? (Copying from Sangbae's notes)
        Mx = np.linalg.inv(
                np.matmul(
                    np.matmul(
                        J,
                        np.linalg.inv(Mq)
                    ), 
                    J.T
                )
            )

        # print(Mx.shape)
        # print(J.shape)
        # ===================== CALCULATE CONTROL OUTPUTS =====================
        # print(np.matmul(np.linalg.inv(Mq), Cv).shape)
        tau_ctrl = np.matmul(J.T,
            np.matmul(self.D, self.dx0 - d_x) \
            + \
            np.matmul(self.K, self.x0 - x) \
            # + \
            # np.matmul(Mx,
            #     np.matmul(J, np.matmul(np.linalg.inv(Mq), Cv))
            #     -
            #     Jdot_qdot
            # )
        )

        # ======================== UPDATE DEBUG VALUES ========================
        self.debug["dx0"].append(self.dx0)
        self.debug["x0"].append(self.x0)
        self.debug["theta_MXd"].append(theta_MXd)
        self.debug["times"].append(context.get_time())

        # ======================== UPDATE VISUALIZATION =======================
        visualization.AddMeshcatTriad(
            self.meshcat, "impedance_setpoint",
            X_PT=RigidTransform(
                p=self.x0[3:].flatten(),
                R=RotationMatrix(RollPitchYaw(self.x0[:3]))
            )
        )

        output.SetFromVector(tau_ctrl.flatten())
