import numpy as np

from constants import SystemConstants
import plant.manipulator

from collections import defaultdict

# Drake imports
import pydrake

class KinematicController(pydrake.systems.framework.LeafSystem):
    def __init__(self, sys_consts: SystemConstants):
        pydrake.systems.framework.LeafSystem.__init__(self)

        # System constants/parameters
        self.sys_consts = sys_consts
        self.nq = plant.manipulator.data["nq"]

        self.debug = defaultdict(list)

        # Bookkeeping terms
        self.last_time = None
        self.d_theta_L_integrator = 0

        # Target values
        self.d_Td = -self.sys_consts.w_L/2
        self.d_theta_Ld = 0.5
        self.d_Xd = 0
        self.theta_MYd = None
        self.theta_MZd = None

        self.debug = defaultdict(list)

        # =========================== DECLARE INPUTS ==========================
        # Force inputs
        self.DeclareVectorInputPort(
            "F_GN", pydrake.systems.framework.BasicVector(1))

        # Positions
        self.DeclareVectorInputPort(
            "theta_L", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "theta_MX", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "theta_MY", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "theta_MZ", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "d_T", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "d_X", pydrake.systems.framework.BasicVector(1))

        # Velocities
        self.DeclareVectorInputPort(
            "d_theta_L", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "d_theta_MX", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "d_theta_MY", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "d_theta_MZ", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "d_d_T", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "d_d_X", pydrake.systems.framework.BasicVector(1))

        # Other inputs
        self.DeclareVectorInputPort(
            "T_hat", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "N_hat", pydrake.systems.framework.BasicVector(3))

        # Manipulator inputs
        self.DeclareVectorInputPort(
            "J_translational",
            pydrake.systems.framework.BasicVector(3*self.nq))
        self.DeclareVectorInputPort(
            "J_rotational",
            pydrake.systems.framework.BasicVector(3*self.nq))

        # ========================== DECLARE OUTPUTS ==========================
        self.DeclareVectorOutputPort(
            "tau_out", pydrake.systems.framework.BasicVector(self.nq),
            self.CalcOutput)

        # =========================== DECLARE INPUTS ==========================

    def CalcOutput(self, context, output):
        # ============================ LOAD INPUTS ============================
        # Force inputs
        F_GN = self.GetInputPort("F_GN").Eval(context)[0]

        # Positions
        theta_L = self.GetInputPort("theta_L").Eval(context)[0]
        d_T = self.GetInputPort("d_T").Eval(context)[0]
        d_X = self.GetInputPort("d_X").Eval(context)[0]
        theta_MX = self.GetInputPort("theta_MX").Eval(context)[0]
        theta_MY = self.GetInputPort("theta_MY").Eval(context)[0]
        theta_MZ = self.GetInputPort("theta_MZ").Eval(context)[0]

        # Velocities
        d_theta_L = self.GetInputPort("d_theta_L").Eval(context)[0]
        d_d_T = self.GetInputPort("d_d_T").Eval(context)[0]
        d_d_X = self.GetInputPort("d_d_X").Eval(context)[0]
        d_theta_MX = self.GetInputPort("d_theta_MX").Eval(context)[0]
        d_theta_MY = self.GetInputPort("d_theta_MY").Eval(context)[0]
        d_theta_MZ = self.GetInputPort("d_theta_MZ").Eval(context)[0]

        # Other inputs
        T_hat = np.array([self.GetInputPort("T_hat").Eval(context)]).T
        N_hat = np.array([self.GetInputPort("N_hat").Eval(context)]).T

        # Manipulator inputs
        J_translational = np.array(
            self.GetInputPort("J_translational").Eval(context)).reshape(
                (3, self.nq))
        J_rotational = np.array(
            self.GetInputPort("J_rotational").Eval(context)).reshape(
                (3, self.nq))

        # ==================== CALCULATE INTERMEDIATE TERMS ===================
        if self.theta_MYd is None:
            self.theta_MYd = theta_MY
        if self.theta_MZd is None:
            self.theta_MZd = theta_MZ

        current_time = context.get_time()
        if self.last_time is None:
            dt = 0
        else:
            dt = current_time - self.last_time
        self.last_time = current_time
        self.d_theta_L_integrator += dt*(self.d_theta_Ld - d_theta_L)

        tau_O_est = -self.sys_consts.k_J*theta_L - \
            self.sys_consts.b_J*d_theta_L

        ff_term = -F_GN - tau_O_est/(self.sys_consts.w_L/2)
        
        # ===================== CALCULATE CONTROL OUTPUTS =====================
        F_CT = 1000*(self.d_Td - d_T) - 100*d_d_T
        F_CN = 10*(self.d_theta_Ld - d_theta_L) + ff_term
        F_CX = 100*(self.d_Xd - d_X) - 10 * d_d_X
        theta_MXd = theta_L
        tau_X = 100*(theta_MXd - theta_MX)  + 10*(d_theta_L - d_theta_MX)
        tau_Y = 10*(self.theta_MYd - theta_MY) - 5*d_theta_MY
        tau_Z = 10*(self.theta_MZd - theta_MZ) - 5*d_theta_MZ

        F = F_CT*T_hat + F_CN*N_hat + F_CX * np.array([[1, 0, 0]]).T
        tau = np.array([[tau_X, tau_Y, tau_Z]]).T

        tau_ctrl = np.matmul(J_translational.T, F) \
            + np.matmul(J_rotational.T, tau)

        # ======================== UPDATE DEBUG VALUES ========================
        self.debug["theta_MXd"].append(theta_MXd)
        self.debug["times"].append(current_time)

        output.SetFromVector(tau_ctrl.flatten())
