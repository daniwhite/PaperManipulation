import numpy as np

from ctrl.common import SystemConstants
import plant.manipulator

from collections import defaultdict

# Drake imports
import pydrake

class CartesianImpedanceController(pydrake.systems.framework.LeafSystem):
    def __init__(self, sys_consts: SystemConstants):
        pydrake.systems.framework.LeafSystem.__init__(self)

        # System constants/parameters
        self.sys_consts = sys_consts
        self.nq = plant.manipulator.data["nq"]

        self.debug = defaultdict(list)

        # Other terms
        self.x0 = np.zeros(6)


        # =========================== DECLARE INPUTS ==========================
        # Positions and velocities
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

        # Impedance
        self.DeclareVectorInputPort(
            "K",
            pydrake.systems.framework.BasicVector(6)
        )
        self.DeclareVectorInputPort(
            "D",
            pydrake.systems.framework.BasicVector(6)
        )
        self.DeclareVectorInputPort(
            "x0",
            pydrake.systems.framework.BasicVector(6)
        )
        self.DeclareVectorInputPort(
            "dx0",
            pydrake.systems.framework.BasicVector(6)
        )
        self.DeclareVectorInputPort(
            "feedforward_wrench",
            pydrake.systems.framework.BasicVector(6)
        )

        # ========================== DECLARE OUTPUTS ==========================
        self.DeclareVectorOutputPort(
            "tau_out", pydrake.systems.framework.BasicVector(self.nq),
            self.CalcOutput)
        self.DeclareVectorOutputPort(
            "adjusted_x0_pos", pydrake.systems.framework.BasicVector(3),
            self.calc_adjusted_x0_pos)
        self.DeclareVectorOutputPort(
            "adjusted_x0_rot", pydrake.systems.framework.BasicVector(3),
            self.calc_adjusted_x0_rot)

    def CalcOutput(self, context, output):
        # ============================ LOAD INPUTS ============================
        # Positions and velocities
        rot_vec_M = self.GetInputPort("pose_M_rotational").Eval(context)
        p_M = self.GetInputPort("pose_M_translational").Eval(context)

        omega_M = self.GetInputPort("vel_M_rotational").Eval(context)
        v_M = self.GetInputPort("vel_M_translational").Eval(context)

        # Manipulator inputs
        M = np.array(self.GetInputPort("M").Eval(context)).reshape(
            (self.nq, self.nq))
        J = np.array(self.GetInputPort("J").Eval(context)).reshape(
            (6, self.nq))
        Jdot_qdot = np.expand_dims(
            np.array(self.GetInputPort("Jdot_qdot").Eval(context)), 1)
        Cv = np.expand_dims(
            np.array(self.GetInputPort("Cv").Eval(context)), 1)

        # Impedance
        K_flat = self.GetInputPort("K").Eval(context)
        K = np.diag(K_flat)
        D = np.diag(self.GetInputPort("D").Eval(context))
        x0 = np.expand_dims(self.GetInputPort("x0").Eval(context), 1)
        dx0 = np.expand_dims(self.GetInputPort("dx0").Eval(context), 1)

        # Feedforward forces
        ff_wrench = self.GetInputPort("feedforward_wrench").Eval(context)

        # ==================== CALCULATE INTERMEDIATE TERMS ===================
        # Actual values
        d_x = np.expand_dims(np.array(list(omega_M) + list(v_M)), 1)
        x = np.expand_dims(np.array(list(rot_vec_M) + list(p_M)), 1)

        Mq = M
        # TODO: Should this be pinv? (Copying from Sangbae's notes)
        Mx = np.linalg.pinv(
                np.matmul(
                    np.matmul(
                        J,
                        np.linalg.pinv(Mq)
                    ), 
                    J.T
                )
            )

        # K_flat is N/m, since F = kx and F is N and x is m
        # ff_wrench is N
        # so ff_wrench / K_flat is (N) / (N/m) = N * m/N = ,
        ff_diff = ff_wrench / K_flat
        x0 += np.expand_dims(ff_diff, 1)
    
        # ===================== CALCULATE CONTROL OUTPUTS =====================
        Vq = Cv
        compliance_terms = np.matmul(D, dx0 - d_x) \
            + \
            np.matmul(K, x0 - x)
        cancelation_terms = np.matmul(
            Mx,
            np.matmul(
                np.matmul(J, np.linalg.inv(Mq)),
                Vq
            ) \
            - \
            Jdot_qdot
        )
        F_ctrl = cancelation_terms + compliance_terms
        tau_ctrl = np.matmul(J.T, F_ctrl)

        # ======================== UPDATE DEBUG VALUES ========================
        self.debug["dx0"].append(dx0)
        self.debug["x0"].append(x0)
        self.debug["times"].append(context.get_time())

        self.x0 = x0

        output.SetFromVector(tau_ctrl.flatten())

    def calc_adjusted_x0_pos(self, context, output):
        output.SetFromVector(self.x0[3:])

    def calc_adjusted_x0_rot(self, context, output):
        output.SetFromVector(self.x0[:3])
