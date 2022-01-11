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

        self.debug = defaultdict(list)

        # Other terms
        self.debug = defaultdict(list)
        self.meshcat = meshcat


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

        # ========================== DECLARE OUTPUTS ==========================
        self.DeclareVectorOutputPort(
            "tau_out", pydrake.systems.framework.BasicVector(self.nq),
            self.CalcOutput)

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
        K = np.diag(self.GetInputPort("K").Eval(context))
        D = np.diag(self.GetInputPort("D").Eval(context))
        x0 = np.expand_dims(self.GetInputPort("x0").Eval(context), 1)
        dx0 = np.expand_dims(self.GetInputPort("dx0").Eval(context), 1)

        # ==================== CALCULATE INTERMEDIATE TERMS ===================
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
    
        # ===================== CALCULATE CONTROL OUTPUTS =====================
        # print(np.matmul(np.linalg.inv(Mq), Cv).shape)
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

        # ======================== UPDATE VISUALIZATION =======================
        visualization.AddMeshcatTriad(
            self.meshcat, "impedance_setpoint",
            X_PT=RigidTransform(
                p=x0[3:].flatten(),
                R=RotationMatrix(RollPitchYaw(x0[:3]))
            )
        )

        output.SetFromVector(tau_ctrl.flatten())
