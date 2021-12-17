import numpy as np

from ctrl.common import SystemConstants
import plant.manipulator

from collections import defaultdict

# Drake imports
import pydrake
from pydrake.all import (
    MathematicalProgram, eq, Solve
)

class InverseDynamicsController(pydrake.systems.framework.LeafSystem):
    def __init__(self, options, sys_consts: SystemConstants):
        pydrake.systems.framework.LeafSystem.__init__(self)

        # System constants/parameters
        self.sys_consts = sys_consts
        self.nq = plant.manipulator.data["nq"]

        # Options
        self.model_friction = options['model_friction']
        self.measure_joint_wrench = options['measure_joint_wrench']

        # Target values
        self.d_Td = -self.sys_consts.w_L/2
        self.d_theta_Ld = 0.5
        self.d_Xd = 0
        self.theta_MYd = None
        self.theta_MZd = None

        self.debug = defaultdict(list)

        # =========================== DECLARE INPUTS ==========================
        # Torque inputs
        self.DeclareVectorInputPort(
            "tau_g",
            pydrake.systems.framework.BasicVector(self.nq))
        self.DeclareVectorInputPort(
            "joint_centering_torque",
            pydrake.systems.framework.BasicVector(self.nq))

        # Force inputs
        self.DeclareVectorInputPort(
            "F_GT", pydrake.systems.framework.BasicVector(1))
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
            "p_CT", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "p_CN", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "p_LT", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "p_LN", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "d_T", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "d_N", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "d_X", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "p_MConM", pydrake.systems.framework.BasicVector(3))
        
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
            "d_d_N", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "d_d_X", pydrake.systems.framework.BasicVector(1))

        # Manipulator inputs
        self.DeclareVectorInputPort(
            "J",
            pydrake.systems.framework.BasicVector(6*self.nq))
        self.DeclareVectorInputPort(
            "J_translational",
            pydrake.systems.framework.BasicVector(3*self.nq))
        self.DeclareVectorInputPort(
            "J_rotational",
            pydrake.systems.framework.BasicVector(3*self.nq))
        self.DeclareVectorInputPort(
            "Jdot_qdot", pydrake.systems.framework.BasicVector(6))
        self.DeclareVectorInputPort(
            "M",
            pydrake.systems.framework.BasicVector(
                self.nq*self.nq))
        self.DeclareVectorInputPort(
            "Cv",
            pydrake.systems.framework.BasicVector(self.nq))

        # Other inputs
        self.DeclareVectorInputPort(
            "mu_S", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "hats_T", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "s_hat_X", pydrake.systems.framework.BasicVector(1))

        # ========================== DECLARE OUTPUTS ==========================
        self.DeclareVectorOutputPort(
            "tau_out", pydrake.systems.framework.BasicVector(self.nq),
            self.CalcOutput)

    def CalcOutput(self, context, output):
        # ============================ LOAD INPUTS ============================
        # Torque inputs
        tau_g = np.expand_dims(
            np.array(self.GetInputPort("tau_g").Eval(context)), 1)
        joint_centering_torque = np.expand_dims(np.array(self.GetInputPort(
            "joint_centering_torque").Eval(context)), 1)

        # Force inputs
        F_GT = self.GetInputPort("F_GT").Eval(context)
        F_GN = self.GetInputPort("F_GN").Eval(context)

        # Positions
        theta_L = self.GetInputPort("theta_L").Eval(context)
        theta_MX = self.GetInputPort("theta_MX").Eval(context)[0]
        theta_MY = self.GetInputPort("theta_MY").Eval(context)[0]
        theta_MZ = self.GetInputPort("theta_MZ").Eval(context)[0]
        p_CT = self.GetInputPort("p_CT").Eval(context)
        p_CN = self.GetInputPort("p_CN").Eval(context)
        p_LT = self.GetInputPort("p_LT").Eval(context)
        p_LN = self.GetInputPort("p_LN").Eval(context)
        d_T = self.GetInputPort("d_T").Eval(context)
        d_N = self.GetInputPort("d_N").Eval(context)
        d_X = self.GetInputPort("d_X").Eval(context)
        p_MConM = np.array([self.GetInputPort("p_MConM").Eval(context)]).T

        # Velocities
        d_theta_L = self.GetInputPort("d_theta_L").Eval(context)
        d_theta_MX = self.GetInputPort("d_theta_MX").Eval(context)
        d_theta_MY = self.GetInputPort("d_theta_MY").Eval(context)
        d_theta_MZ = self.GetInputPort("d_theta_MZ").Eval(context)
        d_d_T = self.GetInputPort("d_d_T").Eval(context)
        d_d_N = self.GetInputPort("d_d_N").Eval(context)
        d_d_X = self.GetInputPort("d_d_X").Eval(context)

        # Manipulator inputs
        J = np.array(self.GetInputPort("J").Eval(context)).reshape(
            (6, self.nq))
        J_translational = np.array(
            self.GetInputPort("J_translational").Eval(context)).reshape(
                (3, self.nq))
        J_rotational = np.array(
            self.GetInputPort("J_rotational").Eval(context)).reshape(
                (3, self.nq))
        Jdot_qdot = np.expand_dims(
            np.array(self.GetInputPort("Jdot_qdot").Eval(context)), 1)
        M = np.array(self.GetInputPort("M").Eval(context)).reshape(
            (self.nq, self.nq))
        Cv = np.expand_dims(
            np.array(self.GetInputPort("Cv").Eval(context)), 1)

        # Other inputs
        mu_S = self.GetInputPort("mu_S").Eval(context)
        hats_T = self.GetInputPort("hats_T").Eval(context)
        s_hat_X = self.GetInputPort("s_hat_X").Eval(context)

        # ============================= OTHER PREP ============================
        if self.theta_MYd is None:
            self.theta_MYd = theta_MY
        if self.theta_MZd is None:
            self.theta_MZd = theta_MZ

        # Calculate desired values
        dd_d_Td = 1000*(self.d_Td - d_T) - 100*d_d_T
        dd_theta_Ld = 10*(self.d_theta_Ld - d_theta_L)
        a_MX_d = 100*(self.d_Xd - d_X) - 10*d_d_X
        theta_MXd = theta_L
        alpha_MXd = 100*(theta_MXd - theta_MX)  + 10*(d_theta_L - d_theta_MX)
        alpha_MYd = 10*(self.theta_MYd - theta_MY) - 5*d_theta_MY
        alpha_MZd = 10*(self.theta_MZd - theta_MZ) - 5*d_theta_MZ
        dd_d_Nd = 0

        # =========================== SOLVE PROGRAM ===========================
        ## 1. Define an instance of MathematicalProgram
        prog = MathematicalProgram()

        ## 2. Add decision variables
        # Contact values
        F_NM = prog.NewContinuousVariables(1, 1, name="F_NM")
        F_ContactMY = prog.NewContinuousVariables(1, 1, name="F_ContactMY")
        F_ContactMZ = prog.NewContinuousVariables(1, 1, name="F_ContactMZ")
        F_NL = prog.NewContinuousVariables(1, 1, name="F_NL")
        if self.model_friction:
            F_FMT = prog.NewContinuousVariables(1, 1, name="F_FMT")
            F_FMX =prog.NewContinuousVariables(1, 1, name="F_FMX")
            F_FLT =prog.NewContinuousVariables(1, 1, name="F_FLT")
            F_FLX =prog.NewContinuousVariables(1, 1, name="F_FLX")
        else:
            F_FMT = np.array([[0]])
            F_FMX = np.array([[0]])
            F_FLT = np.array([[0]])
            F_FLX = np.array([[0]])
        F_ContactM_XYZ = np.array([F_FMX, F_ContactMY, F_ContactMZ])[:,:,0]

        # Object forces and torques
        if not self.measure_joint_wrench:
            F_OT = prog.NewContinuousVariables(1, 1, name="F_OT")
            F_ON = prog.NewContinuousVariables(1, 1, name="F_ON")
            tau_O = -self.sys_consts.k_J*theta_L \
                    - self.sys_consts.b_J*d_theta_L

        # Control values
        tau_ctrl = prog.NewContinuousVariables(
            self.nq, 1, name="tau_ctrl")

        # Object accelerations
        a_MX = prog.NewContinuousVariables(1, 1, name="a_MX")
        a_MT = prog.NewContinuousVariables(1, 1, name="a_MT")
        a_MY = prog.NewContinuousVariables(1, 1, name="a_MY")
        a_MZ = prog.NewContinuousVariables(1, 1, name="a_MZ")
        a_MN = prog.NewContinuousVariables(1, 1, name="a_MN")
        a_LT = prog.NewContinuousVariables(1, 1, name="a_LT")
        a_LN = prog.NewContinuousVariables(1, 1, name="a_LN")
        alpha_MX = prog.NewContinuousVariables(1, 1, name="alpha_MX")
        alpha_MY = prog.NewContinuousVariables(1, 1, name="alpha_MY")
        alpha_MZ = prog.NewContinuousVariables(1, 1, name="alpha_MZ")
        alpha_a_MXYZ = np.array(
            [alpha_MX, alpha_MY, alpha_MZ, a_MX, a_MY, a_MZ])[:,:,0]

        # Derived accelerations
        dd_theta_L = prog.NewContinuousVariables(1, 1, name="dd_theta_L")
        dd_d_N = prog.NewContinuousVariables(1, 1, name="dd_d_N")
        dd_d_T = prog.NewContinuousVariables(1, 1, name="dd_d_T")

        ddq = prog.NewContinuousVariables(self.nq, 1, name="ddq")

        ## Constraints
        # "set_description" calls gives us useful names for printing
        prog.AddConstraint(eq(
            self.sys_consts.m_L*a_LT, F_FLT+F_GT+F_OT
        )).evaluator().set_description("Link tangential force balance")
        prog.AddConstraint(eq(
            self.sys_consts.m_L*a_LN, F_NL + F_GN + F_ON
        )).evaluator().set_description("Link normal force balance") 
        prog.AddConstraint(eq(
            self.sys_consts.I_L*dd_theta_L, \
            (-self.sys_consts.w_L/2)*F_ON - (p_CN-p_LN) * F_FLT + \
                (p_CT-p_LT)*F_NL + tau_O
        )).evaluator().set_description("Link moment balance") 
        prog.AddConstraint(eq(
            F_NL, -F_NM
        )).evaluator().set_description("3rd law normal forces")
        if self.model_friction:
            prog.AddConstraint(eq(F_FMT, -F_FLT)).evaluator().set_description(
                    "3rd law friction forces (T hat)") 
            prog.AddConstraint(eq(F_FMX, -F_FLX)).evaluator().set_description(
                    "3rd law friction forces (X hat)") 
        prog.AddConstraint(eq(
            -dd_theta_L*(self.sys_consts.h_L/2+self.sys_consts.r) + \
                d_theta_L**2*self.sys_consts.w_L/2 - a_LT + a_MT,
            -dd_theta_L*d_N + dd_d_T - d_theta_L**2*d_T - 2*d_theta_L*d_d_N
        )).evaluator().set_description("d_N derivative is derivative")
        prog.AddConstraint(eq(
            -dd_theta_L*self.sys_consts.w_L/2 - \
                d_theta_L**2*self.sys_consts.h_L/2 - \
                d_theta_L**2*self.sys_consts.r - a_LN + a_MN,
            dd_theta_L*d_T + dd_d_N - d_theta_L**2*d_N + 2*d_theta_L*d_d_T
        )).evaluator().set_description("d_N derivative is derivative") 
        prog.AddConstraint(eq(
            dd_d_N, 0
        )).evaluator().set_description("No penetration")
        if self.model_friction:
            prog.AddConstraint(eq(
                F_FLT, mu_S*F_NL*self.sys_consts.mu*hats_T
            )).evaluator().set_description("Friction relationship LT")
            prog.AddConstraint(eq(
                F_FLX, mu_S*F_NL*self.sys_consts.mu*s_hat_X
            )).evaluator().set_description("Friction relationship LX")
        
        if not self.measure_joint_wrench:
            prog.AddConstraint(eq(
                a_LT, -(self.sys_consts.w_L/2)*d_theta_L**2
            )).evaluator().set_description("Hinge constraint (T hat)")
            prog.AddConstraint(eq(
                a_LN, (self.sys_consts.w_L/2)*dd_theta_L
            )).evaluator().set_description("Hinge constraint (N hat)")
        
        for i in range(6):
            lhs_i = alpha_a_MXYZ[i,0]
            assert not hasattr(lhs_i, "shape")
            rhs_i = (Jdot_qdot + np.matmul(J, ddq))[i,0]
            assert not hasattr(rhs_i, "shape")
            prog.AddConstraint(lhs_i == rhs_i).evaluator().set_description(
                "Relate manipulator and end effector with joint " + \
                "accelerations " + str(i)) 

        tau_contact_trn = np.matmul(
            J_translational.T, F_ContactM_XYZ)
        tau_contact_rot = np.matmul(
            J_rotational.T, np.cross(p_MConM, F_ContactM_XYZ, axis=0))
        tau_contact = tau_contact_trn + tau_contact_rot
        tau_out = tau_ctrl - tau_g + joint_centering_torque
        for i in range(self.nq):
            M_ddq_row_i = (np.matmul(M, ddq) + Cv)[i,0]
            assert not hasattr(M_ddq_row_i, "shape")
            tau_i = (tau_g + tau_contact + tau_out)[i,0]
            assert not hasattr(tau_i, "shape")
            prog.AddConstraint(
                M_ddq_row_i == tau_i
            ).evaluator().set_description("Manipulator equations " + str(i))
        
        # Projection equations
        prog.AddConstraint(eq(
            a_MT, np.cos(theta_L)*a_MY + np.sin(theta_L)*a_MZ
        ))
        prog.AddConstraint(eq(
            a_MN, -np.sin(theta_L)*a_MY + np.cos(theta_L)*a_MZ
        ))
        prog.AddConstraint(eq(
            F_FMT, np.cos(theta_L)*F_ContactMY + np.sin(theta_L)*F_ContactMZ
        ))
        prog.AddConstraint(eq(
            F_NM, -np.sin(theta_L)*F_ContactMY + np.cos(theta_L)*F_ContactMZ
        ))

        prog.AddConstraint(
            dd_d_T[0,0] == dd_d_Td
        ).evaluator().set_description("Desired dd_d_Td constraint" + str(i))
        prog.AddConstraint(
            dd_theta_L[0,0] == dd_theta_Ld
        ).evaluator().set_description("Desired a_LN constraint" + str(i))
        prog.AddConstraint(
            a_MX[0,0] == a_MX_d
        ).evaluator().set_description("Desired a_MX constraint" + str(i))
        prog.AddConstraint(
            alpha_MX[0,0] == alpha_MXd
        ).evaluator().set_description("Desired alpha_MX constraint" + str(i))
        prog.AddConstraint(
            alpha_MY[0,0] == alpha_MYd
        ).evaluator().set_description("Desired alpha_MY constraint" + str(i))
        prog.AddConstraint(
            alpha_MZ[0,0] == alpha_MZd
        ).evaluator().set_description("Desired alpha_MZ constraint" + str(i))
        prog.AddConstraint(
            dd_d_N[0,0] == dd_d_Nd
        ).evaluator().set_description("Desired dd_d_N constraint" + str(i))

        result = Solve(prog)
        assert result.is_success()
        tau_ctrl_result = []
        for i in range(self.nq):
            tau_ctrl_result.append(result.GetSolution()[
                prog.FindDecisionVariableIndex(tau_ctrl[i,0])])
        tau_ctrl_result = np.expand_dims(tau_ctrl_result, 1)

        # %DEBUG_APPEND%
        # control effort
        self.debug["dd_d_Td"].append(dd_d_Td)
        self.debug["dd_theta_Ld"].append(dd_theta_Ld)
        self.debug["a_MX_d"].append(a_MX_d)
        self.debug["alpha_MXd"].append(alpha_MXd)
        self.debug["alpha_MYd"].append(alpha_MYd)
        self.debug["alpha_MZd"].append(alpha_MZd)
        self.debug["dd_d_Nd"].append(dd_d_Nd)

        # decision variables
        if self.model_friction:
            self.debug["F_FMT"].append(result.GetSolution()[
                prog.FindDecisionVariableIndex(F_FMT[0,0])])
            self.debug["F_FMX"].append(result.GetSolution()[
                prog.FindDecisionVariableIndex(F_FMX[0,0])])
            self.debug["F_FLT"].append(result.GetSolution()[
                prog.FindDecisionVariableIndex(F_FLT[0,0])])
            self.debug["F_FLX"].append(result.GetSolution()[
                prog.FindDecisionVariableIndex(F_FLX[0,0])])
        else:
            self.debug["F_FMT"].append(F_FMT)
            self.debug["F_FMX"].append(F_FMX)
            self.debug["F_FLT"].append(F_FLT)
            self.debug["F_FLX"].append(F_FLX)
        self.debug["F_NM"].append(result.GetSolution()[
            prog.FindDecisionVariableIndex(F_NM[0,0])])
        self.debug["F_ContactMY"].append(result.GetSolution()[
            prog.FindDecisionVariableIndex(F_ContactMY[0,0])])
        self.debug["F_ContactMZ"].append(result.GetSolution()[
            prog.FindDecisionVariableIndex(F_ContactMZ[0,0])])
        self.debug["F_NL"].append(result.GetSolution()[
            prog.FindDecisionVariableIndex(F_NL[0,0])])
        for i in range(self.nq):
            self.debug["tau_ctrl_" + str(i)].append(result.GetSolution()[prog.FindDecisionVariableIndex(tau_ctrl[i,0])])
        self.debug["a_MX"].append(result.GetSolution()[
            prog.FindDecisionVariableIndex(a_MX[0,0])])
        self.debug["a_MT"].append(result.GetSolution()[
            prog.FindDecisionVariableIndex(a_MT[0,0])])
        self.debug["a_MY"].append(result.GetSolution()[
            prog.FindDecisionVariableIndex(a_MY[0,0])])
        self.debug["a_MZ"].append(result.GetSolution()[
            prog.FindDecisionVariableIndex(a_MZ[0,0])])
        self.debug["a_MN"].append(result.GetSolution()[
            prog.FindDecisionVariableIndex(a_MN[0,0])])
        self.debug["a_LT"].append(result.GetSolution()[
            prog.FindDecisionVariableIndex(a_LT[0,0])])
        self.debug["a_LN"].append(result.GetSolution()[
            prog.FindDecisionVariableIndex(a_LN[0,0])])
        self.debug["alpha_MX"].append(result.GetSolution()[
            prog.FindDecisionVariableIndex(alpha_MX[0,0])])
        self.debug["alpha_MY"].append(result.GetSolution()[
            prog.FindDecisionVariableIndex(alpha_MY[0,0])])
        self.debug["alpha_MZ"].append(result.GetSolution()[
            prog.FindDecisionVariableIndex(alpha_MZ[0,0])])
        self.debug["dd_theta_L"].append(result.GetSolution()[
            prog.FindDecisionVariableIndex(dd_theta_L[0,0])])
        self.debug["dd_d_N"].append(result.GetSolution()[
            prog.FindDecisionVariableIndex(dd_d_N[0,0])])
        self.debug["dd_d_T"].append(result.GetSolution()[
            prog.FindDecisionVariableIndex(dd_d_T[0,0])])
        for i in range(self.nq):
            self.debug["ddq_" + str(i)].append(result.GetSolution()[prog.FindDecisionVariableIndex(ddq[i,0])])
        self.debug["theta_MXd"].append(theta_MXd)

        output.SetFromVector(tau_ctrl_result.flatten())
