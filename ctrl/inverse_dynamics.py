# Scipy + numpy
import numpy as np
import scipy.interpolate

# Project files
from constants import SystemConstants
import plant.manipulator
from config import hinge_rotation_axis
import config

# Drake imports
import pydrake
from pydrake.all import MathematicalProgram, eq, Solve,RigidTransform, RotationMatrix, RollPitchYaw

# Other libraries
from collections import defaultdict


class InverseDynamicsController(pydrake.systems.framework.LeafSystem):
    def __init__(self, options, sys_consts: SystemConstants):
        pydrake.systems.framework.LeafSystem.__init__(self)

        self.last_theta_MX = None
        self.last_theta_MY = None
        self.last_theta_MZ = None
        self.last_theta_MXd = None
        self.last_theta_MYd = None
        self.last_theta_MZd = None

        # System constants/parameters
        self.sys_consts = sys_consts
        self.nq = plant.manipulator.data["nq"]

        self.printed_yet = False

        # Copied over stuff from impedance control
        orientation_map = np.load(config.base_path + "orientation_map.npz")
        self.get_theta_Z_func = scipy.interpolate.interp1d(
            orientation_map['theta_Ls'],
            orientation_map['theta_L_EE'],
            fill_value='extrapolate'
        )
        desired_contact_distance = self.sys_consts.w_L/2
        desired_radius = np.sqrt(
            self.sys_consts.r**2 + desired_contact_distance**2)
        self.offset_y = desired_radius - desired_contact_distance

        # Options
        self.model_friction = options['model_friction']
        self.measure_joint_wrench = options['measure_joint_wrench']

        # Target values
        self.d_Td = -self.sys_consts.w_L/2
        self.d_theta_Ld = 0.5
        self.d_Hd = 0

        self.debug = defaultdict(list)

        self.constraint_data = {}
        self.constraint_data_initialized = False

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
        self.DeclareVectorInputPort(
            "gravity_torque_about_joint",
            pydrake.systems.framework.BasicVector(1))

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
            "d_H", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "p_MConM", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "p_JL", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "p_JC", pydrake.systems.framework.BasicVector(3))
        
        # Velocities
        self.DeclareVectorInputPort(
            "d_theta_L", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "omega_MX", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "omega_MY", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "omega_MZ", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "d_d_T", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "d_d_N", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "d_d_H", pydrake.systems.framework.BasicVector(1))

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
            "s_hat_H", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "I_LJ", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "T_hat", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "N_hat", pydrake.systems.framework.BasicVector(3))

        # Impedance control copy stuff
        self.DeclareVectorInputPort(
            "pose_L_rotational", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "pose_L_translational", pydrake.systems.framework.BasicVector(3))

        # ========================== DECLARE OUTPUTS ==========================
        self.DeclareVectorOutputPort(
            "tau_out", pydrake.systems.framework.BasicVector(self.nq),
            self.CalcOutput)
        self.DeclareVectorOutputPort(
            "HTNd", pydrake.systems.framework.BasicVector(3),
            self.calc_HTNd
        )
        self.DeclareVectorOutputPort(
            "rot_XYZd", pydrake.systems.framework.BasicVector(3),
            self.calc_rot_XYZd
        )

    def CalcOutput(self, context, output):
        # ============================ LOAD INPUTS ============================
        # Torque inputs
        tau_g = np.expand_dims(
            np.array(self.GetInputPort("tau_g").Eval(context)), 1)
        joint_centering_torque = np.expand_dims(np.array(self.GetInputPort(
            "joint_centering_torque").Eval(context)), 1)

        # Force inputs
        # PROGRAMMING: Add zeros here?
        F_GT = self.GetInputPort("F_GT").Eval(context)[0]
        F_GN = self.GetInputPort("F_GN").Eval(context)[0]
        I_LJ = self.GetInputPort("I_LJ").Eval(context)[0]
        gravity_torque_about_joint = np.expand_dims(np.array(self.GetInputPort(
            "gravity_torque_about_joint").Eval(context)), 1)

        # Positions
        theta_L = self.GetInputPort("theta_L").Eval(context)[0]
        theta_MX = self.GetInputPort("theta_MX").Eval(context)[0]
        theta_MY = self.GetInputPort("theta_MY").Eval(context)[0]
        theta_MZ = self.GetInputPort("theta_MZ").Eval(context)[0]
        if theta_MZ > np.pi/2:
            theta_MY = theta_MY*-1 + np.pi

        p_CT = self.GetInputPort("p_CT").Eval(context)[0]
        p_CN = self.GetInputPort("p_CN").Eval(context)[0]
        p_LT = self.GetInputPort("p_LT").Eval(context)[0]
        p_LN = self.GetInputPort("p_LN").Eval(context)[0]
        d_T = self.GetInputPort("d_T").Eval(context)[0]
        d_N = self.GetInputPort("d_N").Eval(context)[0]
        d_H = self.GetInputPort("d_H").Eval(context)[0]
        p_MConM = np.array([self.GetInputPort("p_MConM").Eval(context)]).T
        p_JL = np.array([self.GetInputPort("p_JL").Eval(context)]).T
        p_JC = np.array([self.GetInputPort("p_JC").Eval(context)]).T

        # Velocities
        d_theta_L = self.GetInputPort("d_theta_L").Eval(context)[0]
        omega_MX = self.GetInputPort("omega_MX").Eval(context)[0]
        omega_MY = self.GetInputPort("omega_MY").Eval(context)[0]
        omega_MZ = self.GetInputPort("omega_MZ").Eval(context)[0]
        d_d_T = self.GetInputPort("d_d_T").Eval(context)[0]
        d_d_N = self.GetInputPort("d_d_N").Eval(context)[0]
        d_d_H = self.GetInputPort("d_d_H").Eval(context)[0]

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
        mu_S = self.GetInputPort("mu_S").Eval(context)[0]
        hats_T = self.GetInputPort("hats_T").Eval(context)[0]
        s_hat_H = self.GetInputPort("s_hat_H").Eval(context)[0]

        T_hat = np.array([self.GetInputPort("T_hat").Eval(context)]).T
        N_hat = np.array([self.GetInputPort("N_hat").Eval(context)]).T
        H_hat = np.array([[0,0,0]]).T
        H_hat[hinge_rotation_axis] = 1

        # ======================== IMPEDANCE CTRL COPY ========================
        # Evaluate inputs
        pose_L_rotational = self.GetInputPort(
            "pose_L_rotational").Eval(context)
        pose_L_translational = self.GetInputPort(
            "pose_L_translational").Eval(context)

        # Calc X_L_SP, the transform from the link to the setpoint
        # TODO: should this be different?
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

        rotation = RollPitchYaw(X_W_SP.rotation()).vector()
        rotation[0] += plant.manipulator.RotX_L_Md

        # ========================= CALC DESIRED VALS =========================
        # Load in the values that we got from our link offset
        # TODO: move this to its own thing?
        theta_MXd = rotation[0]
        theta_MYd = rotation[1]
        theta_MZd = rotation[2]

        # Calculated desired roll, pitch, and yaw rates.
        # Overall angular velocity math here comes from equation 3.35 on p. 76
        # of MR and this link:
        # http://personal.maths.surrey.ac.uk/T.Bridges/SLOSH/3-2-1-Eulerangles.pdf
        # (Specifically, bottom of page 6.)
        d_theta_MXd = 10*(theta_MXd - theta_MX)
        d_theta_MYd = 10*(theta_MYd - theta_MY)
        d_theta_MZd = 10*(theta_MZd - theta_MZ)
        d_Theta_d = np.array([[d_theta_MXd, d_theta_MYd, d_theta_MZd]]).T
        R = RollPitchYaw(
            theta_MX, theta_MY, theta_MZ).ToRotationMatrix().matrix()
        B_inv = np.array([
            [1,  0,              -np.sin(theta_MY)],
            [0,  np.cos(theta_MX), np.cos(theta_MY)*np.sin(theta_MX)],
            [0, -np.sin(theta_MX), np.cos(theta_MY)*np.cos(theta_MX)],
        ])
        # Calculate desired angular velocity based on desired roll, pitch, and
        # yaw rates. We multiply by B_inv to convert to angular velocity in the
        # body frame, and multiply by R to move it from the body frame to the
        # world frame.
        omega_MXd, omega_MYd, omega_MZd = np.matmul(
            R, np.matmul(B_inv, d_Theta_d)).flatten()

        # Calculate desired accelerations.
        Kp_dd_d_Td = 1000
        dd_d_Td = Kp_dd_d_Td*(self.d_Td - d_T) - 2*np.sqrt(Kp_dd_d_Td)*d_d_T
        dd_theta_Ld = 10*(self.d_theta_Ld - d_theta_L)
        a_MH_d = 1000*(self.d_Hd - d_H) - 2*np.sqrt(1000)*d_d_H
        alpha_MXd = 10*(omega_MXd - omega_MX)
        alpha_MYd = 10*(omega_MYd - omega_MY)
        alpha_MZd = 10*(omega_MZd -omega_MZ)
        dd_d_Nd = 0

        # =========================== SOLVE PROGRAM ===========================
        ## 1. Define an instance of MathematicalProgram
        prog = MathematicalProgram()

        ## 2. Add decision variables
        # Contact values
        F_NM = prog.NewContinuousVariables(1, 1, name="F_NM")
        F_ContactMX = prog.NewContinuousVariables(1, 1, name="F_ContactMX")
        F_ContactMY = prog.NewContinuousVariables(1, 1, name="F_ContactMY")
        F_ContactMZ = prog.NewContinuousVariables(1, 1, name="F_ContactMZ")
        F_NL = prog.NewContinuousVariables(1, 1, name="F_NL")
        if self.model_friction:
            F_FMT = prog.NewContinuousVariables(1, 1, name="F_FMT")
            F_FMH = prog.NewContinuousVariables(1, 1, name="F_FMH")
            F_FLT = prog.NewContinuousVariables(1, 1, name="F_FLT")
            F_FLH = prog.NewContinuousVariables(1, 1, name="F_FLH")
        else:
            F_FMT = np.array([[0]])
            F_FMH = np.array([[0]])
            F_FLT = np.array([[0]])
            F_FLH = np.array([[0]])
        F_ContactM_XYZ = np.array([F_ContactMX, F_ContactMY, F_ContactMZ])[:,:,0]
        F_ContactL_XYZ = -F_ContactM_XYZ

        # Object forces and torques
        if not self.measure_joint_wrench:
            F_OT = prog.NewContinuousVariables(1, 1, name="F_OT")
            F_ON = prog.NewContinuousVariables(1, 1, name="F_ON")
            tau_O = -self.sys_consts.k_J*theta_L \
                    - self.sys_consts.b_J*d_theta_L
        # TODO: I don't currently have an option for not doing this

        # Control values
        tau_ctrl = prog.NewContinuousVariables(
            self.nq, 1, name="tau_ctrl")

        # Object accelerations
        a_MH = prog.NewContinuousVariables(1, 1, name="a_MH")
        a_MT = prog.NewContinuousVariables(1, 1, name="a_MT")
        a_MN = prog.NewContinuousVariables(1, 1, name="a_MN")

        a_MX = prog.NewContinuousVariables(1, 1, name="a_MX")
        a_MY = prog.NewContinuousVariables(1, 1, name="a_MY")
        a_MZ = prog.NewContinuousVariables(1, 1, name="a_MZ")

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

        # Dependent terms I'll need later
        contact_torque_about_joint = np.cross(
            p_JC, F_ContactL_XYZ, axis=0).flatten()[hinge_rotation_axis]

        ## Constraints
        # "set_description" calls gives us useful names for printing
        prog.AddConstraint(eq(
            self.sys_consts.m_L*a_LT, F_GT+F_OT+F_FLT
        )).evaluator().set_description("Link tangential force balance")
        prog.AddConstraint(eq(
            self.sys_consts.m_L*a_LN, F_NL + F_GN + F_ON
        )).evaluator().set_description("Link normal force balance") 
        prog.AddConstraint(eq(
            I_LJ*dd_theta_L, \
            contact_torque_about_joint + gravity_torque_about_joint + tau_O
        )).evaluator().set_description("Link moment balance") 
        prog.AddConstraint(eq(
            F_NL, -F_NM
        )).evaluator().set_description("3rd law normal forces")
        if self.model_friction:
            prog.AddConstraint(eq(F_FMT, -F_FLT)).evaluator().set_description(
                    "3rd law friction forces (T hat)") 
            prog.AddConstraint(eq(F_FMH, -F_FLH)).evaluator().set_description(
                    "3rd law friction forces (H hat)") 
        prog.AddConstraint(eq(
            -dd_theta_L*(self.sys_consts.h_L/2+self.sys_consts.r) + \
                d_theta_L**2*self.sys_consts.w_L/2 - a_LT + a_MT,
            -dd_theta_L*d_N + dd_d_T - d_theta_L**2*d_T - 2*d_theta_L*d_d_N
        )).evaluator().set_description("d_T derivative is derivative")
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
                F_FLH, mu_S*F_NL*self.sys_consts.mu*s_hat_H
            )).evaluator().set_description("Friction relationship LH")
        
        if not self.measure_joint_wrench:
            p_JL_norm = np.linalg.norm(p_JL)
            prog.AddConstraint(eq(
                a_LT, -p_JL_norm*d_theta_L**2
            )).evaluator().set_description("Hinge constraint (T hat)")
            prog.AddConstraint(eq(
                a_LN, p_JL_norm*dd_theta_L
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
        a_HTN_to_XYZ = (T_hat*a_MT + N_hat*a_MN + a_MH*H_hat).flatten()
        prog.AddConstraint(eq(a_MX, a_HTN_to_XYZ[0]
            )).evaluator().set_description("a HTN to XYZ converstion 0")
        prog.AddConstraint(eq(a_MY, a_HTN_to_XYZ[1]
            )).evaluator().set_description("a HTN to XYZ converstion 1")
        prog.AddConstraint(eq(a_MZ, a_HTN_to_XYZ[2]
            )).evaluator().set_description("a HTN to XYZ converstion 2")
        F_HTN_to_XYZ = (T_hat*F_FMT + N_hat*F_NM + H_hat*F_FMH).flatten()
        prog.AddConstraint(eq(F_ContactMX, F_HTN_to_XYZ[0]
            )).evaluator().set_description("F_C HTN to XYZ converstion 0")
        prog.AddConstraint(eq(F_ContactMY, F_HTN_to_XYZ[1]
            )).evaluator().set_description("F_C HTN to XYZ converstion 1")
        prog.AddConstraint(eq(F_ContactMZ, F_HTN_to_XYZ[2]
            )).evaluator().set_description("F_C HTN to XYZ converstion 2")

        prog.AddConstraint(
            dd_d_T[0,0] == dd_d_Td
        ).evaluator().set_description("Desired dd_d_Td constraint")
        prog.AddConstraint(
            dd_theta_L[0,0] == dd_theta_Ld
        ).evaluator().set_description("Desired a_LN constraint")
        prog.AddConstraint(
            a_MH[0,0] == a_MH_d
        ).evaluator().set_description("Desired a_MH constraint")
        prog.AddConstraint(
            alpha_MX[0,0] == alpha_MXd
        ).evaluator().set_description("Desired alpha_MX constraint")
        prog.AddConstraint(
            alpha_MY[0,0] == alpha_MYd
        ).evaluator().set_description("Desired alpha_MY constraint")
        prog.AddConstraint(
            alpha_MZ[0,0] == alpha_MZd
        ).evaluator().set_description("Desired alpha_MZ constraint")
        prog.AddConstraint(
            dd_d_N[0,0] == dd_d_Nd
        ).evaluator().set_description("Desired dd_d_N constraint")


        # Initialize constraint data if it doesn't exist
        if not self.constraint_data_initialized:
            self.constraint_data['t'] = []
            for c in prog.GetAllConstraints():
                c_name = c.evaluator().get_description()
                print(c_name)
                # Watch out for accidental duplicated names
                assert c_name not in self.constraint_data.keys()

                constraint_data_ = {}
                constraint_data_['A'] = []
                constraint_data_['b'] = []
                var_names_ = []
                for v in c.variables():
                    var_names_.append(v.get_name())
                constraint_data_['var_names'] = var_names_

                self.constraint_data[c_name] = constraint_data_
        self.constraint_data_initialized = True

        # Populate constraint data
        self.constraint_data['t'].append(context.get_time())
        for c in prog.GetAllConstraints():
            c_name = c.evaluator().get_description()
            self.constraint_data[c_name]['A'].append(c.evaluator().A())
            assert c.evaluator().lower_bound() == c.evaluator().upper_bound()
            self.constraint_data[c_name]['b'].append(
                c.evaluator().lower_bound())

        result = Solve(prog)
        if result.is_success():
            if not self.printed_yet:

                print("==============================")
                print("ALL CONSTRAINTS")
                print("------------------------------")
                for c in prog.GetAllConstraints():
                    print(c.evaluator().get_description())
                    print(c)
                    print(c.evaluator().upper_bound())
                    print(c.evaluator().lower_bound())
                    print(c.evaluator().A())
                    print("evaluator:", c.evaluator())
                    print("variables:", c.variables())
                    print()
                print("==============================")
                print("CONSTRAINTS BY VARIABLE")
                for var in prog.decision_variables():
                    print("------------------------------")
                    print(var.get_name())
                    v_num_c = 0
                    print(" ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")
                    for c in prog.GetAllConstraints():
                        for v_ in c.variables():
                            if var.get_name() == v_.get_name():
                                print(c.evaluator().get_description())
                                v_num_c += 1
                    print(" ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")
                    print("Total constraints for {}: {}".format(
                        var.get_name(), v_num_c
                    ))
            
                print("Num constraints:", len(prog.GetAllConstraints()))
                print("Total unknowns:", len(prog.decision_variables()))
            tau_ctrl_result = []
            for i in range(self.nq):
                tau_ctrl_result.append(result.GetSolution()[
                    prog.FindDecisionVariableIndex(tau_ctrl[i,0])])
            tau_ctrl_result = np.expand_dims(tau_ctrl_result, 1)
            self.debug["is_success"].append(True)
        else:
            tau_ctrl_result = np.zeros((self.nq, 1))
            self.debug["is_success"].append(False)

        # ======================== UPDATE DEBUG VALUES ========================
        self.debug["times"].append(context.get_time())

        # control effort
        self.debug["dd_d_Td"].append(dd_d_Td)
        self.debug["dd_theta_Ld"].append(dd_theta_Ld)
        self.debug["a_MH_d"].append(a_MH_d)
        self.debug["alpha_MXd"].append(alpha_MXd)
        self.debug["alpha_MYd"].append(alpha_MYd)
        self.debug["alpha_MZd"].append(alpha_MZd)
        self.debug["dd_d_Nd"].append(dd_d_Nd)

        # decision variables
        if self.model_friction:
            self.debug["F_FMT"].append(result.GetSolution()[
                prog.FindDecisionVariableIndex(F_FMT[0,0])])
            self.debug["F_FMH"].append(result.GetSolution()[
                prog.FindDecisionVariableIndex(F_FMH[0,0])])
            self.debug["F_FLT"].append(result.GetSolution()[
                prog.FindDecisionVariableIndex(F_FLT[0,0])])
            self.debug["F_FLH"].append(result.GetSolution()[
                prog.FindDecisionVariableIndex(F_FLH[0,0])])
        else:
            self.debug["F_FMT"].append(F_FMT)
            self.debug["F_FMH"].append(F_FMH)
            self.debug["F_FLT"].append(F_FLT)
            self.debug["F_FLH"].append(F_FLH)

        F_ContactL_XYZ_out = np.array([[
            -result.GetSolution()[prog.FindDecisionVariableIndex(F_ContactMX[0,0])],
            -result.GetSolution()[prog.FindDecisionVariableIndex(F_ContactMY[0,0])],
            -result.GetSolution()[prog.FindDecisionVariableIndex(F_ContactMZ[0,0])],
        ]]).T

        F_ContactM_XYZ_out = np.array([[
            result.GetSolution()[prog.FindDecisionVariableIndex(F_ContactMX[0,0])],
            result.GetSolution()[prog.FindDecisionVariableIndex(F_ContactMY[0,0])],
            result.GetSolution()[prog.FindDecisionVariableIndex(F_ContactMZ[0,0])],
        ]]).T
        tau_contact_trn_out = np.matmul(
            J_translational.T, F_ContactM_XYZ_out)
        tau_contact_rot_out = np.matmul(
            J_rotational.T, np.cross(p_MConM, F_ContactM_XYZ_out, axis=0))
        tau_contact_out = tau_contact_trn_out + tau_contact_rot_out
        self.debug["tau_contact"].append(tau_contact_out)
        self.debug["tau_g"].append(tau_g)
        self.debug["joint_centering_torque"].append(joint_centering_torque)
        
        contact_torque_about_joint_out = np.cross(p_JC,
            F_ContactL_XYZ_out,
            axis=0).flatten()[hinge_rotation_axis]
        self.debug["gravity_torque_about_joint"].append(gravity_torque_about_joint)
        self.debug["contact_torque_about_joint_out"].append(contact_torque_about_joint_out)
        self.debug["F_OT"].append(result.GetSolution()[
            prog.FindDecisionVariableIndex(F_OT[0,0])])
        self.debug["F_ON"].append(result.GetSolution()[
            prog.FindDecisionVariableIndex(F_ON[0,0])])
        self.debug["mu_S"].append(mu_S)
        self.debug["mu"].append(self.sys_consts.mu)
        self.debug["M"].append(M)
        self.debug["Cv"].append(Cv)
        self.debug["s_hat_T"].append(hats_T)

        self.debug["a_MH"].append(result.GetSolution()[
            prog.FindDecisionVariableIndex(a_MH[0,0])])
        self.debug["F_NM"].append(result.GetSolution()[
            prog.FindDecisionVariableIndex(F_NM[0,0])])
        self.debug["F_ContactMX"].append(result.GetSolution()[
            prog.FindDecisionVariableIndex(F_ContactMX[0,0])])
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
        self.debug["d_theta_MXd"].append(d_theta_MXd)
        self.debug["d_theta_MYd"].append(d_theta_MYd)
        self.debug["d_theta_MZd"].append(d_theta_MZd)
        self.debug["omega_MXd"].append(omega_MXd)
        self.debug["omega_MYd"].append(omega_MYd)
        self.debug["omega_MZd"].append(omega_MZd)
        self.debug["theta_MXd"].append(theta_MXd)
        self.debug["theta_MYd"].append(theta_MYd)
        self.debug["theta_MZd"].append(theta_MZd)


        self.printed_yet = True
        output.SetFromVector(tau_ctrl_result.flatten())

    def calc_HTNd(self, context, output):
        out_vec = [
            self.d_Hd,
            self.d_Td+self.sys_consts.w_L/2,
            0
        ]
        output.SetFromVector(out_vec)
    
    def calc_rot_XYZd(self, context, output):
        theta_L = self.GetInputPort("theta_L").Eval(context)[0]
        theta_MXd = theta_L
        out_vec = [0,0,0]
        output.SetFromVector(out_vec)
