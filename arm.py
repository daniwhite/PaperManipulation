"""Functions for creating and controlling the finger manipulator."""
# General imports
import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify
import re
from collections import defaultdict

# Local imports
import constants
import finger
import paper
import pedestal
from common import get_contact_point_from_results

# Drake imports
import pydrake
from pydrake.multibody.tree import SpatialInertia, UnitInertia, JacobianWrtVariable
from pydrake.all import BasicVector, MultibodyPlant, ContactResults, SpatialVelocity, SpatialForce, FindResourceOrThrow, RigidTransform, RotationMatrix

FINGER_NAME = "panda_leftfinger"


def AddArm(plant, scene_graph=None):
    """
    Creates the panda arm.
    """
    parser = pydrake.multibody.parsing.Parser(plant, scene_graph)
    arm_instance = parser.AddModelFromFile("panda_arm_hand.urdf")

    # Weld pedestal to world
    jnt = plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("panda_link0", arm_instance),
        RigidTransform(RotationMatrix(), [0, 0.85, 0])
    )
    # Weld fingers (offset matches original urdf)
    plant.WeldFrames(
        plant.GetFrameByName("panda_hand", arm_instance),
        plant.GetFrameByName("panda_leftfinger", arm_instance),
        RigidTransform(RotationMatrix(), [0, 0, 0.0584])
    )
    plant.WeldFrames(
        plant.GetFrameByName("panda_hand", arm_instance),
        plant.GetFrameByName("panda_rightfinger", arm_instance),
        RigidTransform(RotationMatrix(), [0, 0, 0.0584])
    )

    return arm_instance


class ArmFoldingController(pydrake.systems.framework.LeafSystem):
    """Base class for implementing a controller at the finger."""

    def __init__(self, arm_acc_log, ll_idx, finger_idx, options, sys_params, jnt_frc_log):
        pydrake.systems.framework.LeafSystem.__init__(self)

        # Initialize system parameters
        self.v_stiction = sys_params['v_stiction']
        self.I_M = sys_params['I_M']
        self.I_L = sys_params['I_L']
        self.w_L = sys_params['w_L']
        self.m_L = sys_params['m_L']
        self.b_J = sys_params['b_J']
        self.k_J = sys_params['k_J']
        self.g = sys_params['g']

        # Initialize control targets
        self.d_Td = -0.03
        self.d_theta_Ld = 2*np.pi / 5  # 1 rotation per 5 secs
        self.a_LNd = 0.1
        self.d_Xd = 0
        self.v_LNd = 0

        # Initialize estimates
        self.mu_hat = 0.8

        # Initialize gains
        self.k = 10
        self.K_centering = 1
        self.D_centering = 0.1
        self.lamda = 100 # Sliding surface time constant
        self.P = 10000 # Adapatation law gain

        # Initialize logs
        self.arm_acc_log = arm_acc_log
        self.debug = defaultdict(list)
        self.debug['times'] = []
        self.jnt_frc_log = jnt_frc_log
        self.jnt_frc_log.append(SpatialForce(np.zeros((3, 1)), np.zeros((3, 1))))
        self.d_d_N_sqr_log = []

        # Initialize intermediary variables
        self.last_v_LN = 0
        self.init_q = None
        self.t_contact_start =  None

        # Other parameters
        self.ll_idx = ll_idx
        self.finger_idx = finger_idx
        self.use_friction_adaptive_ctrl = options['use_friction_adaptive_ctrl']
        self.use_friction_robust_adaptive_ctrl = options['use_friction_robust_adaptive_ctrl']
        self.d_d_N_sqr_log_len = 100
        self.d_d_N_sqr_lim = 2e-4

        self.arm_plant = MultibodyPlant(constants.DT)
        AddArm(self.arm_plant)
        self.arm_plant.Finalize()
        self.arm_plant_context = self.arm_plant.CreateDefaultContext()

        self.nq_arm = self.arm_plant.get_actuation_input_port().size()

        # Input ports
        self.DeclareVectorInputPort("q", BasicVector(self.nq_arm*2))
        self.DeclareAbstractInputPort(
            "poses", pydrake.common.value.AbstractValue.Make([RigidTransform(), RigidTransform()]))
        self.DeclareAbstractInputPort(
            "vels", pydrake.common.value.AbstractValue.Make([SpatialVelocity(), SpatialVelocity()]))
        self.DeclareAbstractInputPort(
            "contact_results",
            pydrake.common.value.AbstractValue.Make(ContactResults()))

        # Output ports
        self.DeclareVectorOutputPort(
            "arm_actuation", pydrake.systems.framework.BasicVector(
                self.nq_arm),
            self.CalcOutput)

        self.init_math()

    def CalcOutput(self, context, output):
        ## Load inputs
        # This input put is already restricted to the arm, but it includes both q and v
        state = self.get_input_port(0).Eval(context)
        poses = self.get_input_port(1).Eval(context)
        vels = self.get_input_port(2).Eval(context)
        contact_results = self.get_input_port(3).Eval(context)
 
        # Get time
        self.debug['times'].append(context.get_time())

        ## Process contact
        # Get contact point
        contact_point = None
        slip_speed = None
        pen_depth = None
        N_hat = None
        for i in range(contact_results.num_point_pair_contacts()):
            point_pair_contact_info = \
                contact_results.point_pair_contact_info(i)

            a_idx = int(point_pair_contact_info.bodyA_index())
            b_idx = int(point_pair_contact_info.bodyB_index())

            if ((a_idx == self.ll_idx) and (b_idx == self.finger_idx) or
                    (a_idx == self.finger_idx) and (b_idx == self.ll_idx)):
                contact_point = point_pair_contact_info.contact_point()
                slip_speed = point_pair_contact_info.slip_speed()
                pen_point_pair = point_pair_contact_info.point_pair()
                pen_depth = pen_point_pair.depth
                # PROGRAMMING: check sign
                N_hat = np.expand_dims(pen_point_pair.nhat_BA_W, 1)
        
        # Get contact times
        raw_in_contact = not (contact_point is None)
        if raw_in_contact:
            if self.t_contact_start is None:
                self.t_contact_start = self.debug['times'][-1]
        else:
            self.t_contact_start =  None
        in_contact = raw_in_contact and self.debug['times'][-1] - self.t_contact_start > 0.002

        # Process state
        q = state[:self.nq_arm]
        v = state[self.nq_arm:]
        self.arm_plant.SetPositions(self.arm_plant_context, q)
        self.arm_plant.SetVelocities(self.arm_plant_context, v)
        real_q_dot = self.arm_plant.MapVelocityToQDot(self.arm_plant_context, v)

        if self.init_q is None:
            self.init_q = q

        # Get gravity 
        grav = self.arm_plant.CalcGravityGeneralizedForces(
            self.arm_plant_context)

        # Get desired forces
        M = self.arm_plant.CalcMassMatrixViaInverseDynamics(self.arm_plant_context)
        Cv = self.arm_plant.CalcBiasTerm(self.arm_plant_context)
        self.debug["M"].append(M)
        self.debug["C"].append(Cv)
        F_d = self.get_control_forces(poses, vels, contact_point, slip_speed, pen_depth, N_hat, q, v, M, Cv)
        if not in_contact:
            F_d = np.array([[0, 0, 0.1]]).T

        # Convert forces to joint torques
        finger_body = self.arm_plant.GetBodyByName(FINGER_NAME)
        J_full = self.arm_plant.CalcJacobianSpatialVelocity(
            self.arm_plant_context,
            JacobianWrtVariable.kQDot,
            finger_body.body_frame(),
            [0, 0, 0],
            self.arm_plant.world_frame(),
            self.arm_plant.world_frame())
        J = J_full[3:,:]

        J_plus = np.linalg.pinv(J)
        nullspace_basis = np.eye(self.nq_arm) - np.matmul(J_plus, J)

        # Add a PD controller projected into the nullspace of the Jacobian that keeps us close to the nominal configuration
        joint_centering_torque = np.matmul(nullspace_basis, self.K_centering*(self.init_q - q) + self.D_centering*(-v))

        tau_d = (J.T@F_d).flatten()

        tau_ctrl = tau_d - grav + joint_centering_torque
        output.SetFromVector(tau_ctrl)

        # Debug
        self.debug["tau_g"].append(grav)

        self.debug['F_d'].append(F_d)
        self.debug['tau_d'].append(tau_d)
        self.debug['tau_ctrl'].append(tau_ctrl)
        self.debug['J'].append(J_full)
        self.debug['real_q_dot'].append(real_q_dot)
        
        self.debug['raw_in_contact'].append(raw_in_contact)
        self.debug['in_contact'].append(in_contact)


    def get_control_forces(self, poses, vels, contact_point, slip_speed, pen_depth, N_hat, q, v, M, Cv):
        inputs = {}
        jnt_frcs = self.jnt_frc_log[-1]
        inputs['F_OT'] = F_OT = jnt_frcs.translational()[1]
        inputs['F_ON'] = F_ON = jnt_frcs.translational()[2]
        inputs['tau_O'] = tau_O = jnt_frcs.rotational()[0]

        # Directions
        R = poses[self.ll_idx].rotation()
        y_hat = np.array([[0, 1, 0]]).T
        z_hat = np.array([[0, 0, 1]]).T
        if N_hat is None:
            T_hat = R@y_hat
            N_hat = R@z_hat
        else:
            T_hat = np.matmul(
                np.array([
                    [0,  0, 0],
                    [0,  0, 1],
                    [0, -1, 0],
                ]),
                N_hat)

        T_proj_mat = T_hat@(T_hat.T)
        N_proj_mat = N_hat@(N_hat.T)

        # Helper functions
        def get_T_proj(vec):
            T_vec = np.matmul(T_proj_mat, vec)
            T_mag = np.linalg.norm(T_vec)
            T_sgn = np.sign(np.matmul(T_hat.T, T_vec))
            T = T_mag.flatten()*T_sgn.flatten()
            return T[0]

        def get_N_proj(vec):
            N_vec = np.matmul(N_proj_mat, vec)
            N_mag = np.linalg.norm(N_vec)
            N_sgn = np.sign(np.matmul(N_hat.T, N_vec))
            N = N_mag.flatten()*N_sgn.flatten()
            return N[0]

        def step5(x):
            '''Python version of MultibodyPlant::StribeckModel::step5 method'''
            x3 = x * x * x
            return x3 * (10 + x * (6 * x - 15))

        def stribeck(us, uk, v):
            '''
            Python version of MultibodyPlant::StribeckModel::ComputeFrictionCoefficient

            From
            https://github.com/RobotLocomotion/drake/blob/b09e40db4b1c01232b22f7705fb98aa99ef91f87/multibody/plant/images/stiction.py
            '''
            u = uk
            if v < 1:
                u = us * step5(v)
            elif (v >= 1) and (v < 3):
                u = us - (us - uk) * step5((v - 1) / 2)
            return u

        # Constants
        inputs['w_L'] = w_L = self.w_L
        inputs['h_L'] = h_L = paper.PAPER_HEIGHT
        inputs['m_M'] = m_M = finger.MASS # TODO get rid of
        inputs['m_L'] = self.m_L
        inputs['I_L'] = self.I_L
        inputs['I_M'] = self.I_M
        inputs['b_J'] = self.b_J
        inputs['k_J'] = self.k_J

        # Positions
        p_L = np.array([poses[self.ll_idx].translation()[0:3]]).T
        inputs['p_LT'] = get_T_proj(p_L)
        inputs['p_LN'] = p_LN = get_N_proj(p_L)

        p_M = np.array([poses[self.finger_idx].translation()[0:3]]).T
        inputs['p_MN'] = get_N_proj(p_M)

        angle_axis = poses[self.ll_idx].rotation().ToAngleAxis()
        theta_L = angle_axis.angle()
        if sum(angle_axis.axis()) < 0:
            theta_L *= -1
        inputs['theta_L'] = theta_L

        p_LLE = N_hat * -h_L/2 + T_hat * w_L/2
        p_LE = p_L + p_LLE

        p_LEM = p_M - p_LE
        p_LEMT = get_T_proj(p_LEM)

        # Velocities
        inputs['d_theta_L'] = d_theta_L = vels[self.ll_idx].rotational()[0]
        inputs['d_theta_M'] = d_theta_M = vels[self.finger_idx].rotational()[
            0]
        omega_vec_L = np.array([[d_theta_L, 0, 0]]).T
        omega_vec_M = np.array([[d_theta_M, 0, 0]]).T

        v_L = np.array([vels[self.ll_idx].translational()[0:3]]).T
        inputs['v_LN'] = v_LN = get_N_proj(v_L)
        inputs['v_LT'] = v_LT = get_T_proj(v_L)

        v_M = np.array([vels[self.finger_idx].translational()[0:3]]).T
        inputs['v_MN'] = v_MN = get_N_proj(v_M)
        inputs['v_MT'] = v_MT = get_T_proj(v_M)

        # Assume for now that link edge is not moving
        # PROGRAMMING: Get rid of this assumption
        v_LEM = v_M
        v_LEMT = get_T_proj(p_LEM)

        if contact_point is None:
            F_CN = 0.1
            F_CT = self.get_pre_contact_F_CT(p_LEMT, v_LEMT)
            tau_M = 0

            d_d_N_sqr_sum = np.nan

            d_X = p_M[0]
            d_d_X = v_M[0]
            F_FMX = 0
        else:
            pen_vec = pen_depth*N_hat

            # Positions
            p_C = np.array([contact_point]).T
            inputs['p_CT'] = get_T_proj(p_C)
            inputs['p_CN'] = get_N_proj(p_C)
            r = np.linalg.norm(p_C-p_M) # TODO: make this support multiple directions

            inputs['r'] = r 

            d = p_C - p_LE + pen_vec/2
            inputs['d_T'] = d_T = get_T_proj(d)
            inputs['d_N'] = d_N = get_N_proj(d)
            d_X = d[0]

            p_MConM = p_C - p_M
            p_LConL = p_C - p_L

            # Velocities
            v_WConL = v_L + np.cross(omega_vec_L, p_LConL, axis=0)
            v_WConM = v_M + np.cross(omega_vec_M, p_MConM, axis=0)

            inputs['d_d_T'] = d_d_T = -d_theta_L*h_L/2 - \
                d_theta_L*r - v_LT + v_MT + d_theta_L*d_N
            inputs['d_d_N'] = d_d_N = -d_theta_L*w_L/2-v_LN+v_MN-d_theta_L*d_T
            d_d_X = v_WConM[0]

            v_S_raw = v_WConM - v_WConL
            v_S_N = np.matmul(N_proj_mat, v_S_raw)
            v_S = v_S_raw - v_S_N

            s_hat = v_S/np.linalg.norm(v_S)
            inputs['hats_T'] = get_T_proj(s_hat)
            s_hat_X = s_hat[0]

            # Targets
            inputs['a_LNd'] = a_LNd = self.a_LNd

            # Forces
            if self.use_friction_adaptive_ctrl:
                mu = self.mu_hat
            else:
                mu_paper = constants.FRICTION
                mu = 2*mu_paper/(1+mu_paper)
            inputs['mu'] = mu
            stribeck_mu = stribeck(1, 1, slip_speed/self.v_stiction)
            inputs['mu_S'] = stribeck_mu

            # Gravity
            F_G = np.array([[0, 0, -self.m_L*constants.g]]).T
            inputs['F_GT'] = get_T_proj(F_G)
            inputs['F_GN'] = get_N_proj(F_G)

            # Pack inputs according to order from algebra
            inps_ = []
            for inp in self.alg_inputs:
                var_name = inp.name
                var_str = self.latex_to_str(var_name)
                inps_.append(inputs[var_str])

            F_NL = self.get_F_NL(inps_)

            F_FMX = stribeck_mu*F_NL*mu*s_hat_X

            # Calculate metric used to tell whether or not contact transients have passed
            if len(self.d_d_N_sqr_log) < self.d_d_N_sqr_log_len:
                self.d_d_N_sqr_log.append(d_d_N**2)
            else:
                self.d_d_N_sqr_log = self.d_d_N_sqr_log[1:] + [d_d_N**2]
            d_d_N_sqr_sum = sum(self.d_d_N_sqr_log)

            def sat(phi):
                if phi > 0:
                    return min(phi, 1)
                return max(phi, -1)

            f_mu = self.get_f_mu(inps_)
            f = self.get_f(inps_)
            g_mu = self.get_g_mu(inps_)
            g = self.get_g(inps_)

            s_d_T = self.lamda*(d_T - self.d_Td) + (d_d_T)
            s_F = v_LN - self.v_LNd
            phi = 0.001
            s_delta_d_T = s_d_T - phi*sat(s_d_T/phi)
            Y = g_mu - f_mu*a_LNd
            dt = 0
            if len(self.d_d_N_sqr_log) >= self.d_d_N_sqr_log_len and d_d_N_sqr_sum < self.d_d_N_sqr_lim: # Check if d_N is oscillating
                if len(self.debug['times']) >= 2:
                    dt = self.debug['times'][-1] - self.debug['times'][-2]
                    self.mu_hat += -self.P*dt*Y*s_delta_d_T
                if self.mu_hat > 1:
                    self.mu_hat = 1
                if self.mu_hat < 0:
                    self.mu_hat = 0

            self.v_LNd += a_LNd * dt

            # Even if we're not using adaptive control (i.e. learning mu), we'll still the lambda term to implement sliding mode control
            u_hat = -(f*a_LNd) + g - m_M * self.lamda*d_d_T
            
            if self.use_friction_adaptive_ctrl:
                Y_mu_term = Y*self.mu_hat
            else:
                Y_mu_term = Y*constants.FRICTION

            F_CT = u_hat + Y_mu_term
            if self.use_friction_robust_adaptive_ctrl:
                # TODO: remove this option
                F_CT += -self.k*s_delta_d_T# - k_robust*np.sign(s_delta_d_T)
            else:
                F_CT += -self.k*s_d_T
            F_CN = self.get_F_CN(inps_) - self.k*s_F

        s_d_X = self.lamda*(d_X - self.d_Xd) + (d_d_X)
        F_CX = -self.k*s_d_X - F_FMX

        F_M = F_CN*N_hat + F_CT*T_hat
        F_M[0] = F_CX

        if self.debug is not None:
            self.debug['F_CNs'].append(F_CN)
            self.debug['F_CTs'].append(F_CT)
            self.debug['F_OTs'].append(F_OT)
            self.debug['F_ONs'].append(F_ON)
            self.debug['tau_Os'].append(tau_O)
            self.debug['mu_ests'].append(self.mu_hat)
            self.debug['d_d_N_sqr_sum'].append(d_d_N_sqr_sum)
            self.debug['v_LNds'].append(self.v_LNd)

        return F_M.flatten()


    def get_pre_contact_F_CT(self, p_LEMT, v_LEMT):
        Kp = 0.1
        Kd = 1
        return Kp*(self.d_Td - p_LEMT) - Kd*v_LEMT


    def init_math(self):
        self.alg_inputs = []

        # PROGRAMMING: Do this with Drake instead?
        # PROGRAMMING: Can I get some of these lists from sympy?
        # Physical and geometric quantities
        m_L = sp.symbols(r"m_L")
        self.alg_inputs.append(m_L)
        m_M = sp.symbols(r"m_M")
        self.alg_inputs.append(m_M)
        w_L = sp.symbols(r"w_L")
        self.alg_inputs.append(w_L)
        I_L = sp.symbols(r"I_L")
        self.alg_inputs.append(I_L)
        h_L = sp.symbols(r"h_L")
        self.alg_inputs.append(h_L)
        r = sp.symbols(r"r")
        self.alg_inputs.append(r)
        I_M = sp.symbols(r"I_M")
        self.alg_inputs.append(I_M)

        # Friction coefficients
        mu = sp.symbols(r"\mu")
        self.alg_inputs.append(mu)
        mu_S = sp.symbols(r"\mu_{S}")
        self.alg_inputs.append(mu_S)
        hats_T = sp.symbols(r"\hat{s}_T")
        self.alg_inputs.append(hats_T)

        # System gains
        b_J = sp.symbols(r"b_J")
        self.alg_inputs.append(b_J)
        k_J = sp.symbols(r"k_J")
        self.alg_inputs.append(k_J)

        # Positions
        p_CN = sp.symbols(r"p_{CN}")
        self.alg_inputs.append(p_CN)
        p_CT = sp.symbols(r"p_{CT}")
        self.alg_inputs.append(p_CT)
        p_MN = sp.symbols(r"p_{MN}")
        self.alg_inputs.append(p_MN)
        p_LN = sp.symbols(r"p_{LN}")
        self.alg_inputs.append(p_LN)
        p_LT = sp.symbols(r"p_{LT}")
        self.alg_inputs.append(p_LT)
        theta_L = sp.symbols(r"\theta_L")
        self.alg_inputs.append(theta_L)
        d_T = sp.symbols(r"d_T")
        self.alg_inputs.append(d_T)
        d_N = sp.symbols(r"d_N")
        self.alg_inputs.append(d_N)

        # Velocities
        v_MN = sp.symbols(r"v_{MN}")
        self.alg_inputs.append(v_MN)
        v_MT = sp.symbols(r"v_{MT}")
        self.alg_inputs.append(v_MT)
        v_LN = sp.symbols(r"v_{LN}")
        self.alg_inputs.append(v_LN)
        v_LT = sp.symbols(r"v_{LT}")
        self.alg_inputs.append(v_LT)
        d_theta_L = sp.symbols(r"\dot\theta_L")
        self.alg_inputs.append(d_theta_L)
        d_theta_M = sp.symbols(r"\dot\theta_M")
        self.alg_inputs.append(d_theta_M)
        d_d_T = sp.symbols(r"\dot{d}_T")
        self.alg_inputs.append(d_d_T)
        d_d_N = sp.symbols(r"\dot{d}_N")
        self.alg_inputs.append(d_d_N)

        # Input forces
        F_GT = sp.symbols(r"F_{GT}")
        self.alg_inputs.append(F_GT)
        F_GN = sp.symbols(r"F_{GN}")
        self.alg_inputs.append(F_GN)
        F_OT, F_ON, tau_O = sp.symbols(r"F_{OT}, F_{ON} \tau_O")
        self.alg_inputs.append(F_OT)
        self.alg_inputs.append(F_ON)
        self.alg_inputs.append(tau_O)

        # Control inputs
        a_LNd = sp.symbols(r"a_{LNd}")
        self.alg_inputs.append(a_LNd)

        outputs = [
            a_LT, dd_theta_L, a_MT, a_MN, F_NM, F_FL, F_FM, tau_M, dd_theta_M, dd_d_N, F_NL, F_CN, F_CT, a_LN, dd_d_T,
        ] = sp.symbols(
            r"a_{LT}, \ddot\theta_L, a_{MT}, a_{MN}, F_{NM}, F_{FL}, F_{FM}, \tau_M, \ddot\theta_M, \ddot{d}_N, F_{NL}, F_{CN}, F_{CT}, a_{LN}, \ddot{d}_T"
        )
        outputs = list(outputs)
        self.outputs = outputs

        t = sp.symbols("t")
        theta_L_func = sp.Function(r'\theta_L')(t)
        N_hat = sp.Function(r'\hat N')(theta_L_func)
        T_hat = sp.Function(r'\hat T')(theta_L_func)

        d_T_func = sp.Function(r"d_T")(t)
        d_N_func = sp.Function(r"d_N")(t)
        d_g = d_T_func*T_hat + d_N_func*N_hat

        d_vel_g = sp.diff(d_g, t)

        d_vel_g = d_vel_g.subs(sp.diff(N_hat, t), -
                               sp.diff(theta_L_func, t)*T_hat)
        d_vel_g = d_vel_g.subs(
            sp.diff(T_hat, t), sp.diff(theta_L_func, t)*N_hat)

        d_acc_g = sp.diff(d_vel_g, t)
        d_acc_g = d_acc_g.subs(sp.diff(N_hat, t), -
                               sp.diff(theta_L_func, t)*T_hat)
        d_acc_g = d_acc_g.subs(
            sp.diff(T_hat, t), sp.diff(theta_L_func, t)*N_hat)

        d_acc_cos_g = d_acc_g
        d_acc_cos_g = d_acc_cos_g.subs(sp.diff(theta_L_func, t, t), dd_theta_L)
        d_acc_cos_g = d_acc_cos_g.subs(sp.diff(d_T_func, t, t), dd_d_T)
        d_acc_cos_g = d_acc_cos_g.subs(sp.diff(d_N_func, t, t), dd_d_N)
        d_acc_cos_g = d_acc_cos_g.subs(sp.diff(theta_L_func, t), d_theta_L)
        d_acc_cos_g = d_acc_cos_g.subs(sp.diff(d_T_func, t), d_d_T)
        d_acc_cos_g = d_acc_cos_g.subs(sp.diff(d_N_func, t), d_d_N)
        d_acc_cos_g = d_acc_cos_g.subs(d_T_func, d_T)
        d_acc_cos_g = d_acc_cos_g.subs(d_N_func, d_N)

        dd_d_g_T = d_acc_cos_g.subs(N_hat, 0).subs(T_hat, 1)

        dd_d_g_N = d_acc_cos_g.subs(T_hat, 0).subs(N_hat, 1)

        p_M_func = sp.Function(r"p_M")(t)
        p_L_func = sp.Function(r"p_L")(t)
        v_M = sp.symbols(r"v_M")
        v_L = sp.symbols(r"v_L")
        d_s = (p_M_func + r*N_hat) - (p_L_func + (w_L/2)*T_hat - (h_L/2)*N_hat)

        d_vel_s = sp.diff(d_s, t)
        d_vel_s = d_vel_s.subs(sp.diff(N_hat, t), -
                               sp.diff(theta_L_func, t)*T_hat)
        d_vel_s = d_vel_s.subs(
            sp.diff(T_hat, t), sp.diff(theta_L_func, t)*N_hat)

        d_acc_s = sp.diff(d_vel_s, t)
        d_acc_s = d_acc_s.subs(sp.diff(N_hat, t), -
                               sp.diff(theta_L_func, t)*T_hat)
        d_acc_s = d_acc_s.subs(
            sp.diff(T_hat, t), sp.diff(theta_L_func, t)*N_hat)

        d_acc_cos_s = d_acc_s
        d_acc_cos_s = d_acc_cos_s.subs(sp.diff(theta_L_func, t, t), dd_theta_L)
        d_acc_cos_s = d_acc_cos_s.subs(sp.diff(d_T_func, t, t), dd_d_T)
        d_acc_cos_s = d_acc_cos_s.subs(sp.diff(d_N_func, t, t), dd_d_N)
        d_acc_cos_s = d_acc_cos_s.subs(sp.diff(theta_L_func, t), d_theta_L)
        d_acc_cos_s = d_acc_cos_s.subs(sp.diff(d_T_func, t), d_d_T)
        d_acc_cos_s = d_acc_cos_s.subs(sp.diff(d_N_func, t), d_d_N)
        d_acc_cos_s = d_acc_cos_s.subs(d_T_func, d_T)
        d_acc_cos_s = d_acc_cos_s.subs(d_N_func, d_N)

        dd_d_s_T = d_acc_cos_s.subs(N_hat, 0).subs(T_hat, 1)
        dd_d_s_T = dd_d_s_T.subs(sp.diff(p_M_func, t, t), a_MT)
        dd_d_s_T = dd_d_s_T.subs(sp.diff(p_L_func, t, t), a_LT)
        dd_d_s_T

        dd_d_s_N = d_acc_cos_s.subs(T_hat, 0).subs(N_hat, 1)
        dd_d_s_N = dd_d_s_N.subs(sp.diff(p_M_func, t, t), a_MN)
        dd_d_s_N = dd_d_s_N.subs(sp.diff(p_L_func, t, t), a_LN)
        dd_d_s_N

        nat_eqs = [
            # Link tangential force balance
            [m_L*a_LT, F_FL+F_GT+F_OT],
            # Link normal force balance
            [m_L*a_LN, F_NL + F_GN + F_ON, ],
            # Manipulator tangential force balance
            [m_M*a_MT, F_FM + F_CT, ],
            # Manipulator normal force balance
            [m_M*a_MN, F_NM+F_CN, ],
            # Link moment balance
            [I_L*dd_theta_L, (-w_L/2)*F_ON - (p_CN-p_LN) * \
             F_FL + (p_CT-p_LT)*F_NL + tau_O, ],
            # Manipulator moment balance
            [I_M*dd_theta_M, tau_M-F_FM*(p_CN-p_MN), ],
            # 3rd law normal forces
            [F_NL, -F_NM],
            # Friction relationship L
            [F_FL, mu*mu_S*F_NL*hats_T],
            # Friction relationship M
            [F_FM, -F_FL],
            # d_T derivative is derivative
            [dd_d_s_T, dd_d_g_T],
            # d_N derivative is derivative
            [dd_d_s_N, dd_d_g_N],
            # No penetration
            [dd_d_N, 0],
        ]
        env_eqs = nat_eqs

        A = []
        b = []
        for lhs, rhs in env_eqs:
            A_row = []
            b_term = rhs - lhs
            for output_term in outputs:
                try:
                    coeff_L = lhs.coeff(output_term)
                except AttributeError:
                    coeff_L = 0
                try:
                    coeff_R = rhs.coeff(output_term)
                except AttributeError:
                    coeff_R = 0
                coeff = coeff_L - coeff_R
                A_row.append(coeff)
                b_term += coeff * output_term
            A.append(A_row)
            b.append(b_term)
        A = sp.Matrix(A)
        A.simplify()
        self.A = A
        b = sp.Matrix([b]).T
        b.simplify()
        self.b = b
        x = sp.Matrix([outputs]).T
        x.simplify()
        self.x = x

        L, U, perm = A.LUdecomposition()
        P = sp.eye(A.rows).permuteFwd(perm)

        y = sp.MatrixSymbol("y", L.shape[0], 1).as_explicit()
        y_00 = (P*b)[0]
        y_out = y.subs(y[0,0], y_00)
        for i in range(1, L.shape[0]):
            y_i0 = y[i,0] - (L*y_out - P*b)[i]
            y_out = y_out.subs(y[i,0], y_i0)
        y = y_out.simplify()

        assert (L*y - P*b).simplify() == sp.zeros(L.shape[0], 1)

        self.A_prime = A_prime = U
        self.b_prime = b_prime = y

        # Grab indices
        F_CN_idx = list(x).index(F_CN)
        F_CT_idx = list(x).index(F_CT)
        a_LN_idx = list(x).index(a_LN)
        F_NL_idx = list(x).index(F_NL)

        ## F_CN
        # Slice
        F_CN_eq_lhs = (A_prime)[F_CN_idx-1,:]
        F_CN_eq_rhs = (b_prime)[F_CN_idx-1]
        F_CN_eq_rhs /= F_CN_eq_lhs[F_CN_idx] # rhs has to go first to do anything
        F_CN_eq_lhs /= F_CN_eq_lhs[F_CN_idx]
        for (i, elem) in enumerate(F_CN_eq_lhs):
            if (i == F_CN_idx) or (i == a_LN_idx):
                continue
            assert elem == 0

        # Grab coeffs
        F_CN__a_LN_coef = F_CN_eq_lhs[a_LN_idx]
        F_CN__constant_coef = F_CN_eq_rhs

        # Pack expressions
        self.F_CN_exp = F_CN__constant_coef - F_CN__a_LN_coef*a_LNd

        ## F_CT
        F_CT_eq_lhs = (A_prime)[F_CT_idx-1,:]
        F_CT_eq_rhs = (b_prime)[F_CT_idx-1]
        F_CT_eq_rhs /= F_CT_eq_lhs[F_CT_idx] # rhs has to go first to do anything
        F_CT_eq_lhs /= F_CT_eq_lhs[F_CT_idx]

        # Grab coeffs
        F_CT__a_LN_mixed_coef = F_CT_eq_lhs[a_LN_idx]
        F_CT__a_LN_mu_coef = F_CT__a_LN_mixed_coef.expand().coeff(mu)
        F_CT__a_LN_only_coef = (F_CT__a_LN_mixed_coef - F_CT__a_LN_mu_coef*mu).simplify()

        F_CT__constant_mixed_coef = F_CT_eq_rhs
        F_CT__constant_mu_coef = F_CT__constant_mixed_coef.expand().coeff(mu)
        F_CT__constant_only_coef = (F_CT__constant_mixed_coef - F_CT__constant_mu_coef*mu).simplify()

        ## F_NL
        F_NL_eq_lhs = (A_prime)[F_NL_idx-1,:]
        F_NL_eq_rhs = (b_prime)[F_NL_idx-1]
        F_NL_eq_rhs /= F_NL_eq_lhs[F_NL_idx] # rhs has to go first to do anything
        F_NL_eq_lhs /= F_NL_eq_lhs[F_NL_idx]
        F_NL_eq_lhs /= F_NL_eq_lhs[F_NL_idx]
        for (i, elem) in enumerate(F_NL_eq_lhs):
            if (i == F_NL_idx) or (i == a_LN_idx):
                continue
            assert elem == 0

        # Grab coeffs
        F_NL__a_LN_coef = F_NL_eq_lhs[a_LN_idx]
        F_NL__constant_coef = F_NL_eq_rhs

        # Pack expressions
        self.F_NL_exp = F_NL__constant_coef - F_NL__a_LN_coef*a_LNd

        # Pack adaptive terms
        self.f_mu_exp = F_CT__a_LN_mu_coef
        self.f_exp = F_CT__a_LN_only_coef
        self.g_mu_exp = F_CT__constant_mu_coef
        self.g_exp = F_CT__constant_only_coef

        # Generate output functions
        self.get_F_CN = lambdify([self.alg_inputs], self.F_CN_exp)
        self.get_F_NL = lambdify([self.alg_inputs], self.F_NL_exp)
        self.get_f = lambdify([self.alg_inputs], self.f_exp)
        self.get_f_mu = lambdify([self.alg_inputs], self.f_mu_exp)
        self.get_g = lambdify([self.alg_inputs], self.g_exp)
        self.get_g_mu = lambdify([self.alg_inputs], self.g_mu_exp)


    def latex_to_str(self, sym):
        out = str(sym)
        out = re.sub(r"\\ddot\{([^}]*)\}", r"dd_\1", out)
        out = re.sub(r"\\dot\{([^}]*)\}", r"d_\1", out)
        out = out.replace(r"\ddot", "dd_")
        out = out.replace(r"\dot", "d_")
        out = out.replace(r"{", "").replace(r"}", "").replace("\\", "")
        return out
