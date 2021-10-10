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
from pydrake.all import BasicVector, MultibodyPlant, ContactResults, SpatialVelocity, SpatialForce, FindResourceOrThrow, RigidTransform, RotationMatrix, AngleAxis
from pydrake.all import (
    MathematicalProgram, Solve, eq, le, ge,
)

if constants.USE_NEW_MESHCAT:
    import sys
    sys.path.append("manipulation/")
    from manipulation.meshcat_cpp_utils import StartMeshcat, AddMeshcatTriad

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


def step5(x):
    '''Python version of MultibodyPlant::StribeckModel::step5 method'''
    x3 = x * x * x
    return x3 * (10 + x * (6 * x - 15))


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
        self.k_F = 100
        self.K_centering = 1
        self.D_centering = 0.1
        self.lamda = 100 # Sliding surface time constant
        self.P = 10000 # Adapatation law gain

        # Initialize logs
        self.arm_acc_log = arm_acc_log
        # %DEBUG_APPEND%
        self.debug = defaultdict(list)
        self.debug['times'] = []
        self.jnt_frc_log = jnt_frc_log
        self.jnt_frc_log.append(SpatialForce(np.zeros((3, 1)), np.zeros((3, 1))))
        self.d_d_N_sqr_log = []

        # Initialize intermediary variables
        self.last_v_LN = 0
        self.init_q = None
        self.t_contact_start =  None
        self.last_Jdot_qdot = np.zeros((6, 1))

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


    def set_meshcat(self, meshcat):
        self.meshcat = meshcat


    def process_contact_results(self, contact_results):
        contact_point = None
        slip_speed = None
        pen_depth = None
        N_hat = None
        self.contacts = []
        for i in range(contact_results.num_point_pair_contacts()):
            point_pair_contact_info = \
                contact_results.point_pair_contact_info(i)

            a_idx = int(point_pair_contact_info.bodyA_index())
            b_idx = int(point_pair_contact_info.bodyB_index())
            self.contacts.append([a_idx, b_idx])

            if ((a_idx == self.ll_idx) and (b_idx == self.finger_idx) or
                    (a_idx == self.finger_idx) and (b_idx == self.ll_idx)):
                contact_point = point_pair_contact_info.contact_point()
                slip_speed = point_pair_contact_info.slip_speed()
                pen_point_pair = point_pair_contact_info.point_pair()
                pen_depth = pen_point_pair.depth
                # PROGRAMMING: check sign
                N_hat = np.expand_dims(pen_point_pair.nhat_BA_W, 1)

        # Get contact times
        raw_in_contact = (not (contact_point is None))
        if raw_in_contact:
            if self.t_contact_start is None:
                self.t_contact_start = self.debug['times'][-1]
        else:
            self.t_contact_start =  None
        in_contact = raw_in_contact and self.debug['times'][-1] - self.t_contact_start > 0.002

        return contact_point, slip_speed, pen_depth, N_hat, raw_in_contact, in_contact


    def evaluate_arm(self, state):
        q = state[:self.nq_arm]
        v = state[self.nq_arm:]
        self.arm_plant.SetPositions(self.arm_plant_context, q)
        self.arm_plant.SetVelocities(self.arm_plant_context, v)

        M = self.arm_plant.CalcMassMatrixViaInverseDynamics(self.arm_plant_context)
        Cv = self.arm_plant.CalcBiasTerm(self.arm_plant_context)

        tau_g = self.arm_plant.CalcGravityGeneralizedForces(self.arm_plant_context)

        finger_body = self.arm_plant.GetBodyByName(FINGER_NAME)
        J = self.arm_plant.CalcJacobianSpatialVelocity(
            self.arm_plant_context,
            JacobianWrtVariable.kQDot,
            finger_body.body_frame(),
            [0, 0, 0],
            self.arm_plant.world_frame(),
            self.arm_plant.world_frame())
        
        Jdot_qdot = self.last_Jdot_qdot
        if len(self.debug['times']) > 1:
            dt_for_Jdot = self.debug['times'][-1] - self.debug['times'][-2]
            if dt_for_Jdot > 0 and len(self.debug['J']) > 0:
                Jdot = (J - self.debug['J'][-1])/(dt_for_Jdot)
                Jdot_qdot = np.expand_dims(np.matmul(Jdot, v), 1)
        self.last_Jdot_qdot = Jdot_qdot

        J_rotational = J[:3,:]
        J_translational = J[3:,:]

        return q, v, tau_g, M, Cv, J, Jdot_qdot, J_translational, J_rotational


    def calc_inputs(self, poses, vels, in_contact, N_hat):
        jnt_frcs = self.jnt_frc_log[-1]
        F_OT = jnt_frcs.translational()[1]
        F_ON = jnt_frcs.translational()[2]
        tau_O = jnt_frcs.rotational()[0]

        # Directions
        R = poses[self.ll_idx].rotation()
        y_hat = np.array([[0, 1, 0]]).T
        z_hat = np.array([[0, 0, 1]]).T
        if not in_contact:
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
        T_hat_geo = R@y_hat
        N_hat_geo = R@z_hat
        # %DEBUG_APPEND%
        self.debug['T_hat_geos'].append(T_hat_geo)
        self.debug['N_hat_geos'].append(N_hat_geo)

        T_proj_mat = T_hat@(T_hat.T)
        N_proj_mat = N_hat@(N_hat.T)

        # Helper functions
        def get_T_proj(vec):
            T_vec = np.matmul(T_proj_mat, vec)
            T_mag = np.linalg.norm(T_vec)
            T_sgn = np.sign(np.matmul(T_hat.T, T_vec))
            T = T_mag.flatten()*T_sgn.flatten()
            return T[0]
        self.get_T_proj = get_T_proj

        def get_N_proj(vec):
            N_vec = np.matmul(N_proj_mat, vec)
            N_mag = np.linalg.norm(N_vec)
            N_sgn = np.sign(np.matmul(N_hat.T, N_vec))
            N = N_mag.flatten()*N_sgn.flatten()
            return N[0]
        self.get_N_proj = get_N_proj

        # Constants
        w_L = self.w_L
        h_L = paper.PAPER_HEIGHT
        # m_M = finger.MASS # TODO get rid of
        m_L = self.m_L
        I_L = self.I_L
        # I_M = self.I_M
        # b_J = self.b_J
        # k_J = self.k_J

        # Positions
        p_L = np.array([poses[self.ll_idx].translation()[0:3]]).T
        p_LT = get_T_proj(p_L)
        p_LN = get_N_proj(p_L)

        p_M = np.array([poses[self.finger_idx].translation()[0:3]]).T
        p_MN = get_N_proj(p_M)

        angle_axis = poses[self.ll_idx].rotation().ToAngleAxis()
        theta_L = angle_axis.angle()
        if sum(angle_axis.axis()) < 0:
            theta_L *= -1

        p_LLE = N_hat * -h_L/2 + T_hat * w_L/2
        p_LE = p_L + p_LLE

        p_LEM = p_M - p_LE
        # p_LEMT = get_T_proj(p_LEM)
        p_LEMT = (p_LEM.T@T_hat)[0,0]

        # Velocities
        d_theta_L = vels[self.ll_idx].rotational()[0]
        d_theta_M = vels[self.finger_idx].rotational()[0]
        omega_vec_L = np.array([[d_theta_L, 0, 0]]).T
        omega_vec_M = np.array([[d_theta_M, 0, 0]]).T

        v_L = np.array([vels[self.ll_idx].translational()[0:3]]).T
        v_LN = get_N_proj(v_L)
        v_LT = get_T_proj(v_L)

        v_M = np.array([vels[self.finger_idx].translational()[0:3]]).T
        v_MN = get_N_proj(v_M)
        v_MT = get_T_proj(v_M)

        # Assume for now that link edge is not moving
        # PROGRAMMING: Get rid of this assumption
        v_LEM = v_M
        v_LEMT = get_T_proj(v_LEM)

        p_LEMX = p_LEM[0]
        v_LEMX = v_LEM[0]

        return m_L, h_L, w_L, I_L,  d_theta_L, F_OT, F_ON, tau_O, p_LN, p_LT, theta_L, p_LE, p_M, \
            p_L, N_proj_mat, v_L, v_M, omega_vec_L, omega_vec_M, v_LT, v_LN, v_MT, v_MN, p_LEMT, \
            v_LEMT, T_hat, N_hat, p_LEMX, v_LEMX

    def calc_inputs_contact(self, pen_depth, N_hat, contact_point, slip_speed, p_LE, p_M, p_L, \
            N_proj_mat, d_theta_L, v_L, v_M, omega_vec_L, omega_vec_M, h_L, w_L, \
            v_LT, v_LN, v_MT, v_MN):
        pen_vec = pen_depth*N_hat

        # Positions
        p_C = np.array([contact_point]).T
        p_CT = self.get_T_proj(p_C)
        p_CN = self.get_N_proj(p_C)
        r = np.linalg.norm(p_C-p_M) # TODO: make this support multiple directions


        d = p_C - p_LE + pen_vec/2
        d_T = self.get_T_proj(d)
        d_N = self.get_N_proj(d)
        d_X = d[0]

        p_MConM = p_C - p_M
        p_LConL = p_C - p_L

        # Velocities
        v_WConL = v_L + np.cross(omega_vec_L, p_LConL, axis=0)
        v_WConM = v_M + np.cross(omega_vec_M, p_MConM, axis=0)

        d_d_T = -d_theta_L*h_L/2 - d_theta_L*r - v_LT + v_MT + d_theta_L*d_N
        d_d_N = -d_theta_L*w_L/2-v_LN+v_MN-d_theta_L*d_T
        d_d_X = v_WConM[0]

        v_S_raw = v_WConM - v_WConL
        v_S_N = np.matmul(N_proj_mat, v_S_raw)
        v_S = v_S_raw - v_S_N

        s_hat = v_S/np.linalg.norm(v_S)
        hats_T = self.get_T_proj(s_hat)
        s_hat_X = s_hat[0]

        # Targets
        a_LNd = self.a_LNd

        # Forces
        if self.use_friction_adaptive_ctrl:
            mu = self.mu_hat
        else:
            mu_paper = constants.FRICTION
            mu = 2*mu_paper/(1+mu_paper)
        stribeck_mu = stribeck(1, 1, slip_speed/self.v_stiction)
        mu_S = stribeck_mu

        # Gravity
        F_G = np.array([[0, 0, -self.m_L*constants.g]]).T
        F_GT = self.get_T_proj(F_G)
        F_GN = self.get_N_proj(F_G)

        # def sat(phi):
        #     if phi > 0:
        #         return min(phi, 1)
        #     return max(phi, -1)

        # f_mu = self.get_f_mu(inps_)
        # f = self.get_f(inps_)
        # g_mu = self.get_g_mu(inps_)
        # g = self.get_g(inps_)

        # s_d_T = self.lamda*(d_T - self.d_Td) + (d_d_T)
        # s_F = v_LN - self.v_LNd
        # phi = 0.001
        # s_delta_d_T = s_d_T - phi*sat(s_d_T/phi)
        # Y = g_mu - f_mu*a_LNd
        # dt = 0
        # if len(self.d_d_N_sqr_log) >= self.d_d_N_sqr_log_len and d_d_N_sqr_sum < self.d_d_N_sqr_lim: # Check if d_N is oscillating
        #     if len(self.debug['times']) >= 2:
        #         dt = self.debug['times'][-1] - self.debug['times'][-2]
        #         self.mu_hat += -self.P*dt*Y*s_delta_d_T
        #     if self.mu_hat > 1:
        #         self.mu_hat = 1
        #     if self.mu_hat < 0:
        #         self.mu_hat = 0

        # Even if we're not using adaptive control (i.e. learning mu), we'll still the lambda term to implement sliding mode control
        # u_hat = -(f*a_LNd) + g - m_M * self.lamda*d_d_T

        # if self.use_friction_adaptive_ctrl:
        #     Y_mu_term = Y*self.mu_hat
        # else:
        #     Y_mu_term = Y*constants.FRICTION

        # F_CT = u_hat + Y_mu_term
        # if self.use_friction_robust_adaptive_ctrl:
        #     # TODO: remove this option
        #     F_CT += -self.k*s_delta_d_T# - k_robust*np.sign(s_delta_d_T)
        # else:
        #     F_CT += -self.k*s_d_T
        # F_CN = self.get_F_CN(inps_) - self.k*s_F

        return r, mu, d_N, d_T, d_d_N, d_d_T, \
            F_GT, F_GN, p_CN, p_CT, p_MConM, mu_S, \
            hats_T, s_hat_X, d_X, d_d_X

    def CalcOutput(self, context, output):
        ## Load inputs
        # This input put is already restricted to the arm, but it includes both q and v
        state = self.get_input_port(0).Eval(context)
        poses = self.get_input_port(1).Eval(context)
        vels = self.get_input_port(2).Eval(context)
        contact_results = self.get_input_port(3).Eval(context)

        # Get time
        # %DEBUG_APPEND%
        self.debug['times'].append(context.get_time())

        # Process contact
        contact_point, slip_speed, pen_depth, N_hat, raw_in_contact, in_contact = \
            self.process_contact_results(contact_results)

        # Process state
        q, v, tau_g, M, Cv, J, Jdot_qdot, J_translational, J_rotational = self.evaluate_arm(state)

        # Book keeping
        if self.init_q is None:
            self.init_q = q
        dt = 0
        if len(self.debug['times']) >= 2:
            dt = self.debug['times'][-1] - self.debug['times'][-2]
        self.v_LNd += self.a_LNd * dt

        # Precompute other inputs
        m_L, h_L, w_L, I_L,  d_theta_L, F_OT, F_ON, tau_O, p_LN, p_LT, theta_L, p_LE, p_M, \
            p_L, N_proj_mat, v_L, v_M, omega_vec_L, omega_vec_M, v_LT, v_LN, v_MT, v_MN, p_LEMT, \
            v_LEMT, T_hat, N_hat, p_LEMX, v_LEMX \
            = self.calc_inputs(poses, vels, raw_in_contact, N_hat)
        if in_contact:
            r, mu, d_N, d_T, d_d_N, d_d_T, F_GT, F_GN, p_CN, p_CT, p_MConM, mu_S, hats_T, s_hat_X, \
                d_X, d_d_X = self.calc_inputs_contact(pen_depth, N_hat, \
                    contact_point, slip_speed, p_LE, p_M, p_L, N_proj_mat, d_theta_L, v_L, v_M, \
                        omega_vec_L, omega_vec_M, h_L, w_L, v_LT, v_LN, v_MT, v_MN)
        
            # Calculate metric used to tell whether or not contact transients have passed
            if len(self.d_d_N_sqr_log) < self.d_d_N_sqr_log_len:
                self.d_d_N_sqr_log.append(d_d_N**2)
            else:
                self.d_d_N_sqr_log = self.d_d_N_sqr_log[1:] + [d_d_N**2]
            d_d_N_sqr_sum = sum(self.d_d_N_sqr_log)

        # Get torques
        if in_contact:
            tau_ctrl = self.get_contact_control_torques( m_L, h_L, w_L, I_L, r, mu,  d_theta_L, d_N, d_T, \
                d_d_N, d_d_T, F_GT, F_GN, F_OT, F_ON, tau_O,  p_CN, p_CT, p_LN, p_LT, p_MConM, \
                theta_L, mu_S, hats_T, s_hat_X, Jdot_qdot, J_translational, J_rotational, J, M, \
                Cv, tau_g)
        else:
            tau_ctrl = self.get_pre_contact_control_torques(p_LE, p_M, v_M, J_translational)

        J_plus = np.linalg.pinv(J)
        nullspace_basis = np.eye(self.nq_arm) - np.matmul(J_plus, J)

        # Add a PD controller projected into the nullspace of the Jacobian that keeps us close to the nominal configuration
        joint_centering_torque = np.matmul(nullspace_basis, self.K_centering*(self.init_q - q) + self.D_centering*(-v))

        tau_out = tau_ctrl - tau_g + joint_centering_torque
        output.SetFromVector(tau_out)

        # Debug
        # %DEBUG_APPEND%
        self.debug["tau_g"].append(tau_g)

        # self.debug['F_d'].append(F_d)
        self.debug['tau_ctrl'].append(tau_ctrl)
        self.debug['tau_out'].append(tau_out)
        self.debug['J'].append(J)

        self.debug['raw_in_contact'].append(raw_in_contact)
        self.debug['in_contact'].append(in_contact)

        
        self.debug['F_OTs'].append(F_OT)
        self.debug['F_ONs'].append(F_ON)
        self.debug['tau_Os'].append(tau_O)
        self.debug['mu_ests'].append(self.mu_hat)
        # self.debug['d_d_N_sqr_sum'].append(d_d_N_sqr_sum)
        self.debug['v_LNds'].append(self.v_LNd)
        self.debug['p_LEMTs'].append(p_LEMT)
        self.debug["M"].append(M)
        self.debug["C"].append(Cv)


        X_PT = RigidTransform()
        if contact_point is not None:
            z_hat = np.array([[0, 0, 1]]).T
            axis = np.cross(N_hat, z_hat, axis=0)
            axis /= np.sqrt(axis.T@axis)
            angle = np.arccos(np.matmul(N_hat.T, z_hat))
            angle *= np.sign(N_hat.T@z_hat)
            X_PT.set_rotation(AngleAxis(angle=angle, axis=axis))
            X_PT.set_translation(contact_point)
        if constants.USE_NEW_MESHCAT:
            AddMeshcatTriad(self.meshcat, "painter/" + "contact_point",
                        length=0.15, radius=0.006, X_PT=X_PT)


    def get_pre_contact_control_torques(self, p_LE, p_M, v_M, J_translational):
        Kpx = 100
        Kdx = 20

        Kpy = 100
        Kdy = 20

        Kpz = 10

        self.v_MZd = 0.01

        F_CX = -Kpx*p_M[0] - Kdx*v_M[0]
        F_CY = Kpy*(self.d_Td - (p_M[1] - p_LE[1])) - Kdy*v_M[1]
        F_CZ = Kpz*(self.v_MZd-v_M[2])

        F_d = np.array([[F_CX, F_CY, F_CZ]]).T
        tau_ctrl = (J_translational.T@F_d).flatten()

        # %DEBUG_APPEND%
        self.debug['F_CXs'].append(F_CX)
        self.debug['F_CNs'].append(0)
        self.debug['F_CTs'].append(0)

        return tau_ctrl
        

        # F_CT = self.get_pre_contact_F_CT(p_LEMT, v_LEMT)
        # tau_M = 0

        # d_d_N_sqr_sum = np.nan

        # d_X = p_M[0]
        # d_d_X = v_M[0]
        # F_FMX = 0
        # self.tau_ctrl_result = None

    def get_contact_control_torques(self, \
            m_L, h_L, w_L, I_L, r, mu, 
            d_theta_L, d_N, d_T, d_d_N, d_d_T, \
            F_GT, F_GN, F_OT, F_ON, tau_O, 
            p_CN, p_CT, p_LN, p_LT, p_MConM, theta_L,
            mu_S, hats_T, s_hat_X,
            Jdot_qdot, J_translational, J_rotational, J,
            M, Cv, tau_g):
        ## 1. Define an instance of MathematicalProgram
        prog = MathematicalProgram()

        ## 2. Add decision variables
        # Contact values
        tau_contact = prog.NewContinuousVariables(self.nq_arm, 1)
        F_NM = prog.NewContinuousVariables(1, 1)
        F_FMT = prog.NewContinuousVariables(1, 1)
        F_FMX = prog.NewContinuousVariables(1, 1)
        F_ContactMY = prog.NewContinuousVariables(1, 1)
        F_ContactMZ = prog.NewContinuousVariables(1, 1)
        F_NL = prog.NewContinuousVariables(1, 1)
        F_FLT = prog.NewContinuousVariables(1, 1)
        F_FLX = prog.NewContinuousVariables(1, 1)
        F_ContactM_XYZ = np.array([F_FMX, F_ContactMY, F_ContactMZ])[:,:,0]

        # Control values
        tau_ctrl = prog.NewContinuousVariables(7, 1)
        F_CX = prog.NewContinuousVariables(1, 1)
        F_CY = prog.NewContinuousVariables(1, 1)
        F_CZ = prog.NewContinuousVariables(1, 1)
        F_CN = prog.NewContinuousVariables(1, 1)
        F_CT = prog.NewContinuousVariables(1, 1)
        F_CXYZ = np.array([F_CX, F_CY, F_CZ])[:,:,0]

        # Object accelerations
        a_MX = prog.NewContinuousVariables(1, 1)
        a_MT = prog.NewContinuousVariables(1, 1)
        a_MY = prog.NewContinuousVariables(1, 1)
        a_MZ = prog.NewContinuousVariables(1, 1)
        a_MN = prog.NewContinuousVariables(1, 1)
        a_LT = prog.NewContinuousVariables(1, 1)
        a_LN = prog.NewContinuousVariables(1, 1)
        alpha_MX = prog.NewContinuousVariables(1, 1)
        alpha_MY = prog.NewContinuousVariables(1, 1)
        alpha_MZ = prog.NewContinuousVariables(1, 1)
        alpha_a_MXYZ = np.array([alpha_MX, alpha_MY, alpha_MZ, a_MX, a_MY, a_MZ])[:,:,0]

        # Derived accelerations
        dd_theta_L = prog.NewContinuousVariables(1, 1)
        dd_d_N = prog.NewContinuousVariables(1, 1)
        dd_d_T = prog.NewContinuousVariables(1, 1)
        dd_d_s_T = -dd_theta_L*h_L/2 - dd_theta_L*r + d_theta_L**2 - a_LT + a_MT
        dd_d_s_N = -dd_theta_L - (h_L*d_theta_L**2)/2 - r*d_theta_L**2 - a_LN + a_MN
        dd_d_g_T = -dd_theta_L*d_N + dd_d_T - d_T*d_theta_L**2 - 2*d_theta_L*d_d_N
        dd_d_g_N = dd_theta_L*d_T + dd_d_N - d_N*d_theta_L**2 + 2*d_theta_L*d_d_T

        ddq = prog.NewContinuousVariables(7, 1)

        prog.AddCost(np.matmul(tau_ctrl.T, tau_ctrl)[0,0])

        ## Constraints
        # Link tangential force balance
        prog.AddConstraint(eq(m_L*a_LT, F_FLT+F_GT+F_OT))
        # Link normal force balance
        prog.AddConstraint(eq(m_L*a_LN, F_NL + F_GN + F_ON))
        # Link moment balance
        prog.AddConstraint(eq(I_L*dd_theta_L, (-w_L/2)*F_ON - (p_CN-p_LN) * F_FLT + (p_CT-p_LT)*F_NL + tau_O))
        # 3rd law normal forces
        prog.AddConstraint(eq(F_NL, -F_NM))
        # 3rd law friction forces
        prog.AddConstraint(eq(F_FMT, -F_FLT))
        # d_T derivative is derivative
        prog.AddConstraint(eq(dd_d_s_T, dd_d_g_T))
        # d_N derivative is derivative
        prog.AddConstraint(eq(dd_d_s_N, dd_d_g_N))
        # No penetration
        # prog.AddConstraint(eq(dd_d_N, 0))
        # Friction relationship LT
        prog.AddConstraint(eq(F_FLT, mu_S*F_NL*mu*hats_T))
        # Friction relationship LX
        prog.AddConstraint(eq(F_FLX, mu_S*F_NL*mu*s_hat_X))

        # Relate manipulator and end effector with joint velocities in Y direction
        prog.AddConstraint(eq(alpha_a_MXYZ, (Jdot_qdot + np.matmul(J, ddq))))

        # Manipulator equations
        tau_contact_trn = np.matmul(J_translational.T, F_ContactM_XYZ)
        tau_contact_rot = np.matmul(J_rotational.T, np.cross(p_MConM, F_ContactM_XYZ, axis=0))
        tau_contact = tau_contact_trn + tau_contact_rot
        prog.AddConstraint(eq(np.matmul(M, ddq) + Cv, tau_g + tau_contact + tau_ctrl))
        
        # Projection equations
        prog.AddConstraint(eq(a_MT, np.cos(theta_L)*a_MY + np.sin(theta_L)*a_MZ))
        prog.AddConstraint(eq(a_MN, -np.sin(theta_L)*a_MY + np.cos(theta_L)*a_MZ))
        prog.AddConstraint(eq(F_CT, np.cos(theta_L)*F_CY + np.sin(theta_L)*F_CZ))
        prog.AddConstraint(eq(F_CN, -np.sin(theta_L)*F_CY + np.cos(theta_L)*F_CZ))
        prog.AddConstraint(eq(F_FMT, np.cos(theta_L)*F_ContactMY + np.sin(theta_L)*F_ContactMZ))
        prog.AddConstraint(eq(F_NM, -np.sin(theta_L)*F_ContactMY + np.cos(theta_L)*F_ContactMZ))

        # Relate forces and torques
        prog.AddConstraint(eq(tau_ctrl, np.matmul(J_translational.T,F_CXYZ)-tau_g))

        # Desired quantities
        prog.AddConstraint(eq(self.a_LNd, a_LN))
        prog.AddConstraint(eq(dd_d_T, 0))
        prog.AddConstraint(eq(a_MX, 0))
        prog.AddConstraint(eq(alpha_MY, 0))
        # Don't want to rotate around the u axis

        ##
        result = Solve(prog)
        # self.tau_ctrl_result = []
        tau_ctrl_result = []
        for i in range(self.nq_arm):
            tau_ctrl_result.append(result.GetSolution()[prog.FindDecisionVariableIndex(tau_ctrl[i,0])])

        # F_CN = 0.01
        # F_CT = self.get_pre_contact_F_CT(p_LEMT, v_LEMT)
        # F_FMX = 0

        F_CX = result.GetSolution()[prog.FindDecisionVariableIndex(F_CX[0,0])]
        F_CT = result.GetSolution()[prog.FindDecisionVariableIndex(F_CT[0,0])]# -self.k*s_d_T
        F_CN = result.GetSolution()[prog.FindDecisionVariableIndex(F_CN[0,0])]# - self.k_F*s_F
        # %DEBUG_APPEND%
        self.debug['F_CXs'].append(F_CX)
        self.debug['F_CNs'].append(F_CN)
        self.debug['F_CTs'].append(F_CT)

        return tau_ctrl_result

        # s_d_X = self.lamda*(d_X - self.d_Xd) + (d_d_X)
        # F_CX = -self.k*s_d_X# - F_FMX

        # F_M = F_CN*N_hat + F_CT*T_hat
        # F_M[0] = F_CX

        # return F_M.flatten()
