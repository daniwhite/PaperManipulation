"""Functions for creating and controlling the finger manipulator."""
# General imports
import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify
import re
from collections import defaultdict

# Local imports
import constants
import manipulator
import paper
import pedestal
from common import get_contact_point_from_results

# Drake imports
import pydrake
from pydrake.multibody.tree import SpatialInertia, UnitInertia, JacobianWrtVariable
from pydrake.all import BasicVector, MultibodyPlant, ContactResults, SpatialVelocity, SpatialForce, FindResourceOrThrow, RigidTransform, RotationMatrix, AngleAxis, RollPitchYaw
from pydrake.all import (
    MathematicalProgram, Solve, eq, le, ge
)

if constants.USE_NEW_MESHCAT:
    import sys
    sys.path.append("manipulation/")
    from manipulation.meshcat_cpp_utils import StartMeshcat, AddMeshcatTriad


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

class FoldingController(pydrake.systems.framework.LeafSystem):
    """Base class for implementing a controller for whatever."""

    def __init__(self, manipulator_acc_log, ll_idx, contact_body_idx, options, sys_params, jnt_frc_log):
        pydrake.systems.framework.LeafSystem.__init__(self)
        self.contacts = []

        # Initialize system parameters
        self.v_stiction = sys_params['v_stiction']
        self.I_L = sys_params['I_L']
        self.w_L = sys_params['w_L']
        self.m_L = sys_params['m_L']
        self.b_J = sys_params['b_J']
        self.k_J = sys_params['k_J']
        self.g = sys_params['g']

        # Initialize control targets
        self.d_Td = -0.12
        self.v_LNd = 0.1
        self.d_Xd = 0
        self.pre_contact_v_MNd = 0.1
        self.theta_MYd = None
        self.theta_MZd = None

        # Initialize estimates
        self.mu_hat = 0.8

        # Initialize gains
        self.k = 10
        self.k_F = 100
        self.K_centering = 1
        self.D_centering = 0.1
        self.lamda = 100 # Sliding surface time constant
        self.P = 10000 # Adapatation law gain
        self.pre_contact_Kp = 10

        # Initialize logs
        self.manipulator_acc_log = manipulator_acc_log
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

        self.manipulator_plant = MultibodyPlant(constants.DT)
        manipulator.data["add_plant_function"](self.manipulator_plant)
        self.manipulator_plant.Finalize()
        self.manipulator_plant_context = self.manipulator_plant.CreateDefaultContext()

        # Other parameters
        self.ll_idx = ll_idx
        self.contact_body_idx = contact_body_idx
        self.use_friction_adaptive_ctrl = options['use_friction_adaptive_ctrl']
        self.use_friction_robust_adaptive_ctrl = options['use_friction_robust_adaptive_ctrl']
        self.d_d_N_sqr_log_len = 100
        self.d_d_N_sqr_lim = 2e-4

        self.nq_manipulator = self.manipulator_plant.get_actuation_input_port().size()

        # Input ports
        self.DeclareVectorInputPort("q", BasicVector(self.nq_manipulator*2))
        self.DeclareAbstractInputPort(
            "poses", pydrake.common.value.AbstractValue.Make([RigidTransform(), RigidTransform()]))
        self.DeclareAbstractInputPort(
            "vels", pydrake.common.value.AbstractValue.Make([SpatialVelocity(), SpatialVelocity()]))
        self.DeclareAbstractInputPort(
            "contact_results",
            pydrake.common.value.AbstractValue.Make(ContactResults()))

        # Output ports
        self.DeclareVectorOutputPort(
            "actuation", pydrake.systems.framework.BasicVector(
                self.nq_manipulator),
            self.CalcOutput)


    def set_meshcat(self, meshcat):
        self.meshcat = meshcat


    def process_contact_results(self, contact_results):
        contact_point = None
        slip_speed = None
        pen_depth = None
        N_hat = None
        self.contacts.append([])
        for i in range(contact_results.num_point_pair_contacts()):
            point_pair_contact_info = \
                contact_results.point_pair_contact_info(i)

            a_idx = int(point_pair_contact_info.bodyA_index())
            b_idx = int(point_pair_contact_info.bodyB_index())
            self.contacts[-1].append([a_idx, b_idx])

            if ((a_idx == self.ll_idx) and (b_idx == self.contact_body_idx) or
                    (a_idx == self.contact_body_idx) and (b_idx == self.ll_idx)):
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
        in_contact = raw_in_contact and self.debug['times'][-1] - self.t_contact_start > 0.5

        ret = {
            "contact_point": contact_point, "slip_speed": slip_speed,
            "pen_depth": pen_depth, "N_hat": N_hat,
            "raw_in_contact": raw_in_contact, "in_contact": in_contact,
        }
        return ret


    def evaluate_manipulator(self, state):
        q = state[:self.nq_manipulator]
        v = state[self.nq_manipulator:]
        self.manipulator_plant.SetPositions(self.manipulator_plant_context, q)
        self.manipulator_plant.SetVelocities(self.manipulator_plant_context, v)

        M = self.manipulator_plant.CalcMassMatrixViaInverseDynamics(self.manipulator_plant_context)
        Cv = np.expand_dims(self.manipulator_plant.CalcBiasTerm(self.manipulator_plant_context), 1)

        tau_g = self.manipulator_plant.CalcGravityGeneralizedForces(self.manipulator_plant_context)

        contact_body = self.manipulator_plant.GetBodyByName(
            manipulator.data["contact_body_name"])
        J = self.manipulator_plant.CalcJacobianSpatialVelocity(
            self.manipulator_plant_context,
            JacobianWrtVariable.kV,
            contact_body.body_frame(),
            [0, 0, 0],
            self.manipulator_plant.world_frame(),
            self.manipulator_plant.world_frame())
        Jdot_qdot_raw = self.manipulator_plant.CalcBiasSpatialAcceleration(
            self.manipulator_plant_context,
            JacobianWrtVariable.kV,
            contact_body.body_frame(),
            [0, 0, 0],
            self.manipulator_plant.world_frame(),
            self.manipulator_plant.world_frame())

        Jdot_qdot = np.expand_dims(np.array(list(Jdot_qdot_raw.rotational()) + list(Jdot_qdot_raw.translational())), 1)

        J_rotational = J[:3,:]
        J_translational = J[3:,:]

        ret = {
            "q": q, "v": v, "tau_g": tau_g, "M": M, "Cv": Cv, "J": J,
            "Jdot_qdot": Jdot_qdot, "J_translational": J_translational,
            "J_rotational": J_rotational
        }
        return ret


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
        self.debug['T_hat'].append(T_hat)
        self.debug['T_hat_geos'].append(T_hat_geo)
        self.debug['N_hat'].append(N_hat)
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
        self.debug['p_L'].append(p_L)
        p_LT = get_T_proj(p_L)
        p_LN = get_N_proj(p_L)

        p_M = np.array([poses[self.contact_body_idx].translation()[0:3]]).T
        p_MN = get_N_proj(p_M)

        pose_M = poses[self.contact_body_idx]
        rot_vec_M = RollPitchYaw(pose_M.rotation()).vector()
        theta_MX = rot_vec_M[0]
        theta_MY = rot_vec_M[1]
        theta_MZ = rot_vec_M[2]

        pose_L = poses[self.ll_idx]
        rot_vec_L = RollPitchYaw(pose_L.rotation()).vector()
        theta_L = rot_vec_L[0]

        p_LLE = N_hat * -h_L/2 + T_hat * w_L/2
        p_LE = p_L + p_LLE

        p_LEM = p_M - p_LE
        # p_LEMT = get_T_proj(p_LEM)
        p_LEMT = (p_LEM.T@T_hat)[0,0]

        # Velocities
        d_theta_L = vels[self.ll_idx].rotational()[0]
        d_theta_MX = vels[self.contact_body_idx].rotational()[0]
        d_theta_MY = vels[self.contact_body_idx].rotational()[1]
        d_theta_MZ = vels[self.contact_body_idx].rotational()[2]
        omega_vec_L = np.array([[d_theta_L, 0, 0]]).T
        omega_vec_M = vels[self.contact_body_idx].rotational()

        v_L = np.array([vels[self.ll_idx].translational()[0:3]]).T
        v_LN = get_N_proj(v_L)
        v_LT = get_T_proj(v_L)

        v_M = np.array([vels[self.contact_body_idx].translational()[0:3]]).T
        v_MN = get_N_proj(v_M)
        v_MT = get_T_proj(v_M)
        vel_M = np.array([list(vels[self.contact_body_idx].rotational()) + \
                          list(vels[self.contact_body_idx].translational())]).T

        # Assume for now that link edge is not moving
        # PROGRAMMING: Get rid of this assumption
        v_LEM = v_M
        v_LEMT = get_T_proj(v_LEM)

        p_LEMX = p_LEM[0]
        v_LEMX = v_LEM[0]

        # Gravity
        F_G = np.array([[0, 0, -self.m_L*constants.g]]).T
        F_GT = self.get_T_proj(F_G)
        F_GN = self.get_N_proj(F_G)

        ret = {
            "m_L": m_L, "h_L": h_L, "w_L": w_L, "I_L": I_L,
            "d_theta_L": d_theta_L, "F_OT": F_OT, "F_ON": F_ON, "tau_O": tau_O,
            "p_LN": p_LN, "p_LT": p_LT, "theta_L": theta_L, "p_LE": p_LE,
            "p_M": p_M, "p_L": p_L, "N_proj_mat": N_proj_mat, "v_L": v_L,
            "v_M": v_M, "omega_vec_L": omega_vec_L, "omega_vec_M": omega_vec_M,
            "v_LT": v_LT, "v_LN": v_LN, "v_MT": v_MT, "v_MN": v_MN,
            "p_LEMT": p_LEMT, "v_LEMT": v_LEMT, "T_hat": T_hat,
            "N_hat": N_hat, "p_LEMX": p_LEMX, "v_LEMX": v_LEMX, "pose_M": pose_M,
            "vel_M": vel_M, "theta_MX": theta_MX, "theta_MY": theta_MY, "theta_MZ": theta_MZ, "d_theta_MX": d_theta_MX,
            "F_GT": F_GT, "F_GN": F_GN, "pose_L": pose_L, "d_theta_MY": d_theta_MY, "d_theta_MZ": d_theta_MZ,
        }
        return ret

    def calc_inputs_contact(self, pen_depth, N_hat, contact_point, slip_speed, p_LE, p_M, p_L, \
            N_proj_mat, d_theta_L, v_L, v_M, omega_vec_L, omega_vec_M, h_L, w_L, \
            v_LT, v_LN, v_MT, v_MN):
        pen_vec = pen_depth*N_hat

        # Positions
        p_C = np.array([contact_point]).T
        p_CT = self.get_T_proj(p_C)
        p_CN = self.get_N_proj(p_C)
        r = np.linalg.norm(p_C-p_M) # TODO: make this support multiple directions


        d = p_C - p_LE
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
        # %DEBUG_APPEND%
        self.debug["v_S"].append(v_S)
        self.debug["v_WConM"].append(v_WConM)
        self.debug["v_WConL"].append(v_WConL)
        self.debug["v_S_raw"].append(v_S_raw)
        self.debug["v_L"].append(v_L)
        self.debug["v_M"].append(v_M)

        s_hat = v_S/np.linalg.norm(v_S)
        hats_T = self.get_T_proj(s_hat)
        s_hat_X = s_hat[0]


        # Forces
        if self.use_friction_adaptive_ctrl:
            mu = self.mu_hat
        else:
            mu_paper = constants.FRICTION
            mu = 2*mu_paper/(1+mu_paper)
        stribeck_mu = stribeck(1, 1, slip_speed/self.v_stiction)
        mu_S = stribeck_mu


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
        ret = {
            "r": r, "mu": mu, "d_N": d_N, "d_T": d_T, "d_d_N": d_d_N,
            "d_d_T": d_d_T, "p_CN": p_CN,
            "p_CT": p_CT,"p_MConM": p_MConM, "mu_S": mu_S,
            "hats_T": hats_T, "s_hat_X": s_hat_X, "d_X": d_X,
            "d_d_X": d_d_X
        }

        return ret

    def CalcOutput(self, context, output):
        ## Load inputs
        # This input put is already restricted to the manipulator, but it includes both q and v
        state = self.get_input_port(0).Eval(context)
        poses = self.get_input_port(1).Eval(context)
        vels = self.get_input_port(2).Eval(context)
        contact_results = self.get_input_port(3).Eval(context)

        # Get time
        # %DEBUG_APPEND%
        self.debug['times'].append(context.get_time())

        # Process contact
        contact_values = self.process_contact_results(contact_results)

        # Process state
        manipulator_values = self.evaluate_manipulator(state)
        q = manipulator_values['q']
        v = manipulator_values['v']
        J = manipulator_values['J']

        # Book keeping
        if self.init_q is None:
            self.init_q = q
        dt = 0
        if len(self.debug['times']) >= 2:
            dt = self.debug['times'][-1] - self.debug['times'][-2]

        # Precompute other inputs
        inputs = self.calc_inputs(poses, vels,
            contact_values['raw_in_contact'], contact_values['N_hat'])

        if self.theta_MYd is None:
            self.theta_MYd = inputs['theta_MY']
        if self.theta_MZd is None:
            self.theta_MZd = inputs['theta_MZ']

        del(contact_values['N_hat'])
        in_contact = contact_values['in_contact']
        if in_contact:
            contact_inputs = self.calc_inputs_contact(pen_depth=contact_values["pen_depth"], N_hat=inputs["N_hat"], contact_point = contact_values["contact_point"], slip_speed = contact_values["slip_speed"], p_LE = inputs["p_LE"], p_M = inputs["p_M"], p_L = inputs["p_L"], N_proj_mat = inputs["N_proj_mat"], d_theta_L = inputs["d_theta_L"], v_L = inputs["v_L"], v_M = inputs["v_M"], omega_vec_L = inputs["omega_vec_L"], omega_vec_M = inputs["omega_vec_M"], h_L = inputs["h_L"], w_L = inputs["w_L"],  v_LT = inputs["v_LT"], v_LN = inputs["v_LN"], v_MT = inputs["v_MT"], v_MN = inputs["v_MN"], )
        
            # # Calculate metric used to tell whether or not contact transients have passed
            # if len(self.d_d_N_sqr_log) < self.d_d_N_sqr_log_len:
            #     self.d_d_N_sqr_log.append(d_d_N**2)
            # else:
            #     self.d_d_N_sqr_log = self.d_d_N_sqr_log[1:] + [d_d_N**2]
            # d_d_N_sqr_sum = sum(self.d_d_N_sqr_log)
        else:

            # %DEBUG_APPEND%
            self.debug["v_S"].append(np.ones((3,1))*np.nan)
            self.debug["v_S"].append(np.ones((3,1))*np.nan)
            self.debug["v_WConM"].append(np.ones((3,1))*np.nan)
            self.debug["v_WConL"].append(np.ones((3,1))*np.nan)
            self.debug["v_S_raw"].append(np.ones((3,1))*np.nan)
            self.debug["v_L"].append(np.ones((3,1))*np.nan)
            self.debug["v_M"].append(np.ones((3,1))*np.nan)
            pass
            # d_d_N_sqr_sum = 0

        # Get torques
        J_plus = np.linalg.pinv(J)
        nullspace_basis = np.eye(self.nq_manipulator) - np.matmul(J_plus, J)

        # Add a PD controller projected into the nullspace of the Jacobian that keeps us close to the nominal configuration
        joint_centering_torque = np.matmul(nullspace_basis, self.K_centering*(self.init_q - q) + self.D_centering*(-v))

        if in_contact:
            tau_ctrl = self.get_contact_control_torques( m_L = inputs["m_L"], h_L = inputs["h_L"], w_L = inputs["w_L"], I_L = inputs["I_L"], r = contact_inputs["r"], mu = contact_inputs["mu"], d_theta_L = inputs["d_theta_L"], d_N = contact_inputs["d_N"], d_T = contact_inputs["d_T"], d_d_N = contact_inputs["d_d_N"], d_d_T = contact_inputs["d_d_T"], F_GT = inputs["F_GT"], F_GN = inputs["F_GN"], F_OT = inputs["F_OT"], F_ON = inputs["F_ON"], tau_O = inputs["tau_O"],  p_CN = contact_inputs["p_CN"], p_CT = contact_inputs["p_CT"], p_LN = inputs["p_LN"], p_LT = inputs["p_LT"], p_MConM = contact_inputs["p_MConM"], theta_L = inputs["theta_L"], mu_S = contact_inputs["mu_S"], hats_T = contact_inputs["hats_T"], s_hat_X = contact_inputs["s_hat_X"], Jdot_qdot = manipulator_values["Jdot_qdot"], J_translational = manipulator_values["J_translational"], J_rotational = manipulator_values["J_rotational"], J = manipulator_values["J"], M = manipulator_values["M"], Cv = manipulator_values["Cv"], tau_g = manipulator_values["tau_g"], joint_centering_torque=joint_centering_torque, theta_MX = inputs["theta_MX"], theta_MY = inputs["theta_MY"], theta_MZ = inputs["theta_MZ"], d_theta_MX = inputs["d_theta_MX"], v_LN = inputs["v_LN"], d_X = contact_inputs["d_X"], d_d_X = contact_inputs["d_d_X"], d_theta_MY = inputs["d_theta_MY"], d_theta_MZ = inputs["d_theta_MZ"])
        else:
            tau_ctrl = self.get_pre_contact_control_torques(
                J, inputs['N_hat'], inputs['v_MN'])

            # Control inputs
            self.debug["dd_d_Td"].append(0)
            self.debug["a_LNd"].append(0)
            self.debug["a_MX_d"].append(0)
            self.debug["alpha_MXd"].append(0)
            self.debug["alpha_MYd"].append(0)
            self.debug["alpha_MZd"].append(0)
            self.debug["dd_d_Nd"].append(0)

            # Decision variables
            self.debug["F_NM"].append(0)
            self.debug["F_FMT"].append(0)
            self.debug["F_FMX"].append(0)
            self.debug["F_ContactMY"].append(0)
            self.debug["F_ContactMZ"].append(0)
            self.debug["F_NL"].append(0)
            self.debug["F_FLT"].append(0)
            self.debug["F_FLX"].append(0)
            for i in range(self.nq_manipulator):
                self.debug["tau_ctrl_" + str(i)].append(0)
            self.debug["a_MX"].append(0)
            self.debug["a_MT"].append(0)
            self.debug["a_MY"].append(0)
            self.debug["a_MZ"].append(0)
            self.debug["a_MN"].append(0)
            self.debug["a_LT"].append(0)
            self.debug["a_LN"].append(0)
            self.debug["alpha_MX"].append(0)
            self.debug["alpha_MY"].append(0)
            self.debug["alpha_MZ"].append(0)
            self.debug["dd_theta_L"].append(0)
            self.debug["dd_d_N"].append(0)
            self.debug["dd_d_T"].append(0)
            for i in range(self.nq_manipulator):
                self.debug["ddq_" + str(i)].append(0)

        tau_out = tau_ctrl - manipulator_values['tau_g'] + joint_centering_torque
        output.SetFromVector(tau_out)

        # Debug
        # %DEBUG_APPEND%
        self.debug["tau_g"].append(manipulator_values['tau_g'])

        # self.debug['F_d'].append(F_d)
        self.debug['tau_ctrl'].append(tau_ctrl)
        self.debug['tau_out'].append(tau_out)
        self.debug['J'].append(J)

        self.debug['raw_in_contact'].append(contact_values['raw_in_contact'])
        self.debug['in_contact'].append(in_contact)

        
        self.debug['F_OTs'].append(inputs['F_OT'])
        self.debug['F_ONs'].append(inputs['F_ON'])
        self.debug['tau_Os'].append(inputs['tau_O'])
        self.debug['mu_ests'].append(self.mu_hat)
        # self.debug['d_d_N_sqr_sum'].append(d_d_N_sqr_sum)
        self.debug['v_LNds'].append(self.v_LNd)
        # self.debug['p_LEMTs'].append(p_LEMT)
        self.debug["M"].append(manipulator_values['M'])
        self.debug["C"].append(manipulator_values['Cv'])
        self.debug["joint_centering_torque"].append(joint_centering_torque)

        # X_PT = RigidTransform()
        # if contact_point is not None:
        #     z_hat = np.array([[0, 0, 1]]).T
        #     axis = np.cross(N_hat, z_hat, axis=0)
        #     axis /= np.sqrt(axis.T@axis)
        #     angle = np.arccos(np.matmul(N_hat.T, z_hat))
        #     angle *= np.sign(N_hat.T@z_hat)
        #     X_PT.set_rotation(AngleAxis(angle=angle, axis=axis))
        #     X_PT.set_translation(contact_point)
        if constants.USE_NEW_MESHCAT:
            AddMeshcatTriad(self.meshcat, "axes/" + "pose_M",
                        length=0.15, radius=0.006, X_PT=inputs["pose_M"])
            AddMeshcatTriad(self.meshcat, "axes/" + "pose_L",
                        length=0.15, radius=0.006, X_PT=inputs["pose_L"])


    def get_pre_contact_control_torques(self, J, N_hat, v_MN):
        self.debug['d_theta_L'].append(np.nan)
        self.debug['d_N'].append(np.nan)
        self.debug['d_T'].append(np.nan)
        self.debug['d_d_N'].append(np.nan)
        self.debug['d_d_T'].append(np.nan)
        self.debug['F_GT'].append(np.nan)
        self.debug['F_GN'].append(np.nan)
        self.debug['F_OT'].append(np.nan)
        self.debug['F_ON'].append(np.nan)
        self.debug['tau_O'].append(np.nan)
        self.debug['p_CN'].append(np.nan)
        self.debug['p_CT'].append(np.nan)
        self.debug['p_LN'].append(np.nan)
        self.debug['p_LT'].append(np.nan)
        self.debug['p_MConM'].append([[np.nan]]*3)
        self.debug['theta_L'].append(np.nan)
        self.debug['mu_S'].append(np.nan)
        self.debug['hats_T'].append(np.nan)
        self.debug['s_hat_X'].append(np.nan)
        self.debug['Jdot_qdot'].append(np.array([[np.nan]]*6))
        self.debug['theta_MX'].append(np.nan)
        self.debug['theta_MY'].append(np.nan)
        self.debug['theta_MZ'].append(np.nan)
        self.debug['d_theta_MX'].append(np.nan)
        self.debug['d_theta_MY'].append(np.nan)
        self.debug['d_theta_MZ'].append(np.nan)
        self.debug["theta_MXd"].append(np.nan)


        # TODO: add controller for hitting correct orientation
        # Proportional control for moving towards the link
        v_MNd = self.pre_contact_v_MNd
        Kp = self.pre_contact_Kp
        F_CN = (v_MNd - v_MN)*Kp

        F_d = np.zeros((6,1))
        if self.debug['times'][-1] > paper.settling_time:
            F_d[3:] += N_hat * F_CN

        tau_ctrl = (J.T@F_d).flatten()

        return tau_ctrl

    def get_contact_control_torques(self, \
            m_L, h_L, w_L, I_L, r, mu, 
            d_theta_L, d_N, d_T, d_d_N, d_d_T, \
            F_GT, F_GN, F_OT, F_ON, tau_O, 
            p_CN, p_CT, p_LN, p_LT, p_MConM, theta_L,
            mu_S, hats_T, s_hat_X,
            Jdot_qdot, J_translational, J_rotational, J,
            M, Cv, tau_g, joint_centering_torque, theta_MX, theta_MY, theta_MZ, d_theta_MX, \
            v_LN, d_X, d_d_X, d_theta_MY, d_theta_MZ):
        
        tau_g = np.expand_dims(tau_g, 1)
        assert tau_g.shape == (self.nq_manipulator, 1)
        joint_centering_torque = np.expand_dims(joint_centering_torque, 1)
        assert joint_centering_torque.shape == (self.nq_manipulator, 1)

        self.debug['d_theta_L'].append(d_theta_L)
        self.debug['d_N'].append(d_N)
        self.debug['d_T'].append(d_T)
        self.debug['d_d_N'].append(d_d_N)
        self.debug['d_d_T'].append(d_d_T)
        self.debug['F_GT'].append(F_GT)
        self.debug['F_GN'].append(F_GN)
        self.debug['F_OT'].append(F_OT)
        self.debug['F_ON'].append(F_ON)
        self.debug['tau_O'].append(tau_O)
        self.debug['p_CN'].append(p_CN)
        self.debug['p_CT'].append(p_CT)
        self.debug['p_LN'].append(p_LN)
        self.debug['p_LT'].append(p_LT)
        self.debug['p_MConM'].append(p_MConM)
        self.debug['theta_L'].append(theta_L)
        self.debug['mu_S'].append(mu_S)
        self.debug['hats_T'].append(hats_T)
        self.debug['s_hat_X'].append(s_hat_X)
        self.debug['Jdot_qdot'].append(Jdot_qdot)
        self.debug['theta_MX'].append(theta_MX)
        self.debug['theta_MY'].append(theta_MY)
        self.debug['theta_MZ'].append(theta_MZ)
        self.debug['d_theta_MX'].append(d_theta_MX)
        self.debug['d_theta_MY'].append(d_theta_MY)
        self.debug['d_theta_MZ'].append(d_theta_MZ)

        ## 1. Define an instance of MathematicalProgram
        prog = MathematicalProgram()

        ## 2. Add decision variables
        # Contact values
        F_NM = prog.NewContinuousVariables(1, 1, name="F_NM")
        F_FMT = prog.NewContinuousVariables(1, 1, name="F_FMT")
        F_FMX = prog.NewContinuousVariables(1, 1, name="F_FMX")
        F_ContactMY = prog.NewContinuousVariables(1, 1, name="F_ContactMY")
        F_ContactMZ = prog.NewContinuousVariables(1, 1, name="F_ContactMZ")
        F_NL = prog.NewContinuousVariables(1, 1, name="F_NL")
        F_FLT = prog.NewContinuousVariables(1, 1, name="F_FLT")
        F_FLX = prog.NewContinuousVariables(1, 1, name="F_FLX")
        F_ContactM_XYZ = np.array([F_FMX, F_ContactMY, F_ContactMZ])[:,:,0]

        # Control values
        tau_ctrl = prog.NewContinuousVariables(self.nq_manipulator, 1, name="tau_ctrl")

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
        alpha_a_MXYZ = np.array([alpha_MX, alpha_MY, alpha_MZ, a_MX, a_MY, a_MZ])[:,:,0]

        # Derived accelerations
        dd_theta_L = prog.NewContinuousVariables(1, 1, name="dd_theta_L")
        dd_d_N = prog.NewContinuousVariables(1, 1, name="dd_d_N")
        dd_d_T = prog.NewContinuousVariables(1, 1, name="dd_d_T")

        ddq = prog.NewContinuousVariables(self.nq_manipulator, 1, name="ddq")

        # prog.AddCost(np.matmul(tau_ctrl.T, tau_ctrl)[0,0])

        ## Constraints
        # "set_description" calls gives us useful names for printing
        prog.AddConstraint(
            eq(m_L*a_LT, F_FLT+F_GT+F_OT)).evaluator().set_description(
                "Link tangential force balance")
        prog.AddConstraint(eq(m_L*a_LN, F_NL + F_GN + F_ON)).evaluator().set_description(
                "Link normal force balance") 
        prog.AddConstraint(eq(I_L*dd_theta_L, \
            (-w_L/2)*F_ON - (p_CN-p_LN) * F_FLT + (p_CT-p_LT)*F_NL + tau_O)).evaluator().set_description(
                "Link moment balance") 
        prog.AddConstraint(eq(F_NL, -F_NM)).evaluator().set_description(
                "3rd law normal forces") 
        prog.AddConstraint(eq(F_FMT, -F_FLT)).evaluator().set_description(
                "3rd law friction forces (T hat)") 
        prog.AddConstraint(eq(F_FMX, -F_FLX)).evaluator().set_description(
                "3rd law friction forces (X hat)") 
        prog.AddConstraint(eq(
            -dd_theta_L*(h_L/2+r) + d_theta_L**2*w_L/2 - a_LT + a_MT,
            -dd_theta_L*d_N + dd_d_T - d_theta_L**2*d_T - 2*d_theta_L*d_d_N
        )).evaluator().set_description("d_N derivative is derivative")
        prog.AddConstraint(eq(
            -dd_theta_L*w_L/2 - d_theta_L**2*h_L/2 - d_theta_L**2*r - a_LN + a_MN,
            dd_theta_L*d_T + dd_d_N - d_theta_L**2*d_N + 2*d_theta_L*d_d_T
        )).evaluator().set_description("d_N derivative is derivative") 
        prog.AddConstraint(eq(dd_d_N, 0)).evaluator().set_description("No penetration")
        prog.AddConstraint(eq(F_FLT, mu_S*F_NL*mu*hats_T)).evaluator().set_description(
            "Friction relationship LT")
        prog.AddConstraint(eq(F_FLX, mu_S*F_NL*mu*s_hat_X)).evaluator().set_description(
            "Friction relationship LX") 
        
        for i in range(6):
            lhs_i = alpha_a_MXYZ[i,0]
            assert not hasattr(lhs_i, "shape")
            rhs_i = (Jdot_qdot + np.matmul(J, ddq))[i,0]
            assert not hasattr(rhs_i, "shape")
            prog.AddConstraint(lhs_i == rhs_i).evaluator().set_description(
                "Relate manipulator and end effector with joint accelerations " + str(i)) 

        tau_contact_trn = np.matmul(J_translational.T, F_ContactM_XYZ)
        tau_contact_rot = np.matmul(J_rotational.T, np.cross(p_MConM, F_ContactM_XYZ, axis=0))
        tau_contact = tau_contact_trn + tau_contact_rot
        tau_out = tau_ctrl - tau_g + joint_centering_torque
        for i in range(self.nq_manipulator):
            M_ddq_row_i = (np.matmul(M, ddq) + Cv)[i,0]
            assert not hasattr(M_ddq_row_i, "shape")
            tau_i = (tau_g + tau_contact + tau_out)[i,0]
            assert not hasattr(tau_i, "shape")
            prog.AddConstraint(M_ddq_row_i == tau_i).evaluator().set_description("Manipulator equations " + str(i))
        
        # Projection equations
        prog.AddConstraint(eq(a_MT, np.cos(theta_L)*a_MY + np.sin(theta_L)*a_MZ))
        prog.AddConstraint(eq(a_MN, -np.sin(theta_L)*a_MY + np.cos(theta_L)*a_MZ))
        prog.AddConstraint(eq(F_FMT, np.cos(theta_L)*F_ContactMY + np.sin(theta_L)*F_ContactMZ))
        prog.AddConstraint(eq(F_NM, -np.sin(theta_L)*F_ContactMY + np.cos(theta_L)*F_ContactMZ))

        # Calculate desired values
        dd_d_Td = 1000*(self.d_Td - d_T) - 100*d_d_T
        a_LNd = 10*(self.v_LNd - v_LN)
        a_MX_d = 100*(self.d_Xd - d_X) - 10 * d_d_X
        theta_MXd = theta_L
        alpha_MXd = 100*(theta_MXd - theta_MX)  + 10*(d_theta_L - d_theta_MX)
        alpha_MYd = 10*(self.theta_MYd - theta_MY) - 5*d_theta_MY
        alpha_MZd = 10*(self.theta_MZd - theta_MZ) - 5*d_theta_MZ
        dd_d_Nd = 0
        prog.AddConstraint(dd_d_T[0,0] == dd_d_Td).evaluator().set_description("Desired dd_d_Td constraint" + str(i))
        prog.AddConstraint(a_LN[0,0] == a_LNd).evaluator().set_description("Desired a_LN constraint" + str(i))
        prog.AddConstraint(a_MX[0,0] == a_MX_d).evaluator().set_description("Desired a_MX constraint" + str(i))
        prog.AddConstraint(alpha_MX[0,0] == alpha_MXd).evaluator().set_description("Desired alpha_MX constraint" + str(i))
        prog.AddConstraint(alpha_MY[0,0] == alpha_MYd).evaluator().set_description("Desired alpha_MY constraint" + str(i))
        prog.AddConstraint(alpha_MZ[0,0] == alpha_MZd).evaluator().set_description("Desired alpha_MZ constraint" + str(i))
        prog.AddConstraint(dd_d_N[0,0] == dd_d_Nd).evaluator().set_description("Desired dd_d_N constraint" + str(i))

        result = Solve(prog)
        assert result.is_success()
        tau_ctrl_result = []
        for i in range(self.nq_manipulator):
            tau_ctrl_result.append(result.GetSolution()[prog.FindDecisionVariableIndex(tau_ctrl[i,0])])

        # %DEBUG_APPEND%
        # control effort
        self.debug["dd_d_Td"].append(dd_d_Td)
        self.debug["a_LNd"].append(a_LNd)
        self.debug["a_MX_d"].append(a_MX_d)
        self.debug["alpha_MXd"].append(alpha_MXd)
        self.debug["alpha_MYd"].append(alpha_MYd)
        self.debug["alpha_MZd"].append(alpha_MZd)
        self.debug["dd_d_Nd"].append(dd_d_Nd)

        # decision variables
        self.debug["F_NM"].append(result.GetSolution()[prog.FindDecisionVariableIndex(F_NM[0,0])])
        self.debug["F_FMT"].append(result.GetSolution()[prog.FindDecisionVariableIndex(F_FMT[0,0])])
        self.debug["F_FMX"].append(result.GetSolution()[prog.FindDecisionVariableIndex(F_FMX[0,0])])
        self.debug["F_ContactMY"].append(result.GetSolution()[prog.FindDecisionVariableIndex(F_ContactMY[0,0])])
        self.debug["F_ContactMZ"].append(result.GetSolution()[prog.FindDecisionVariableIndex(F_ContactMZ[0,0])])
        self.debug["F_NL"].append(result.GetSolution()[prog.FindDecisionVariableIndex(F_NL[0,0])])
        self.debug["F_FLT"].append(result.GetSolution()[prog.FindDecisionVariableIndex(F_FLT[0,0])])
        self.debug["F_FLX"].append(result.GetSolution()[prog.FindDecisionVariableIndex(F_FLX[0,0])])
        for i in range(self.nq_manipulator):
            self.debug["tau_ctrl_" + str(i)].append(result.GetSolution()[prog.FindDecisionVariableIndex(tau_ctrl[i,0])])
        self.debug["a_MX"].append(result.GetSolution()[prog.FindDecisionVariableIndex(a_MX[0,0])])
        self.debug["a_MT"].append(result.GetSolution()[prog.FindDecisionVariableIndex(a_MT[0,0])])
        self.debug["a_MY"].append(result.GetSolution()[prog.FindDecisionVariableIndex(a_MY[0,0])])
        self.debug["a_MZ"].append(result.GetSolution()[prog.FindDecisionVariableIndex(a_MZ[0,0])])
        self.debug["a_MN"].append(result.GetSolution()[prog.FindDecisionVariableIndex(a_MN[0,0])])
        self.debug["a_LT"].append(result.GetSolution()[prog.FindDecisionVariableIndex(a_LT[0,0])])
        self.debug["a_LN"].append(result.GetSolution()[prog.FindDecisionVariableIndex(a_LN[0,0])])
        self.debug["alpha_MX"].append(result.GetSolution()[prog.FindDecisionVariableIndex(alpha_MX[0,0])])
        self.debug["alpha_MY"].append(result.GetSolution()[prog.FindDecisionVariableIndex(alpha_MY[0,0])])
        self.debug["alpha_MZ"].append(result.GetSolution()[prog.FindDecisionVariableIndex(alpha_MZ[0,0])])
        self.debug["dd_theta_L"].append(result.GetSolution()[prog.FindDecisionVariableIndex(dd_theta_L[0,0])])
        self.debug["dd_d_N"].append(result.GetSolution()[prog.FindDecisionVariableIndex(dd_d_N[0,0])])
        self.debug["dd_d_T"].append(result.GetSolution()[prog.FindDecisionVariableIndex(dd_d_T[0,0])])
        for i in range(self.nq_manipulator):
            self.debug["ddq_" + str(i)].append(result.GetSolution()[prog.FindDecisionVariableIndex(ddq[i,0])])
        self.debug["theta_MXd"].append(theta_MXd)

        return tau_ctrl_result
