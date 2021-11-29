"""Functions for creating and controlling the finger manipulator."""
# General imports
import numpy as np
from collections import defaultdict
from dataclasses import dataclass

# Local imports
import constants
import manipulator
import paper

import enum

# Drake imports
import pydrake
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.all import (
    MathematicalProgram, Solve, eq, \
    BasicVector, MultibodyPlant, ContactResults, SpatialVelocity, \
    SpatialForce, RigidTransform, RollPitchYaw \
)

if constants.USE_NEW_MESHCAT:
    import sys
    sys.path.append("manipulation/")
    from manipulation.meshcat_cpp_utils import AddMeshcatTriad


def get_attrs_from_class(classname):
    """
    Returns iterator containing all attributes of classname besides the builtin
    ones.
    """
    return filter(lambda s: not s.startswith('__'), dir(classname))


@dataclass
class VisionDerivedData:
    """
    Data derived from vision information.
    """
    T_hat: np.ndarray = np.zeros((3,1))
    N_hat: np.ndarray = np.zeros((3,1))
    theta_L: float = 0
    d_theta_L: float = 0
    pose_L: np.ndarray = np.zeros((3,1))
    p_LN: float = 0
    p_LT: float = 0
    pose_M: np.ndarray = np.zeros((3,1))
    v_LN: float = 0
    v_MN: float = 0
    theta_MX: float = 0
    theta_MY: float = 0
    theta_MZ: float = 0
    d_theta_MX: float = 0
    d_theta_MY: float = 0
    d_theta_MZ: float = 0
    F_GT: float = 0
    F_GN: float = 0
    d_X: float = 0
    d_d_X: float = 0
    d_N: float = 0
    d_d_N: float = 0
    d_T: float = 0
    d_d_T: float = 0
    p_CN: float = 0
    p_CT: float = 0
    p_MConM: np.ndarray = np.zeros((3,1))
    mu_S: float = 0
    hats_T: float = 0
    s_hat_X: float = 0
    in_contact: bool = False


@dataclass
class CheatPortsData:
    """
    Data which we cannot practically measure, but is helpful for constructing a
    'perfect model' simulation.
    """
    F_OT: float = 0
    F_ON: float = 0
    tau_O: float = 0


@dataclass
class ManipulatorData:
    """
    Data about the manipulator itself.
    """
    q: np.ndarray = np.zeros((manipulator.data["nq"], 1))
    v: np.ndarray = np.zeros((manipulator.data["nq"], 1))
    tau_g: np.ndarray = np.zeros((manipulator.data["nq"], 1))
    M: np.ndarray = np.zeros((manipulator.data["nq"], manipulator.data["nq"]))
    J_translational: np.ndarray = np.zeros((3, manipulator.data["nq"]))
    J_rotational: np.ndarray = np.zeros((3, manipulator.data["nq"]))
    J: np.ndarray = np.zeros((6, manipulator.data["nq"]))
    Cv: np.ndarray = np.zeros((manipulator.data["nq"], 1))
    Jdot_qdot: np.ndarray = np.zeros((manipulator.data["nq"], 1))


def stribeck(us, uk, v):
    '''
    Python version of
    `MultibodyPlant::StribeckModel::ComputeFrictionCoefficient`

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

    def __init__(self, manipulator_acc_log, ll_idx, contact_body_idx, \
            options, sys_params, jnt_frc_log):
        pydrake.systems.framework.LeafSystem.__init__(self)

        # Set up plant for evaluation
        self.manipulator_plant = MultibodyPlant(constants.DT)
        manipulator.data["add_plant_function"](self.manipulator_plant)
        self.manipulator_plant.Finalize()
        self.manipulator_plant_context = \
            self.manipulator_plant.CreateDefaultContext()

        # ========================= SYSTEM PARAMETERS =========================
        # Physical parameters
        self.v_stiction = sys_params['v_stiction']
        self.I_L = sys_params['I_L']
        self.w_L = sys_params['w_L']
        self.h_L = paper.PAPER_HEIGHT
        self.m_L = sys_params['m_L']
        self.b_J = sys_params['b_J']
        self.k_J = sys_params['k_J']
        self.g = sys_params['g']
        mu_paper = constants.FRICTION
        self.mu = 2*mu_paper/(1+mu_paper)
        self.r = manipulator.RADIUS

        # Other parameters
        self.ll_idx = ll_idx
        self.contact_body_idx = contact_body_idx
        self.nq_manipulator = \
            self.manipulator_plant.get_actuation_input_port().size()
        self.d_N_thresh = -5e-4

        # ========================== CONTROLLER INIT ==========================
        # Control targets
        self.d_Td = -0.03 #-0.12
        self.v_LNd = 0.1
        self.d_Xd = 0
        self.pre_contact_v_MNd = 0.1
        self.theta_MYd = None
        self.theta_MZd = None

        # Gains
        self.K_centering = 1
        self.D_centering = 0.1
        self.pre_contact_Kp = 10

        # Intermediary variables
        self.last_v_LN = 0
        self.init_q = None
        self.t_contact_start =  None
        self.v_LN_integrator = 0
        self.joint_centering_torque = None

        # ============================== LOG INIT =============================
        # Set up logs
        self.manipulator_acc_log = manipulator_acc_log
        # %DEBUG_APPEND%
        self.debug = defaultdict(list)
        self.debug['times'] = []
        self.jnt_frc_log = jnt_frc_log
        self.jnt_frc_log.append(
            SpatialForce(np.zeros((3, 1)), np.zeros((3, 1))))
        self.contacts = []

        # Set up dataclasses
        self.vision_derived_data = VisionDerivedData()
        self.cheat_ports_data = CheatPortsData()
        self.manipulator_data = ManipulatorData()

        # ============================== DRAKE IO =============================
        # Input ports
        self.DeclareVectorInputPort("q", BasicVector(self.nq_manipulator*2))
        self.DeclareAbstractInputPort(
            "poses", pydrake.common.value.AbstractValue.Make(
                [RigidTransform(), RigidTransform()]))
        self.DeclareAbstractInputPort(
            "vels", pydrake.common.value.AbstractValue.Make(
                [SpatialVelocity(), SpatialVelocity()]))
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


    def update_manipulator_data(self, state):
        q = state[:self.nq_manipulator]
        v = state[self.nq_manipulator:]
        self.manipulator_plant.SetPositions(self.manipulator_plant_context, q)
        self.manipulator_plant.SetVelocities(self.manipulator_plant_context, v)

        M = self.manipulator_plant.CalcMassMatrixViaInverseDynamics(self.manipulator_plant_context)
        Cv = np.expand_dims(self.manipulator_plant.CalcBiasTerm(self.manipulator_plant_context), 1)

        tau_g_raw = self.manipulator_plant.CalcGravityGeneralizedForces(
            self.manipulator_plant_context)
        tau_g = np.expand_dims(tau_g_raw, 1)

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

        Jdot_qdot = np.expand_dims(np.array(list(Jdot_qdot_raw.rotational()) \
            + list(Jdot_qdot_raw.translational())), 1)

        J_rotational = J[:3,:]
        J_translational = J[3:,:]

        self.manipulator_data.q = q
        self.manipulator_data.v = v
        self.manipulator_data.tau_g = tau_g
        self.manipulator_data.M = M
        self.manipulator_data.Cv = Cv
        self.manipulator_data.J = J
        self.manipulator_data.Jdot_qdot = Jdot_qdot
        self.manipulator_data.J_translational = J_translational
        self.manipulator_data.J_rotational = J_rotational


    def update_vision_derived_data(self, pose_L, vel_L, pose_M, vel_M):
        # Load positions
        p_L = np.array([pose_L.translation()[0:3]]).T
        p_M = np.array([pose_M.translation()[0:3]]).T
        
        R = pose_L.rotation()

        rot_vec_L = RollPitchYaw(pose_L.rotation()).vector()
        theta_L = rot_vec_L[0]

        rot_vec_M = RollPitchYaw(pose_M.rotation()).vector()
        theta_MX = rot_vec_M[0]
        theta_MY = rot_vec_M[1]
        theta_MZ = rot_vec_M[2]

        # Load velocities
        v_L = np.array([vel_L.translational()[0:3]]).T
        v_M = np.array([vel_M.translational()[0:3]]).T

        d_theta_L = vel_L.rotational()[0]
        omega_vec_L = np.array([[d_theta_L, 0, 0]]).T

        d_theta_MX = vel_M.rotational()[0]
        d_theta_MY = vel_M.rotational()[1]
        d_theta_MZ = vel_M.rotational()[2]
        omega_vec_M = vel_M.rotational()

        # Define unit vectors
        y_hat = np.array([[0, 1, 0]]).T
        z_hat = np.array([[0, 0, 1]]).T
        T_hat = R@y_hat
        N_hat = R@z_hat

        # Projection
        T_proj_mat = T_hat@(T_hat.T)
        N_proj_mat = N_hat@(N_hat.T)
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

        # Get components
        p_LT = get_T_proj(p_L)
        p_LN = get_N_proj(p_L)

        v_LN = get_N_proj(v_L)
        v_LT = get_T_proj(v_L)

        v_MN = get_N_proj(v_M)
        v_MT = get_T_proj(v_M)

        # Derived terms
        p_LLE = N_hat * -self.h_L/2 + T_hat * self.w_L/2
        p_LE = p_L + p_LLE

        p_C = p_M + N_hat*self.r
        p_CT = self.get_T_proj(p_C)
        p_CN = self.get_N_proj(p_C)

        d = p_C - p_LE
        d_T = self.get_T_proj(d)
        d_N = self.get_N_proj(d)
        d_X = d[0]

        p_MConM = p_C - p_M
        p_LConL = p_C - p_L

        # Velocities
        v_WConL = v_L + np.cross(omega_vec_L, p_LConL, axis=0)
        v_WConM = v_M + np.cross(omega_vec_M, p_MConM, axis=0)

        d_d_T = -d_theta_L*self.h_L/2 - d_theta_L*self.r - v_LT + v_MT \
            + d_theta_L*d_N
        d_d_N = -d_theta_L*self.w_L/2-v_LN+v_MN-d_theta_L*d_T
        d_d_X = v_WConM[0]

        v_S_raw = v_WConM - v_WConL
        v_S_N = np.matmul(N_proj_mat, v_S_raw)
        v_S = v_S_raw - v_S_N

        s_hat = v_S/np.linalg.norm(v_S)
        hats_T = self.get_T_proj(s_hat)
        s_hat_X = s_hat[0]
        s_S = np.linalg.norm(v_S)
    
        # Gravity
        F_G = np.array([[0, 0, -self.m_L*constants.g]]).T
        F_GT = self.get_T_proj(F_G)
        F_GN = self.get_N_proj(F_G)

        stribeck_mu = stribeck(1, 1, s_S/self.v_stiction)
        mu_S = stribeck_mu

        # Calculate whether we are in contact
        raw_in_contact = d_N > self.d_N_thresh
        if raw_in_contact:
            if self.t_contact_start is None:
                self.t_contact_start = self.debug['times'][-1]
        else:
            self.t_contact_start =  None
        in_contact = raw_in_contact and self.debug['times'][-1] \
            - self.t_contact_start > 0.5

        self.vision_derived_data.T_hat = T_hat
        self.vision_derived_data.N_hat = N_hat
        self.vision_derived_data.theta_L = theta_L
        self.vision_derived_data.d_theta_L = d_theta_L
        self.vision_derived_data.pose_L = pose_L
        self.vision_derived_data.p_LN = p_LN
        self.vision_derived_data.p_LT = p_LT
        self.vision_derived_data.pose_M = pose_M
        self.vision_derived_data.v_LN = v_LN
        self.vision_derived_data.v_MN = v_MN
        self.vision_derived_data.theta_MX = theta_MX
        self.vision_derived_data.theta_MY = theta_MY
        self.vision_derived_data.theta_MZ = theta_MZ
        self.vision_derived_data.d_theta_MX = d_theta_MX
        self.vision_derived_data.d_theta_MY = d_theta_MY
        self.vision_derived_data.d_theta_MZ = d_theta_MZ
        self.vision_derived_data.F_GT = F_GT
        self.vision_derived_data.F_GN = F_GN
        self.vision_derived_data.d_X = d_X
        self.vision_derived_data.d_d_X = d_d_X
        self.vision_derived_data.d_N = d_N
        self.vision_derived_data.d_d_N = d_d_N
        self.vision_derived_data.d_T = d_T
        self.vision_derived_data.d_d_T = d_d_T
        self.vision_derived_data.p_CN = p_CN
        self.vision_derived_data.p_CT = p_CT
        self.vision_derived_data.p_MConM = p_MConM
        self.vision_derived_data.mu_S = mu_S
        self.vision_derived_data.hats_T = hats_T
        self.vision_derived_data.s_hat_X = s_hat_X
        self.vision_derived_data.in_contact = in_contact

    def update_cheat_ports_data(self):
        jnt_frcs = self.jnt_frc_log[-1]
        self.cheat_ports_data.F_OT = jnt_frcs.translational()[1]
        self.cheat_ports_data.F_ON = jnt_frcs.translational()[2]
        self.cheat_ports_data.tau_O = jnt_frcs.rotational()[0]

    def update_debug(self):
        for k in get_attrs_from_class(VisionDerivedData):
            self.debug[k].append(getattr(self.vision_derived_data, k))
        for k in get_attrs_from_class(CheatPortsData):
            self.debug[k].append(getattr(self.cheat_ports_data, k))
        for k in get_attrs_from_class(ManipulatorData):
            self.debug[k].append(getattr(self.manipulator_data, k))

    def simulate_vision(self, poses, vels):
        pose_L = poses[self.ll_idx]
        vel_L = vels[self.ll_idx]

        pose_M = poses[self.contact_body_idx]
        vel_M = vels[self.contact_body_idx]

        return pose_L, vel_L, pose_M, vel_M

    def get_contact_control_torques(self):
        raise NotImplementedError

    def non_contact_debug_update(self):
        raise NotImplementedError

    def CalcOutput(self, context, output):
        ## Load inputs
        # This input put is already restricted to the manipulator, but it
        # includes both q and v
        state = self.get_input_port(0).Eval(context)
        poses = self.get_input_port(1).Eval(context)
        vels = self.get_input_port(2).Eval(context)
        contact_results = self.get_input_port(3).Eval(context)

        # Get time
        # %DEBUG_APPEND%
        self.debug['times'].append(context.get_time())

        # Process state
        self.update_manipulator_data(state)

        # Book keeping
        if self.init_q is None:
            self.init_q = self.manipulator_data.q

        # Precompute other inputs
        pose_L, vel_L, pose_M, vel_M = self.simulate_vision(poses, vels)
        self.update_vision_derived_data(pose_L, vel_L, pose_M, vel_M)
        self.update_cheat_ports_data()

        if self.theta_MYd is None:
            self.theta_MYd = self.vision_derived_data.theta_MY
        if self.theta_MZd is None:
            self.theta_MZd = self.vision_derived_data.theta_MZ

        # Get torques
        J_plus = np.linalg.pinv(self.manipulator_data.J)
        nullspace_basis = np.eye(self.nq_manipulator) \
            - np.matmul(J_plus, self.manipulator_data.J)

        # Add a PD controller projected into the nullspace of the Jacobian that
        # keeps us close to the nominal configuration
        self.joint_centering_torque = np.expand_dims(np.matmul(
            nullspace_basis,
            self.K_centering*(self.init_q - self.manipulator_data.q) \
                + self.D_centering*(-self.manipulator_data.v)), 1)

        if self.vision_derived_data.in_contact:
            tau_ctrl = self.get_contact_control_torques()
            assert tau_ctrl.shape == (self.nq_manipulator, 1)
        else:
            tau_ctrl = self.get_pre_contact_control_torques()

        if not self.vision_derived_data.in_contact:
            self.non_contact_debug_update()

        tau_out = tau_ctrl - self.manipulator_data.tau_g \
            + self.joint_centering_torque
        output.SetFromVector(tau_out)

        self.update_debug()

        # Debug
        # %DEBUG_APPEND%
        self.debug['tau_ctrl'].append(tau_ctrl)
        self.debug['tau_out'].append(tau_out)
        self.debug['v_LNds'].append(self.v_LNd)
        self.debug["joint_centering_torque"].append(
            self.joint_centering_torque)

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
                        length=0.15, radius=0.006,
                        X_PT=self.vision_derived_data.pose_M)
            AddMeshcatTriad(self.meshcat, "axes/" + "pose_L",
                        length=0.15, radius=0.006,
                        X_PT=self.vision_derived_data.pose_L)


    def get_pre_contact_control_torques(self):
        # TODO: add controller for hitting correct orientation
        # Proportional control for moving towards the link
        v_MNd = self.pre_contact_v_MNd
        Kp = self.pre_contact_Kp
        F_CN = (v_MNd - self.vision_derived_data.v_MN)*Kp

        F_d = np.zeros((6,1))
        if self.debug['times'][-1] > paper.settling_time:
            F_d[3:] += self.vision_derived_data.N_hat * F_CN

        tau_ctrl = self.manipulator_data.J.T@F_d
        return tau_ctrl



class FoldingPositionController:
    def __init__(self):
        pass

    def get_contact_control_torques(self):
        pass

    def non_contact_debug_update(self):
        pass


class FoldingImpedanceController:
    def __init__(self):
        pass

    def get_contact_control_torques(self):
        pass

    def non_contact_debug_update(self):
        pass


class FoldingInverseDynamicsController(FoldingController):
    def __init__(self, manipulator_acc_log, ll_idx, contact_body_idx, \
            options, sys_params, jnt_frc_log):
        super().__init__(manipulator_acc_log, ll_idx, contact_body_idx, \
            options, sys_params, jnt_frc_log)

        # Options
        self.model_friction = options['model_friction']
        self.measure_joint_wrench = options['measure_joint_wrench']

    def get_contact_control_torques(self):
        assert self.manipulator_data.tau_g.shape == (self.nq_manipulator, 1)
        assert self.joint_centering_torque.shape == (self.nq_manipulator, 1)

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
            tau_O = -self.k_J*self.vision_derived_data.theta_L \
                    - self.b_J*self.vision_derived_data.d_theta_L

        # Control values
        tau_ctrl = prog.NewContinuousVariables(
            self.nq_manipulator, 1, name="tau_ctrl")

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

        ddq = prog.NewContinuousVariables(self.nq_manipulator, 1, name="ddq")

        ## Constraints
        # "set_description" calls gives us useful names for printing
        prog.AddConstraint(
            eq(self.m_L*a_LT, F_FLT+self.vision_derived_data.F_GT+F_OT)).evaluator().set_description(
                "Link tangential force balance")
        prog.AddConstraint(eq(self.m_L*a_LN, F_NL + self.vision_derived_data.F_GN + F_ON)).evaluator().set_description(
                "Link normal force balance") 
        prog.AddConstraint(eq(self.I_L*dd_theta_L, \
            (-self.w_L/2)*F_ON - (self.vision_derived_data.p_CN-self.vision_derived_data.p_LN) * F_FLT + (self.vision_derived_data.p_CT-self.vision_derived_data.p_LT)*F_NL + tau_O)).evaluator().set_description(
                "Link moment balance") 
        prog.AddConstraint(eq(F_NL, -F_NM)).evaluator().set_description(
                "3rd law normal forces")
        if self.model_friction:
            prog.AddConstraint(eq(F_FMT, -F_FLT)).evaluator().set_description(
                    "3rd law friction forces (T hat)") 
            prog.AddConstraint(eq(F_FMX, -F_FLX)).evaluator().set_description(
                    "3rd law friction forces (X hat)") 
        prog.AddConstraint(eq(
            -dd_theta_L*(self.h_L/2+self.r) + self.vision_derived_data.d_theta_L**2*self.w_L/2 - a_LT + a_MT,
            -dd_theta_L*self.vision_derived_data.d_N + dd_d_T - self.vision_derived_data.d_theta_L**2*self.vision_derived_data.d_T - 2*self.vision_derived_data.d_theta_L*self.vision_derived_data.d_d_N
        )).evaluator().set_description("d_N derivative is derivative")
        prog.AddConstraint(eq(
            -dd_theta_L*self.w_L/2 - self.vision_derived_data.d_theta_L**2*self.h_L/2 - self.vision_derived_data.d_theta_L**2*self.r - a_LN + a_MN,
            dd_theta_L*self.vision_derived_data.d_T + dd_d_N - self.vision_derived_data.d_theta_L**2*self.vision_derived_data.d_N + 2*self.vision_derived_data.d_theta_L*self.vision_derived_data.d_d_T
        )).evaluator().set_description("d_N derivative is derivative") 
        prog.AddConstraint(eq(dd_d_N, 0)).evaluator().set_description("No penetration")
        if self.model_friction:
            prog.AddConstraint(eq(F_FLT, self.vision_derived_data.mu_S*F_NL*self.mu*self.vision_derived_data.hats_T)).evaluator().set_description(
                "Friction relationship LT")
            prog.AddConstraint(eq(F_FLX, self.vision_derived_data.mu_S*F_NL*self.mu*self.vision_derived_data.s_hat_X)).evaluator().set_description(
                "Friction relationship LX")
        
        if not self.measure_joint_wrench:
            prog.AddConstraint(eq(a_LT, -(self.w_L/2)*self.vision_derived_data.d_theta_L**2)).evaluator().set_description(
                "Hinge constraint (T hat)")
            prog.AddConstraint(eq(a_LN, (self.w_L/2)*dd_theta_L)).evaluator().set_description(
                "Hinge constraint (N hat)")
        
        for i in range(6):
            lhs_i = alpha_a_MXYZ[i,0]
            assert not hasattr(lhs_i, "shape")
            rhs_i = (self.manipulator_data.Jdot_qdot + np.matmul(self.manipulator_data.J, ddq))[i,0]
            assert not hasattr(rhs_i, "shape")
            prog.AddConstraint(lhs_i == rhs_i).evaluator().set_description(
                "Relate manipulator and end effector with joint accelerations " + str(i)) 

        tau_contact_trn = np.matmul(
            self.manipulator_data.J_translational.T, F_ContactM_XYZ)
        tau_contact_rot = np.matmul(
            self.manipulator_data.J_rotational.T, np.cross(self.vision_derived_data.p_MConM, F_ContactM_XYZ, axis=0))
        tau_contact = tau_contact_trn + tau_contact_rot
        tau_out = tau_ctrl - self.manipulator_data.tau_g + self.joint_centering_torque
        for i in range(self.nq_manipulator):
            M_ddq_row_i = (np.matmul(self.manipulator_data.M, ddq) + self.manipulator_data.Cv)[i,0]
            assert not hasattr(M_ddq_row_i, "shape")
            tau_i = (self.manipulator_data.tau_g + tau_contact + tau_out)[i,0]
            assert not hasattr(tau_i, "shape")
            prog.AddConstraint(M_ddq_row_i == tau_i).evaluator().set_description("Manipulator equations " + str(i))
        
        # Projection equations
        prog.AddConstraint(eq(a_MT, np.cos(self.vision_derived_data.theta_L)*a_MY + np.sin(self.vision_derived_data.theta_L)*a_MZ))
        prog.AddConstraint(eq(a_MN, -np.sin(self.vision_derived_data.theta_L)*a_MY + np.cos(self.vision_derived_data.theta_L)*a_MZ))
        prog.AddConstraint(eq(F_FMT, np.cos(self.vision_derived_data.theta_L)*F_ContactMY + np.sin(self.vision_derived_data.theta_L)*F_ContactMZ))
        prog.AddConstraint(eq(F_NM, -np.sin(self.vision_derived_data.theta_L)*F_ContactMY + np.cos(self.vision_derived_data.theta_L)*F_ContactMZ))

        # Calculate desired values
        dd_d_Td = 1000*(self.d_Td - self.vision_derived_data.d_T) - 100*self.vision_derived_data.d_d_T
        a_LNd = 10*(self.v_LNd - self.vision_derived_data.v_LN)
        a_MX_d = 100*(self.d_Xd - self.vision_derived_data.d_X) - 10 * self.vision_derived_data.d_d_X
        theta_MXd = self.vision_derived_data.theta_L
        alpha_MXd = 100*(theta_MXd - self.vision_derived_data.theta_MX)  + 10*(self.vision_derived_data.d_theta_L - self.vision_derived_data.d_theta_MX)
        alpha_MYd = 10*(self.theta_MYd - self.vision_derived_data.theta_MY) - 5*self.vision_derived_data.d_theta_MY
        alpha_MZd = 10*(self.theta_MZd - self.vision_derived_data.theta_MZ) - 5*self.vision_derived_data.d_theta_MZ
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
        tau_ctrl_result = np.expand_dims(tau_ctrl_result, 1)

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
        if self.model_friction:
            self.debug["F_FMT"].append(result.GetSolution()[prog.FindDecisionVariableIndex(F_FMT[0,0])])
            self.debug["F_FMX"].append(result.GetSolution()[prog.FindDecisionVariableIndex(F_FMX[0,0])])
            self.debug["F_FLT"].append(result.GetSolution()[prog.FindDecisionVariableIndex(F_FLT[0,0])])
            self.debug["F_FLX"].append(result.GetSolution()[prog.FindDecisionVariableIndex(F_FLX[0,0])])
        else:
            self.debug["F_FMT"].append(F_FMT)
            self.debug["F_FMX"].append(F_FMX)
            self.debug["F_FLT"].append(F_FLT)
            self.debug["F_FLX"].append(F_FLX)
        self.debug["F_NM"].append(result.GetSolution()[prog.FindDecisionVariableIndex(F_NM[0,0])])
        self.debug["F_ContactMY"].append(result.GetSolution()[prog.FindDecisionVariableIndex(F_ContactMY[0,0])])
        self.debug["F_ContactMZ"].append(result.GetSolution()[prog.FindDecisionVariableIndex(F_ContactMZ[0,0])])
        self.debug["F_NL"].append(result.GetSolution()[prog.FindDecisionVariableIndex(F_NL[0,0])])
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

    def non_contact_debug_update(self):
        # %DEBUG_APPEND%
        self.debug["dd_d_Td"].append(np.nan)
        self.debug["a_LNd"].append(np.nan)
        self.debug["a_MX_d"].append(np.nan)
        self.debug["alpha_MXd"].append(np.nan)
        self.debug["alpha_MYd"].append(np.nan)
        self.debug["alpha_MZd"].append(np.nan)
        self.debug["dd_d_Nd"].append(np.nan)
        self.debug["F_FMT"].append(np.nan)
        self.debug["F_FMX"].append(np.nan)
        self.debug["F_FLT"].append(np.nan)
        self.debug["F_FLX"].append(np.nan)
        self.debug["F_NM"].append(np.nan)
        self.debug["F_ContactMY"].append(np.nan)
        self.debug["F_ContactMZ"].append(np.nan)
        self.debug["F_NL"].append(np.nan)
        for i in range(self.nq_manipulator):
            self.debug["tau_ctrl_" + str(i)].append(np.nan)
        self.debug["a_MX"].append(np.nan)
        self.debug["a_MT"].append(np.nan)
        self.debug["a_MY"].append(np.nan)
        self.debug["a_MZ"].append(np.nan)
        self.debug["a_MN"].append(np.nan)
        self.debug["a_LT"].append(np.nan)
        self.debug["a_LN"].append(np.nan)
        self.debug["alpha_MX"].append(np.nan)
        self.debug["alpha_MY"].append(np.nan)
        self.debug["alpha_MZ"].append(np.nan)
        self.debug["dd_theta_L"].append(np.nan)
        self.debug["dd_d_N"].append(np.nan)
        self.debug["dd_d_T"].append(np.nan)
        for i in range(self.nq_manipulator):
            self.debug["ddq_" + str(i)].append(np.nan)
        self.debug["theta_MXd"].append(np.nan)


class FoldingSimpleController(FoldingController):
    def __init__(self, manipulator_acc_log, ll_idx, contact_body_idx, \
            options, sys_params, jnt_frc_log):
        super().__init__(manipulator_acc_log, ll_idx, contact_body_idx, \
            options, sys_params, jnt_frc_log)

    def get_contact_control_torques(self):
        if len(self.debug["times"]) > 2:
            dt = self.debug["times"][-1] - self.debug["times"][-2]
        else:
            dt = 0
        self.v_LN_integrator += dt*(self.v_LNd - self.vision_derived_data.v_LN)

        F_ON_approx = -(self.k_J*self.vision_derived_data.theta_L - self.b_J*self.vision_derived_data.d_theta_L)/(self.w_L/2)
        ff_term = -self.vision_derived_data.F_GN - F_ON_approx
        
        F_CT = 1000*(self.d_Td - self.vision_derived_data.d_T) - 100*self.vision_derived_data.d_d_T
        F_CN = 10*(self.v_LNd - self.vision_derived_data.v_LN) + 0*self.v_LN_integrator + ff_term
        F_CX = 100*(self.d_Xd - self.vision_derived_data.d_X) - 10 * self.vision_derived_data.d_d_X
        theta_MXd = self.vision_derived_data.theta_L
        tau_X = 100*(theta_MXd - self.vision_derived_data.theta_MX)  + 10*(self.vision_derived_data.d_theta_L - self.vision_derived_data.d_theta_MX)
        tau_Y = 10*(self.theta_MYd - self.vision_derived_data.theta_MY) - 5*self.vision_derived_data.d_theta_MY
        tau_Z = 10*(self.theta_MZd - self.vision_derived_data.theta_MZ) - 5*self.vision_derived_data.d_theta_MZ

        F = F_CT*self.vision_derived_data.T_hat + F_CN*self.vision_derived_data.N_hat + F_CX * np.array([[1, 0, 0]]).T
        tau = np.array([[tau_X, tau_Y, tau_Z]]).T

        tau_ctrl = np.matmul(self.manipulator_data.J_translational.T, F) \
            + np.matmul(self.manipulator_data.J_rotational.T, tau)
        return tau_ctrl

    def non_contact_debug_update(self):
        pass
