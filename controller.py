"""Functions for creating and controlling the finger manipulator."""
# General imports
import numpy as np
from collections import defaultdict
from dataclasses import dataclass

# Local imports
import constants
import plant.manipulator as manipulator
import plant.paper as paper

import enum

# Drake imports
import pydrake
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.all import (
    MathematicalProgram, Solve, eq, \
    BasicVector, MultibodyPlant, ContactResults, SpatialVelocity, \
    SpatialForce, RigidTransform, RollPitchYaw, RotationMatrix \
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


vision_derived_data_keys = { "T_hat", "N_hat", "theta_L", "d_theta_L", "p_LN", "p_LT", "v_LN", "v_MN", "theta_MX", "theta_MY", "theta_MZ", "d_theta_MX", "d_theta_MY", "d_theta_MZ", "F_GT", "F_GN", "d_X", "d_d_X", "d_N", "d_d_N", "d_T", "d_d_T", "p_CN", "p_CT", "p_MConM", "mu_S", "hats_T", "s_hat_X", "in_contact" }

@dataclass
class VisionDerivedData:
    """
    Data derived from vision information.
    """

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

    def __init__(self, ll_idx, contact_body_idx, \
            options, sys_params):
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
        self.m_M = sys_params['m_M']
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
        self.d_Td = -self.w_L/2
        self.d_theta_Ld = 0.5
        self.d_Xd = 0
        self.pre_contact_v_MNd = 0.1
        self.theta_MYd = None
        self.theta_MZd = None

        # Gains
        self.K_centering = 1
        self.D_centering = 0.1
        self.pre_contact_Kp = 10

        # Intermediary variables
        self.init_q = None
        self.t_contact_start =  None
        self.d_theta_L_integrator = 0
        self.joint_centering_torque = None

        # ============================== LOG INIT =============================
        # Set up logs
        # %DEBUG_APPEND%
        self.debug = defaultdict(list)
        self.debug['times'] = []
        self.contacts = []

        # Set up dataclasses
        self.vision_derived_data = VisionDerivedData()
        self.manipulator_data = ManipulatorData()

        # ============================== DRAKE IO =============================
        # Input ports
        self.DeclareVectorInputPort("q_", BasicVector(self.nq_manipulator*2))
        self.DeclareAbstractInputPort(
            "poses", pydrake.common.value.AbstractValue.Make(
                [RigidTransform(), RigidTransform()]))
        self.DeclareAbstractInputPort(
            "vels", pydrake.common.value.AbstractValue.Make(
                [SpatialVelocity(), SpatialVelocity()]))
        self.DeclareAbstractInputPort(
            "contact_results",
            pydrake.common.value.AbstractValue.Make(ContactResults()))

        self.DeclareVectorInputPort(
            "pose_L_translational", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "pose_L_rotational", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "vel_L_translational", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "vel_L_rotational", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "pose_M_translational", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "pose_M_rotational", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "vel_M_translational", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "vel_M_rotational", pydrake.systems.framework.BasicVector(3))

        self.DeclareVectorInputPort(
            "T_hat", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "N_hat", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "theta_L", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "d_theta_L", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "p_LN", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "p_LT", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "v_LN", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "v_MN", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "theta_MX", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "theta_MY", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "theta_MZ", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "d_theta_MX", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "d_theta_MY", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "d_theta_MZ", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "F_GT", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "F_GN", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "d_X", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "d_d_X", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "d_N", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "d_d_N", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "d_T", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "d_d_T", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "p_CN", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "p_CT", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "p_MConM", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "mu_S", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "hats_T", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "s_hat_X", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "in_contact", pydrake.systems.framework.BasicVector(1))

        self.DeclareVectorInputPort(
            "q", pydrake.systems.framework.BasicVector(self.nq_manipulator))
        self.DeclareVectorInputPort(
            "v", pydrake.systems.framework.BasicVector(self.nq_manipulator))
        self.DeclareVectorInputPort(
            "tau_g",
            pydrake.systems.framework.BasicVector(self.nq_manipulator))
        self.DeclareVectorInputPort(
            "M",
            pydrake.systems.framework.BasicVector(
                self.nq_manipulator*self.nq_manipulator))
        self.DeclareVectorInputPort(
            "Cv",
            pydrake.systems.framework.BasicVector(self.nq_manipulator))
        self.DeclareVectorInputPort(
            "J",
            pydrake.systems.framework.BasicVector(6*self.nq_manipulator))
        self.DeclareVectorInputPort(
            "J_translational",
            pydrake.systems.framework.BasicVector(3*self.nq_manipulator))
        self.DeclareVectorInputPort(
            "J_rotational",
            pydrake.systems.framework.BasicVector(3*self.nq_manipulator))
        self.DeclareVectorInputPort(
            "Jdot_qdot",
            pydrake.systems.framework.BasicVector(6))

        # Output ports
        self.DeclareVectorOutputPort(
            "actuation", pydrake.systems.framework.BasicVector(
                self.nq_manipulator),
            self.CalcOutput)


    def set_meshcat(self, meshcat):
        self.meshcat = meshcat


    def update_manipulator_data(self, state, context):
        q = np.array(self.GetInputPort("q").Eval(context))
        v = np.array(self.GetInputPort("v").Eval(context))
        self.manipulator_plant.SetPositions(self.manipulator_plant_context, q)
        self.manipulator_plant.SetVelocities(self.manipulator_plant_context, v)

        M = np.array(self.GetInputPort("M").Eval(context)).reshape(
            (self.nq_manipulator, self.nq_manipulator))
        Cv = np.expand_dims(
            np.array(self.GetInputPort("Cv").Eval(context)), 1)

        tau_g = np.expand_dims(
            np.array(self.GetInputPort("tau_g").Eval(context)), 1)

        J = np.array(self.GetInputPort("J").Eval(context)).reshape(
            (6, self.nq_manipulator))

        J_rotational = np.array(
            self.GetInputPort("J_rotational").Eval(context)).reshape(
                (3, self.nq_manipulator))

        J_translational = np.array(
            self.GetInputPort("J_translational").Eval(context)).reshape(
                (3, self.nq_manipulator))

        Jdot_qdot = np.expand_dims(
            np.array(self.GetInputPort("Jdot_qdot").Eval(context)), 1)

        self.manipulator_data.q = q
        self.manipulator_data.v = v
        self.manipulator_data.tau_g = tau_g
        self.manipulator_data.M = M
        self.manipulator_data.Cv = Cv
        self.manipulator_data.J = J
        self.manipulator_data.Jdot_qdot = Jdot_qdot
        self.manipulator_data.J_translational = J_translational
        self.manipulator_data.J_rotational = J_rotational

    def update_vision_derived_data(self, context):
        p_L = np.array(
            [self.GetInputPort("pose_L_translational").Eval(context)[0:3]]).T
        p_M = np.array(
            [self.GetInputPort("pose_M_translational").Eval(context)[0:3]]).T
        
        rot_vec_L = np.array(
            [self.GetInputPort("pose_L_rotational").Eval(context)[0:3]]).T
        rot_vec_M = np.array(
            [self.GetInputPort("pose_M_rotational").Eval(context)[0:3]]).T

        v_L = np.array(
            [self.GetInputPort("vel_L_translational").Eval(context)[0:3]]).T
        v_M = np.array(
            [self.GetInputPort("vel_M_translational").Eval(context)[0:3]]).T

        omega_vec_L = np.array(
            [self.GetInputPort("vel_L_rotational").Eval(context)[0:3]]).T
        omega_vec_M = np.array(
            [self.GetInputPort("vel_M_rotational").Eval(context)[0:3]]).T


        theta_L = self.GetInputPort("theta_L").Eval(context)[0]
        theta_MX = self.GetInputPort("theta_MX").Eval(context)[0]
        theta_MY = self.GetInputPort("theta_MY").Eval(context)[0]
        theta_MZ = self.GetInputPort("theta_MZ").Eval(context)[0]

        d_theta_L = self.GetInputPort("d_theta_L").Eval(context)[0]

        d_theta_MX = self.GetInputPort("d_theta_MX").Eval(context)[0]
        d_theta_MY = self.GetInputPort("d_theta_MY").Eval(context)[0]
        d_theta_MZ = self.GetInputPort("d_theta_MZ").Eval(context)[0]

        # Define unit vectors
        T_hat = np.array([self.GetInputPort("T_hat").Eval(context)]).T
        N_hat = np.array([self.GetInputPort("N_hat").Eval(context)]).T

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
        p_LT = self.GetInputPort("p_LT").Eval(context)[0]
        p_LN = self.GetInputPort("p_LN").Eval(context)[0]

        v_LN = self.GetInputPort("p_LN").Eval(context)[0]
        v_LT = self.GetInputPort("p_LT").Eval(context)[0]

        v_MN = self.GetInputPort("v_MN").Eval(context)[0]

        p_CT = self.GetInputPort("p_CT").Eval(context)[0]
        p_CN = self.GetInputPort("p_CN").Eval(context)[0]

        d_T = self.GetInputPort("d_T").Eval(context)[0]
        d_N = self.GetInputPort("d_N").Eval(context)[0]
        d_X = self.GetInputPort("d_X").Eval(context)[0]

        p_MConM = np.array([self.GetInputPort("p_MConM").Eval(context)]).T

        d_d_T = self.GetInputPort("d_d_T").Eval(context)[0]
        d_d_N = self.GetInputPort("d_d_N").Eval(context)[0]
        d_d_X = self.GetInputPort("d_d_X").Eval(context)[0]

        hats_T = self.GetInputPort("hats_T").Eval(context)[0]
        s_hat_X = self.GetInputPort("s_hat_X").Eval(context)[0]
    
        # Gravity
        F_GT = self.GetInputPort("F_GT").Eval(context)[0]
        F_GN = self.GetInputPort("F_GN").Eval(context)[0]

        stribeck_mu = self.GetInputPort("mu_S").Eval(context)[0]
        mu_S = stribeck_mu

        # Calculate whether we are in contact
        in_contact = self.GetInputPort("in_contact").Eval(context)[0]

        self.vision_derived_data.T_hat = T_hat
        self.vision_derived_data.N_hat = N_hat
        self.vision_derived_data.theta_L = theta_L
        self.vision_derived_data.d_theta_L = d_theta_L
        self.vision_derived_data.p_LN = p_LN
        self.vision_derived_data.p_LT = p_LT
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

    def update_debug(self):
        for k in vision_derived_data_keys:
            self.debug[k].append(getattr(self.vision_derived_data, k))
        for k in get_attrs_from_class(ManipulatorData):
            self.debug[k].append(getattr(self.manipulator_data, k))

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
        self.update_manipulator_data(state, context)

        # Book keeping
        if self.init_q is None:
            self.init_q = self.manipulator_data.q

        # Precompute other inputs
        self.update_vision_derived_data(context)

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
        self.debug['d_theta_Lds'].append(self.d_theta_Ld)
        self.debug["joint_centering_torque"].append(
            self.joint_centering_torque)

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
    def __init__(self, ll_idx, contact_body_idx, options, sys_params):
        super().__init__(ll_idx, contact_body_idx,  options, sys_params)

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
        dd_theta_Ld = 10*(self.d_theta_Ld - self.vision_derived_data.d_theta_L)
        a_MX_d = 100*(self.d_Xd - self.vision_derived_data.d_X) - 10 * self.vision_derived_data.d_d_X
        theta_MXd = self.vision_derived_data.theta_L
        alpha_MXd = 100*(theta_MXd - self.vision_derived_data.theta_MX)  + 10*(self.vision_derived_data.d_theta_L - self.vision_derived_data.d_theta_MX)
        alpha_MYd = 10*(self.theta_MYd - self.vision_derived_data.theta_MY) - 5*self.vision_derived_data.d_theta_MY
        alpha_MZd = 10*(self.theta_MZd - self.vision_derived_data.theta_MZ) - 5*self.vision_derived_data.d_theta_MZ
        dd_d_Nd = 0
        prog.AddConstraint(dd_d_T[0,0] == dd_d_Td).evaluator().set_description("Desired dd_d_Td constraint" + str(i))
        prog.AddConstraint(dd_theta_L[0,0] == dd_theta_Ld).evaluator().set_description("Desired a_LN constraint" + str(i))
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
        self.debug["dd_theta_Ld"].append(dd_theta_Ld)
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
        self.debug["dd_theta_Ld"].append(np.nan)
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
    def __init__(self, ll_idx, contact_body_idx, options, sys_params):
        super().__init__(ll_idx, contact_body_idx, options, sys_params)

    def get_contact_control_torques(self):
        if len(self.debug["times"]) > 2:
            dt = self.debug["times"][-1] - self.debug["times"][-2]
        else:
            dt = 0
        self.d_theta_L_integrator += dt*(self.d_theta_Ld - self.vision_derived_data.d_theta_L)

        tau_O_est = -self.k_J*self.vision_derived_data.theta_L - self.b_J*self.vision_derived_data.d_theta_L

        ff_term = -self.vision_derived_data.F_GN - tau_O_est/(self.w_L/2)
        
        F_CT = 1000*(self.d_Td - self.vision_derived_data.d_T) - 100*self.vision_derived_data.d_d_T
        F_CN = 10*(self.d_theta_Ld - self.vision_derived_data.d_theta_L) + ff_term
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
