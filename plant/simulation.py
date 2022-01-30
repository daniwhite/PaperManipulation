import pydrake
# TODO: pull these out
from pydrake.all import (
    BodyIndex, Adder, ConstantVectorSource, Multiplexer, Demultiplexer, Gain,
    LogVectorOutput, MeshcatVisualizerParams, MeshcatVisualizerCpp
)

import numpy as np

from dataclasses import dataclass
import enum
import re

import config

import plant.pedestal as pedestal
import plant.manipulator as manipulator
from plant.paper import Paper

import perception.vision
import perception.proprioception

import ctrl.kinematic_ctrlr
import ctrl.inverse_dynamics
import ctrl.kinematic_ctrlr
import ctrl.cartesian_impedance
import ctrl.impedance_generators.setpoint_generators.circle
import ctrl.impedance_generators.setpoint_generators.link_feedback

from log_wrapper import LogWrapper

import constants

import visualization


# "Dataclasses" (= structs)
@dataclass
class IndependentParams:
    w_L: float
    h_L: float
    m_L: float
    m_M: float
    b_J: float
    k_J: float
    mu: float
    r: float


# Enums
class CtrlParadigm(enum.Enum):
    INVERSE_DYNAMICS = enum.auto()
    KINEMATIC = enum.auto()
    IMPEDANCE = enum.auto()


class ImpedanceType(enum.Enum):
    OFFLINE_TRAJ = enum.auto()
    LINK_FB = enum.auto()
    NONE = enum.auto


class NHatForceCompensationSource(enum.Enum):
    MEASURED = enum.auto()
    CONSTANT = enum.auto()
    NONE = enum.auto()

class Simulation:
    def __init__(self,
            ctrl_paradigm: CtrlParadigm, impedance_type: ImpedanceType,
            n_hat_force_compensation_source: NHatForceCompensationSource,
            params=None, meshcat=None, impedance_stiffness=None):
        # Take in inputs
        if params is None:
            self.params = constants.nominal_sys_consts
        else: 
            self.params = params
        self.ctrl_paradigm = ctrl_paradigm
        self.impedance_type = impedance_type
        # TODO: This needs a better name
        self.n_hat_force_compensation_source = n_hat_force_compensation_source
        self.meshcat = meshcat
        self.impedance_stiffness = impedance_stiffness

        assert (impedance_type == ImpedanceType.NONE) or \
            (ctrl_paradigm == CtrlParadigm.IMPEDANCE)
        assert (impedance_type == NHatForceCompensationSource.NONE) or \
            (ctrl_paradigm == CtrlParadigm.IMPEDANCE)
        assert not((ctrl_paradigm == CtrlParadigm.IMPEDANCE) and \
            (impedance_type == ImpedanceType.NONE))

        # Create builder
        self.builder = pydrake.systems.framework.DiagramBuilder()

        # Create plant
        self.plant, self.scene_graph = \
            pydrake.multibody.plant.AddMultibodyPlantSceneGraph(
                self.builder, time_step=config.DT)
        self.plant.set_stiction_tolerance(constants.v_stiction)
        self.plant.set_penetration_allowance(0.001)

        self.add_mbp_bodies()

        # Init sys_consts
        self.sys_consts = constants.SystemConstants
        self.sys_consts.I_L = self.plant.get_body(
            BodyIndex(self.paper.link_idxs[-1])).default_rotational_inertia(
                ).CalcPrincipalMomentsOfInertia()[0]
        self.sys_consts.v_stiction = constants.v_stiction
        self.sys_consts.w_L = self.params.w_L
        self.sys_consts.h_L = self.params.h_L
        self.sys_consts.m_L = self.params.m_L
        self.sys_consts.m_M = self.params.m_M
        self.sys_consts.b_J = self.params.b_J
        self.sys_consts.k_J = self.params.k_J
        self.sys_consts.mu = self.params.mu
        self.sys_consts.r = self.params.r
        self.sys_consts.g = self.plant.gravity_field().gravity_vector()[-1]*-1


        contact_body = self.plant.GetBodyByName(
            manipulator.data["contact_body_name"])
        self.ll_idx = self.paper.link_idxs[-1]
        self.contact_body_idx = int(contact_body.index())

        self.add_non_ctrl_systems()

        self.add_ctrl_systems()

        self.connect()


    def add_mbp_bodies(self):
        """
        Initialize and add to builder systems any systems that introduce new
        bodies to the multibody plant, so this needs to be called before
        `plant.Finalize()`, which is called at the end of the function.
        """
        # Pedestal
        pedestal.AddPedestal(self.plant)

        # Paper
        self.paper = Paper(self.plant, self.scene_graph,
            default_joint_angle=0,
            k_J=self.params.k_J, b_J=self.params.b_J,
            m_L=self.params.m_L, w_L=self.params.w_L, h_L=self.params.h_L,
            mu=self.params.mu)
        self.paper.weld_paper_edge(
            constants.PEDESTAL_Y_DIM, pedestal.PEDESTAL_Z_DIM)

        # Manipulator
        self.manipulator_instance = manipulator.data["add_plant_function"](
            plant=self.plant, m_M=self.params.m_M, r=self.params.r,
            mu=self.params.mu, scene_graph=self.scene_graph)

        self.plant.Finalize()

    
    def add_non_ctrl_systems(self):
        """
        Initialize and add to builder systems that don't add bodies to the
        plant (so they can be called after `plant.Finalize()`) that are also
        not controllers.
        """
        # Logger
        self.log_wrapper = LogWrapper(
            self.plant.num_bodies(),
            self.contact_body_idx,
            self.paper,
            self.plant
        )
        self.builder.AddNamedSystem("LogWrapper", self.log_wrapper)

        # Proprioception
        self.proprioception = perception.proprioception.ProprioceptionSystem(
            m_M=self.params.m_M,
            r=self.params.r,
            mu=self.params.mu
        )
        self.builder.AddNamedSystem("proprioception", self.proprioception)

        # Vision
        self.vision_processor = perception.vision.VisionProcessor(
            self.sys_consts)
        self.builder.AddNamedSystem(
            "vision_processor", self.vision_processor)
        self.vision = perception.vision.VisionSystem(
            ll_idx=self.ll_idx, contact_body_idx=self.contact_body_idx)
        self.builder.AddNamedSystem("vision", self.vision)

        if self.meshcat is not None:
            # Visualization
            self.end_effector_frame_vis = visualization.FrameVisualizer(
                name="end_effector", meshcat=self.meshcat)
            self.builder.AddNamedSystem(
                "end_effector_frame_vis", self.end_effector_frame_vis)


    def add_inverse_dynamics_ctrl(self):
        options = {
            'model_friction': True,
            'measure_joint_wrench': False,
        }
        self.fold_ctrl = ctrl.inverse_dynamics.InverseDynamicsController(
            sys_consts=self.sys_consts, options=options)

        if self.meshcat is not None:
            self.desired_position_XYZ = ctrl.aux.XTNtoXYZ()
            self.desired_pos_adder = Adder(2, 3)

            self.builder.AddNamedSystem(
                "desired_position_XYZ", self.desired_position_XYZ)
            self.builder.AddNamedSystem(
                "desired_pos_adder", self.desired_pos_adder)

            self.desired_pos_vis = visualization.FrameVisualizer(
                name="desired position", meshcat=self.meshcat, opacity=0.3)
            self.builder.AddNamedSystem("desired_pos_vis", self.desired_pos_vis)

    def add_kinematic_ctrl(self):
        self.fold_ctrl = ctrl.kinematic_ctrlr.KinematicController(
            sys_consts=self.sys_consts)

    def add_impedance_ctrl(self):
        self.add_impedance_n_hat_force_compensation()

        if self.impedance_type == ImpedanceType.OFFLINE_TRAJ:
            if config.num_links == config.NumLinks.TWO:
                desired_radius = self.sys_consts.w_L/2
            elif config.num_links == config.NumLinks.FOUR:
                desired_radius = 3*self.sys_consts.w_L
            self.setpoint_gen = ctrl.impedance_generators.setpoint_generators.\
                    circle.CircularSetpointGenerator(
                sys_consts=self.sys_consts, desired_radius=desired_radius)
        elif self.impedance_type == ImpedanceType.LINK_FB:
            self.setpoint_gen = ctrl.impedance_generators.setpoint_generators.\
                    link_feedback.LinkFeedbackSetpointGenerator(
                        sys_consts=self.sys_consts)
        
        self.fold_ctrl = ctrl.cartesian_impedance.CartesianImpedanceController(
            sys_consts=self.sys_consts)

        # Order is [theta_x, theta_y, theta_z, x, y, z]
        if self.impedance_stiffness is None:
            if config.num_links == config.NumLinks.TWO:
                self.impedance_stiffness = [1000, 10, 10, 10, 1000, 1000]
            elif config.num_links == config.NumLinks.FOUR:
                self.impedance_stiffness = [100, 1, 1, 1, 1000, 10]
        self.K_gen = ConstantVectorSource(self.impedance_stiffness)
        self.D_gen = ConstantVectorSource(2*np.sqrt(self.impedance_stiffness))
        self.demux_setpoint = Demultiplexer([3,3])
        
        self.builder.AddNamedSystem("K_gen", self.K_gen)
        self.builder.AddNamedSystem("D_gen", self.D_gen)
        self.builder.AddNamedSystem("setpoint_gen", self.setpoint_gen)
        self.builder.AddNamedSystem("demux_setpoint", self.demux_setpoint)
        if self.meshcat is not None:
            self.setpoint_vis = visualization.FrameVisualizer(name="impedance_setpoint", meshcat=self.meshcat, opacity=0.3)
            self.builder.AddNamedSystem("setpoint_vis", self.setpoint_vis)
    
    def add_impedance_n_hat_force_compensation(self):
        """
        If we are using impedance control, initialize and add systems related
        to generating normal compensation.
        """
        if self.n_hat_force_compensation_source == \
                NHatForceCompensationSource.NONE:
            ff_wrench_XYZ = ConstantVectorSource([0, 0, 0, 0, 0, 0])
            self.builder.AddNamedSystem("ff_wrench_XYZ", ff_wrench_XYZ)
        else:
            self.ff_force_XYZ = ctrl.aux.XTNtoXYZ()
            self.ff_torque_XYZ = ConstantVectorSource([0, 0, 0])
            self.ff_wrench_XYZ = Multiplexer([3,3])
            
            self.ff_force_XT = ConstantVectorSource([0, 0])
            self.ff_force_XTN = Multiplexer([2, 1])
            
            self.builder.AddNamedSystem("ff_force_XT", self.ff_force_XT)
            self.builder.AddNamedSystem("ff_force_XTN", self.ff_force_XTN)
            self.builder.AddNamedSystem("ff_force_XYZ", self.ff_force_XYZ)
            self.builder.AddNamedSystem("ff_torque_XYZ", self.ff_torque_XYZ)
            self.builder.AddNamedSystem("ff_wrench_XYZ", self.ff_wrench_XYZ)

            if self.n_hat_force_compensation_source == \
                    NHatForceCompensationSource.MEASURED:
                self.ff_force_N = ctrl.aux.NormalForceSelector(
                    ll_idx=self.ll_idx, contact_body_idx=self.contact_body_idx)
            elif self.n_hat_force_compensation_source == \
                    NHatForceCompensationSource.CONSTANT:
                self.ff_force_N = ConstantVectorSource([10])
            
            self.builder.AddNamedSystem("ff_force_N", self.ff_force_N)
    
    def add_ctrl_systems(self):
        if self.ctrl_paradigm == CtrlParadigm.INVERSE_DYNAMICS:
            self.add_inverse_dynamics_ctrl()
        elif self.ctrl_paradigm == CtrlParadigm.KINEMATIC:
            self.add_kinematic_ctrl()
        elif self.ctrl_paradigm == CtrlParadigm.IMPEDANCE:
            self.add_impedance_ctrl()
        
        self.builder.AddNamedSystem("fold_ctrl", self.fold_ctrl)
        self.tau_g_adder = Adder(3, manipulator.data['nq'])
        self.builder.AddNamedSystem("adder", self.tau_g_adder)
        self.tau_g_gain = Gain(-1, manipulator.data['nq'])
        self.builder.AddNamedSystem("tau_g_gain", self.tau_g_gain)
        self.joint_centering_ctrl = ctrl.aux.JointCenteringCtrl()
        self.builder.AddNamedSystem("joint_centering_ctrl", self.joint_centering_ctrl)
        self.pre_contact_ctrl = ctrl.aux.PreContactCtrl()
        self.builder.AddNamedSystem("pre_contact_ctrl", self.pre_contact_ctrl)
        self.ctrl_selector = ctrl.aux.CtrlSelector()
        self.builder.AddNamedSystem("ctrl_selector", self.ctrl_selector)

    def connect(self):
        # Set up self.vision
        self.builder.Connect(self.plant.get_body_poses_output_port(), self.vision.GetInputPort("poses"))
        self.builder.Connect(self.plant.get_body_spatial_velocities_output_port(), self.vision.GetInputPort("vels"))

        # Set up self.vision processor
        self.builder.Connect(self.vision.GetOutputPort("pose_L_translational"), self.vision_processor.GetInputPort("pose_L_translational"))
        self.builder.Connect(self.vision.GetOutputPort("pose_L_rotational"), self.vision_processor.GetInputPort("pose_L_rotational"))
        self.builder.Connect(self.vision.GetOutputPort("vel_L_translational"), self.vision_processor.GetInputPort("vel_L_translational"))
        self.builder.Connect(self.vision.GetOutputPort("vel_L_rotational"), self.vision_processor.GetInputPort("vel_L_rotational"))
        self.builder.Connect(self.vision.GetOutputPort("pose_M_translational"), self.vision_processor.GetInputPort("pose_M_translational"))
        self.builder.Connect(self.vision.GetOutputPort("pose_M_rotational"), self.vision_processor.GetInputPort("pose_M_rotational"))
        self.builder.Connect(self.vision.GetOutputPort("vel_M_translational"), self.vision_processor.GetInputPort("vel_M_translational"))
        self.builder.Connect(self.vision.GetOutputPort("vel_M_rotational"), self.vision_processor.GetInputPort("vel_M_rotational"))

        # Set up self.proprioception
        self.builder.Connect(self.plant.get_state_output_port(self.manipulator_instance), self.proprioception.GetInputPort("state"))

        # Set up logger
        self.builder.Connect(self.plant.get_body_poses_output_port(), self.log_wrapper.GetInputPort("poses"))
        self.builder.Connect(self.plant.get_body_spatial_velocities_output_port(), self.log_wrapper.GetInputPort("vels"))
        self.builder.Connect(self.plant.get_body_spatial_accelerations_output_port(), self.log_wrapper.GetInputPort("accs")) 
        self.builder.Connect(self.plant.get_contact_results_output_port(), self.log_wrapper.GetInputPort("contact_results"))
        self.builder.Connect(self.plant.get_reaction_forces_output_port(), self.log_wrapper.GetInputPort("joint_forces"))
        self.builder.Connect(self.plant.get_generalized_acceleration_output_port(self.manipulator_instance),
                        self.log_wrapper.GetInputPort("manipulator_accs"))
        self.builder.Connect(self.plant.get_state_output_port(self.manipulator_instance), self.log_wrapper.GetInputPort("state"))
        self.builder.Connect(self.proprioception.GetOutputPort("tau_g"), self.log_wrapper.GetInputPort("tau_g"))
        self.builder.Connect(self.proprioception.GetOutputPort("M"), self.log_wrapper.GetInputPort("M"))
        self.builder.Connect(self.proprioception.GetOutputPort("Cv"), self.log_wrapper.GetInputPort("Cv"))
        self.builder.Connect(self.proprioception.GetOutputPort("J"), self.log_wrapper.GetInputPort("J"))
        self.builder.Connect(self.proprioception.GetOutputPort("Jdot_qdot"), self.log_wrapper.GetInputPort("Jdot_qdot"))
        self.builder.Connect(self.vision_processor.GetOutputPort("in_contact"), self.log_wrapper.GetInputPort("in_contact"))
        self.builder.Connect(self.joint_centering_ctrl.get_output_port(), self.log_wrapper.GetInputPort("joint_centering_torque"))

        # Set up visualization
        if self.meshcat is not None:
            self.builder.Connect(self.vision.GetOutputPort("pose_M_translational"), self.end_effector_frame_vis.GetInputPort("pos"))
            self.builder.Connect(self.vision.GetOutputPort("pose_M_rotational"), self.end_effector_frame_vis.GetInputPort("rot"))

        if (self.ctrl_paradigm == CtrlParadigm.INVERSE_DYNAMICS):
            self.builder.Connect(self.vision_processor.GetOutputPort("theta_L"), self.fold_ctrl.GetInputPort("theta_L"))
            self.builder.Connect(self.vision_processor.GetOutputPort("d_theta_L"), self.fold_ctrl.GetInputPort("d_theta_L"))
            self.builder.Connect(self.vision_processor.GetOutputPort("p_LN"), self.fold_ctrl.GetInputPort("p_LN"))
            self.builder.Connect(self.vision_processor.GetOutputPort("p_LT"), self.fold_ctrl.GetInputPort("p_LT"))
            self.builder.Connect(self.vision_processor.GetOutputPort("theta_MX"), self.fold_ctrl.GetInputPort("theta_MX"))
            self.builder.Connect(self.vision_processor.GetOutputPort("theta_MY"), self.fold_ctrl.GetInputPort("theta_MY"))
            self.builder.Connect(self.vision_processor.GetOutputPort("theta_MZ"), self.fold_ctrl.GetInputPort("theta_MZ"))
            self.builder.Connect(self.vision_processor.GetOutputPort("d_theta_MX"), self.fold_ctrl.GetInputPort("d_theta_MX"))
            self.builder.Connect(self.vision_processor.GetOutputPort("d_theta_MY"), self.fold_ctrl.GetInputPort("d_theta_MY"))
            self.builder.Connect(self.vision_processor.GetOutputPort("d_theta_MZ"), self.fold_ctrl.GetInputPort("d_theta_MZ"))
            self.builder.Connect(self.vision_processor.GetOutputPort("F_GT"), self.fold_ctrl.GetInputPort("F_GT"))
            self.builder.Connect(self.vision_processor.GetOutputPort("F_GN"), self.fold_ctrl.GetInputPort("F_GN"))
            self.builder.Connect(self.vision_processor.GetOutputPort("d_X"), self.fold_ctrl.GetInputPort("d_X"))
            self.builder.Connect(self.vision_processor.GetOutputPort("d_d_X"), self.fold_ctrl.GetInputPort("d_d_X"))
            self.builder.Connect(self.vision_processor.GetOutputPort("d_N"), self.fold_ctrl.GetInputPort("d_N"))
            self.builder.Connect(self.vision_processor.GetOutputPort("d_d_N"), self.fold_ctrl.GetInputPort("d_d_N"))
            self.builder.Connect(self.vision_processor.GetOutputPort("d_T"), self.fold_ctrl.GetInputPort("d_T"))
            self.builder.Connect(self.vision_processor.GetOutputPort("d_d_T"), self.fold_ctrl.GetInputPort("d_d_T"))
            self.builder.Connect(self.vision_processor.GetOutputPort("p_CN"), self.fold_ctrl.GetInputPort("p_CN"))
            self.builder.Connect(self.vision_processor.GetOutputPort("p_CT"), self.fold_ctrl.GetInputPort("p_CT"))
            self.builder.Connect(self.vision_processor.GetOutputPort("p_MConM"), self.fold_ctrl.GetInputPort("p_MConM"))
            self.builder.Connect(self.vision_processor.GetOutputPort("mu_S"), self.fold_ctrl.GetInputPort("mu_S"))
            self.builder.Connect(self.vision_processor.GetOutputPort("hats_T"), self.fold_ctrl.GetInputPort("hats_T"))
            self.builder.Connect(self.vision_processor.GetOutputPort("s_hat_X"), self.fold_ctrl.GetInputPort("s_hat_X"))
            self.builder.Connect(self.proprioception.GetOutputPort("tau_g"), self.fold_ctrl.GetInputPort("tau_g"))
            self.builder.Connect(self.proprioception.GetOutputPort("M"), self.fold_ctrl.GetInputPort("M"))
            self.builder.Connect(self.proprioception.GetOutputPort("Cv"), self.fold_ctrl.GetInputPort("Cv"))
            self.builder.Connect(self.proprioception.GetOutputPort("J"), self.fold_ctrl.GetInputPort("J"))
            self.builder.Connect(self.proprioception.GetOutputPort("J_translational"), self.fold_ctrl.GetInputPort("J_translational"))
            self.builder.Connect(self.proprioception.GetOutputPort("J_rotational"), self.fold_ctrl.GetInputPort("J_rotational"))
            self.builder.Connect(self.proprioception.GetOutputPort("Jdot_qdot"), self.fold_ctrl.GetInputPort("Jdot_qdot"))
            self.builder.Connect(self.joint_centering_ctrl.get_output_port(), self.fold_ctrl.GetInputPort("joint_centering_torque"))
            
            if self.meshcat is not None:
                self.builder.Connect(self.fold_ctrl.GetOutputPort("XTNd"), self.desired_position_XYZ.GetInputPort("XTN"))
                self.builder.Connect(self.vision_processor.GetOutputPort("T_hat"), self.desired_position_XYZ.GetInputPort("T_hat"))
                self.builder.Connect(self.vision_processor.GetOutputPort("N_hat"), self.desired_position_XYZ.GetInputPort("N_hat"))
            
                self.builder.Connect(self.vision.GetOutputPort("pose_L_translational"), self.desired_pos_adder.get_input_port(0))
                self.builder.Connect(self.desired_position_XYZ.GetOutputPort("XYZ"), self.desired_pos_adder.get_input_port(1))
                self.builder.Connect(self.desired_pos_adder.get_output_port(), self.desired_pos_vis.GetInputPort("pos"))
            
                self.builder.Connect(self.fold_ctrl.GetOutputPort("rot_XYZd"), self.desired_pos_vis.GetInputPort("rot"))
        elif (self.ctrl_paradigm == CtrlParadigm.KINEMATIC):
            self.builder.Connect(self.vision_processor.GetOutputPort("theta_L"), self.fold_ctrl.GetInputPort("theta_L"))
            self.builder.Connect(self.vision_processor.GetOutputPort("d_theta_L"), self.fold_ctrl.GetInputPort("d_theta_L"))
            self.builder.Connect(self.vision_processor.GetOutputPort("theta_MX"), self.fold_ctrl.GetInputPort("theta_MX"))
            self.builder.Connect(self.vision_processor.GetOutputPort("theta_MY"), self.fold_ctrl.GetInputPort("theta_MY"))
            self.builder.Connect(self.vision_processor.GetOutputPort("theta_MZ"), self.fold_ctrl.GetInputPort("theta_MZ"))
            self.builder.Connect(self.vision_processor.GetOutputPort("d_theta_MX"), self.fold_ctrl.GetInputPort("d_theta_MX"))
            self.builder.Connect(self.vision_processor.GetOutputPort("d_theta_MY"), self.fold_ctrl.GetInputPort("d_theta_MY"))
            self.builder.Connect(self.vision_processor.GetOutputPort("d_theta_MZ"), self.fold_ctrl.GetInputPort("d_theta_MZ"))
            self.builder.Connect(self.vision_processor.GetOutputPort("F_GN"), self.fold_ctrl.GetInputPort("F_GN"))
            self.builder.Connect(self.vision_processor.GetOutputPort("d_X"), self.fold_ctrl.GetInputPort("d_X"))
            self.builder.Connect(self.vision_processor.GetOutputPort("d_d_X"), self.fold_ctrl.GetInputPort("d_d_X"))
            self.builder.Connect(self.vision_processor.GetOutputPort("d_T"), self.fold_ctrl.GetInputPort("d_T"))
            self.builder.Connect(self.vision_processor.GetOutputPort("d_d_T"), self.fold_ctrl.GetInputPort("d_d_T"))
            self.builder.Connect(self.vision_processor.GetOutputPort("T_hat"), self.fold_ctrl.GetInputPort("T_hat"))
            self.builder.Connect(self.vision_processor.GetOutputPort("N_hat"), self.fold_ctrl.GetInputPort("N_hat"))
            self.builder.Connect(self.proprioception.GetOutputPort("J_translational"), self.fold_ctrl.GetInputPort("J_translational"))
            self.builder.Connect(self.proprioception.GetOutputPort("J_rotational"), self.fold_ctrl.GetInputPort("J_rotational"))
        elif (self.ctrl_paradigm == CtrlParadigm.IMPEDANCE):
            self.builder.Connect(self.vision.GetOutputPort("pose_M_translational"), self.fold_ctrl.GetInputPort("pose_M_translational"))
            self.builder.Connect(self.vision.GetOutputPort("pose_M_rotational"), self.fold_ctrl.GetInputPort("pose_M_rotational"))
            self.builder.Connect(self.vision.GetOutputPort("vel_M_translational"), self.fold_ctrl.GetInputPort("vel_M_translational"))
            self.builder.Connect(self.vision.GetOutputPort("vel_M_rotational"), self.fold_ctrl.GetInputPort("vel_M_rotational"))
            self.builder.Connect(self.proprioception.GetOutputPort("M"), self.fold_ctrl.GetInputPort("M"))
            self.builder.Connect(self.proprioception.GetOutputPort("J"), self.fold_ctrl.GetInputPort("J"))
            self.builder.Connect(self.proprioception.GetOutputPort("Jdot_qdot"), self.fold_ctrl.GetInputPort("Jdot_qdot"))
            self.builder.Connect(self.proprioception.GetOutputPort("Cv"), self.fold_ctrl.GetInputPort("Cv"))
            
            self.builder.Connect(self.K_gen.get_output_port(), self.fold_ctrl.GetInputPort("K"))
            self.builder.Connect(self.D_gen.get_output_port(), self.fold_ctrl.GetInputPort("D"))
            self.builder.Connect(self.setpoint_gen.GetOutputPort("x0"), self.fold_ctrl.GetInputPort("x0"))
            self.builder.Connect(self.setpoint_gen.GetOutputPort("dx0"), self.fold_ctrl.GetInputPort("dx0"))
            if self.n_hat_force_compensation_source != NHatForceCompensationSource.NONE:
                self.builder.Connect(self.ff_force_XTN.get_output_port(), self.ff_force_XYZ.GetInputPort("XTN"))
                self.builder.Connect(self.vision_processor.GetOutputPort("T_hat"), self.ff_force_XYZ.GetInputPort("T_hat"))
                self.builder.Connect(self.vision_processor.GetOutputPort("N_hat"), self.ff_force_XYZ.GetInputPort("N_hat"))
                self.builder.Connect(self.ff_torque_XYZ.get_output_port(), self.ff_wrench_XYZ.get_input_port(0))
                self.builder.Connect(self.ff_force_XYZ.get_output_port(), self.ff_wrench_XYZ.get_input_port(1))
                self.builder.Connect(self.ff_force_XT.get_output_port(), self.ff_force_XTN.get_input_port(0))
                self.builder.Connect(self.ff_force_N.get_output_port(), self.ff_force_XTN.get_input_port(1))

                if self.n_hat_force_compensation_source == NHatForceCompensationSource.MEASURED:
                    self.builder.Connect(self.plant.get_contact_results_output_port(), self.ff_force_N.get_input_port())
                    
                
            self.builder.Connect(self.ff_wrench_XYZ.get_output_port(), self.fold_ctrl.GetInputPort("feedforward_wrench"))
            
            self.builder.Connect(self.setpoint_gen.GetOutputPort("x0"), self.demux_setpoint.get_input_port())
            if self.meshcat is not None:
                self.builder.Connect(self.fold_ctrl.GetOutputPort("adjusted_x0_pos"), self.setpoint_vis.GetInputPort("pos"))
                self.builder.Connect(self.fold_ctrl.GetOutputPort("adjusted_x0_rot"), self.setpoint_vis.GetInputPort("rot"))
            
            if type(self.setpoint_gen) is \
                    ctrl.impedance_generators.setpoint_generators.link_feedback.LinkFeedbackSetpointGenerator:
                self.builder.Connect(
                    self.vision.GetOutputPort("pose_L_translational"),
                    self.setpoint_gen.GetInputPort("pose_L_translational")
                )
                self.builder.Connect(
                    self.vision.GetOutputPort("pose_L_rotational"),
                    self.setpoint_gen.GetInputPort("pose_L_rotational")
                )

        # Controller connections
        self.builder.Connect(self.vision_processor.GetOutputPort("in_contact"), self.ctrl_selector.GetInputPort("in_contact"))
        self.builder.Connect(self.fold_ctrl.GetOutputPort("tau_out"), self.ctrl_selector.GetInputPort("contact_ctrl"))
        if self.ctrl_paradigm == CtrlParadigm.INVERSE_DYNAMICS:
            self.builder.Connect(self.pre_contact_ctrl.get_output_port(), self.ctrl_selector.GetInputPort("pre_contact_ctrl"))
        else:
            self.builder.Connect(self.fold_ctrl.get_output_port(0), self.ctrl_selector.GetInputPort("pre_contact_ctrl"))

        self.builder.Connect(self.ctrl_selector.get_output_port(), self.log_wrapper.GetInputPort("tau_ctrl"))

        self.builder.Connect(self.proprioception.GetOutputPort("J"), self.pre_contact_ctrl.GetInputPort("J"))
        self.builder.Connect(self.vision_processor.GetOutputPort("v_MN"), self.pre_contact_ctrl.GetInputPort("v_MN"))
        self.builder.Connect(self.vision_processor.GetOutputPort("N_hat"), self.pre_contact_ctrl.GetInputPort("N_hat"))

        self.builder.Connect(self.proprioception.GetOutputPort("J"), self.joint_centering_ctrl.GetInputPort("J"))
        self.builder.Connect(self.proprioception.GetOutputPort("q"), self.joint_centering_ctrl.GetInputPort("q"))
        self.builder.Connect(self.proprioception.GetOutputPort("v"), self.joint_centering_ctrl.GetInputPort("v"))

        self.builder.Connect(self.proprioception.GetOutputPort("tau_g"), self.tau_g_gain.get_input_port())

        self.builder.Connect(self.joint_centering_ctrl.get_output_port(), self.tau_g_adder.get_input_port(0))
        self.builder.Connect(self.ctrl_selector.get_output_port(), self.tau_g_adder.get_input_port(1))
        self.builder.Connect(self.tau_g_gain.get_output_port(), self.tau_g_adder.get_input_port(2))

        self.builder.Connect(self.tau_g_adder.get_output_port(), self.plant.get_actuation_input_port())
        self.builder.Connect(self.tau_g_adder.get_output_port(), self.log_wrapper.GetInputPort("tau_out"))

        # Visualization and logging
        self.logger = LogVectorOutput(self.log_wrapper.get_output_port(), self.builder)

        if self.meshcat is not None:
            meshcat_params = MeshcatVisualizerParams()
            self.vis = MeshcatVisualizerCpp.AddToBuilder(self.builder, self.scene_graph.get_query_output_port(), self.meshcat, meshcat_params)

            self.end_effector_frame_vis.set_animation(self.vis.get_mutable_recording())
            if self.ctrl_paradigm == CtrlParadigm.INVERSE_DYNAMICS:
                self.desired_pos_vis.set_animation(self.vis.get_mutable_recording())
            elif self.ctrl_paradigm == CtrlParadigm.IMPEDANCE:
                self.setpoint_vis.set_animation(self.vis.get_mutable_recording())

        # Build diagram
        self.diagram = self.builder.Build()


    def get_viz_str(self):
        # Regex processing
        ## Remove objects we don't want to visualize
        objects_to_remove = [
            "VectorLogSink",
            "LogWrapper",
            "scene_graph",
        ]
        if self.meshcat is not None:
            objects_to_remove.append("MeshcatVisualizer")
            objects_to_remove.append("end_effector_frame_vis")
            if self.ctrl_paradigm == CtrlParadigm.IMPEDANCE:
                objects_to_remove.append("setpoint_vis")
        viz_str = self.diagram.GetGraphvizString()
        for obj in objects_to_remove:
            expr = r'([0-9]{15}) \[[^\]]*label="[^"]*' + obj
            ID = re.search(expr, viz_str).group(1)
            viz_str = re.sub(r";[^;]*" + ID + r"[^;]+", "", viz_str)

        ## Fix spacing
        viz_str_split = viz_str.split("{", 1)
        viz_str = viz_str_split[0] + "{" + "\nranksep=2" + viz_str_split[1]
        
        ## Grab some handy variables
        plant_id = re.search(r'(\d+) .*plant', viz_str).group(1)
        vision_id = re.search(r'(\d+) .*\bvision\b', viz_str).group(1)

        # Loop processing
        colored_edges = [
            [plant_id]
        ]

        viz_str_new = ""
        for l in viz_str.split("\n"):
            if ("->" in l) and (l.strip().startswith(plant_id)):
                out_l = l[:-1] + " [style=dotted];"
            else:
                out_l = l
            viz_str_new += out_l + "\n"
        viz_str = viz_str_new

        # Human readable string
        viz_str_human_readable = viz_str
        viz_str_human_readable = re.sub(r' \{', "\n{", viz_str_human_readable)

        viz_str_human_readable_new = ""
        indent_level = 0
        for l in viz_str_human_readable.split("\n"):
            if l.strip() == "}":
                indent_level -= 1
            out_l = " "*4*indent_level + l + "\n"
            if l.strip() == "{":
                indent_level += 1
            viz_str_human_readable_new += out_l
            
        viz_str_human_readable = viz_str_human_readable_new
        # print(viz_str_human_readable)

        return viz_str
