import pydrake
# TODO: pull these out
from pydrake.all import (
    BodyIndex, Adder, ConstantVectorSource, Multiplexer, Demultiplexer, Gain,
    LogVectorOutput, MeshcatVisualizerParams, MeshcatVisualizerCpp, Saturation,
    OutputPort, InputPort
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
import ctrl.impedance_generators.setpoint_generators.offline_loader
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
    # ============================ PUBLIC FUNCTIONS ===========================
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
        assert (n_hat_force_compensation_source == NHatForceCompensationSource.NONE) or \
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

        self._add_mbp_bodies()

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

        self._add_non_ctrl_systems()
        self._add_ctrl_systems()
        self._wire()
        self._context_specific_init()


    def get_viz_str(self):
        # Regex processing
        ## Remove objects we don't want to visualize
        objects_to_remove = [
            # "VectorLogSink",
            # "log",
            "scene_graph",
        ]
        if self.meshcat is not None:
            objects_to_remove.append("MeshcatVisualizer")
            objects_to_remove.append("end_effector_frame_vis")
            objects_to_remove.append("link_frame_vis")
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


    def run_sim(self, clean_exit=True):
        """
        Run the actual simulation and return the log.

        :param clean_exit: Whether or not to catch and print (but otherwise
                           ignore) any errors that happen during the sim
        """
        # Finalize simulation and visualization
        simulator = pydrake.systems.analysis.Simulator(
            self.diagram, self.diagram_context)
        simulator.Initialize()
        if self.meshcat is not None:
            self.vis.StartRecording()

        if clean_exit:
            try:
                simulator.AdvanceTo(config.TSPAN)
            except BaseException as e:
                # Has to be BaseException to get KeyboardInterrupt
                print(type(e))
                print(e)
                if self.meshcat is not None:
                    self.vis.StopRecording()
                    self.vis.PublishRecording()

                return self.logger.FindLog(simulator.get_context())
        else:
            simulator.AdvanceTo(config.TSPAN)

        if self.meshcat is not None:
            self.vis.StopRecording()
            self.vis.PublishRecording()

        return self.logger.FindLog(simulator.get_context())


    # =========================== PRIVATE FUNCTIONS ===========================

    def _add_mbp_bodies(self):
        """
        Initialize and add to builder systems any systems that introduce new
        bodies to the multibody plant, so this needs to be called before
        `plant.Finalize()`, which is called at the end of the function.
        """
        # Pedestal
        pedestal_instance = pedestal.AddPedestal(self.plant)

        # Paper
        self.paper = Paper(self.plant, self.scene_graph,
            default_joint_angle=0,
            k_J=self.params.k_J, b_J=self.params.b_J,
            m_L=self.params.m_L, w_L=self.params.w_L, h_L=self.params.h_L,
            mu=self.params.mu)
        self.paper.weld_paper_edge(pedestal_instance)

        # Manipulator
        self.manipulator_instance = manipulator.data["add_plant_function"](
            plant=self.plant, m_M=self.params.m_M, r=self.params.r,
            mu=self.params.mu, scene_graph=self.scene_graph)

        self.plant.Finalize()

 
    def _add_non_ctrl_systems(self):
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
        self.builder.AddNamedSystem("log", self.log_wrapper)

        # Proprioception
        self.proprioception = perception.proprioception.ProprioceptionSystem(
            m_M=self.params.m_M,
            r=self.params.r,
            mu=self.params.mu
        )
        self.builder.AddNamedSystem("prop", self.proprioception)

        # Vision
        self.vision_processor = perception.vision.VisionProcessor(
            self.sys_consts)
        self.builder.AddNamedSystem(
            "vis_proc", self.vision_processor)
        self.vision = perception.vision.VisionSystem(
            ll_idx=self.ll_idx, contact_body_idx=self.contact_body_idx)
        self.builder.AddNamedSystem("vision", self.vision)

        if self.meshcat is not None:
            # Visualization
            self.end_effector_frame_vis = visualization.FrameVisualizer(
                name="end_effector", meshcat=self.meshcat)
            self.builder.AddNamedSystem(
                "end_effector_frame_vis", self.end_effector_frame_vis)
            self.link_frame_vis = visualization.FrameVisualizer(
                name="last_link", meshcat=self.meshcat)
            self.builder.AddNamedSystem("link_frame_vis", self.link_frame_vis)

    def _add_inverse_dynamics_ctrl(self):
        options = {
            'model_friction': True,
            'measure_joint_wrench': False,
        }
        self.fold_ctrl = ctrl.inverse_dynamics.InverseDynamicsController(
            sys_consts=self.sys_consts, options=options)

        if self.meshcat is not None:
            self.desired_position_XYZ = ctrl.aux.HTNtoXYZ()
            self.desired_pos_adder = Adder(2, 3)

            self.builder.AddNamedSystem(
                "desired_position_XYZ", self.desired_position_XYZ)
            self.builder.AddNamedSystem(
                "desired_pos_adder", self.desired_pos_adder)

            self.desired_pos_vis = visualization.FrameVisualizer(
                name="desired position", meshcat=self.meshcat, opacity=0.3)
            self.builder.AddNamedSystem("desired_pos_vis", self.desired_pos_vis)


    def _add_kinematic_ctrl(self):
        self.fold_ctrl = ctrl.kinematic_ctrlr.KinematicController(
            sys_consts=self.sys_consts)


    def _add_impedance_ctrl(self):
        self._add_impedance_n_hat_force_compensation()

        if self.impedance_type == ImpedanceType.OFFLINE_TRAJ:
            self.setpoint_gen = ctrl.impedance_generators.setpoint_generators.\
                    offline_loader.OfflineTrajLoader()
        elif self.impedance_type == ImpedanceType.LINK_FB:
            self.setpoint_gen = ctrl.impedance_generators.setpoint_generators.\
                    link_feedback.LinkFeedbackSetpointGenerator(
                        sys_consts=self.sys_consts)
        
        self.fold_ctrl = ctrl.cartesian_impedance.CartesianImpedanceController(
            sys_consts=self.sys_consts)

        # Order is [theta_x, theta_y, theta_z, x, y, z]
        if self.impedance_stiffness is None:
            if config.num_links == config.NumLinks.TWO:
                self.impedance_stiffness = [4, 4, 4, 40, 40, 40]
            elif config.num_links == config.NumLinks.FOUR:
                self.impedance_stiffness = [4, 4, 4, 40, 40, 40]
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


    def _add_impedance_n_hat_force_compensation(self):
        """
        If we are using impedance control, initialize and add systems related
        to generating normal compensation.
        """
        if self.n_hat_force_compensation_source == \
                NHatForceCompensationSource.NONE:
            self.ff_wrench_XYZ = ConstantVectorSource([0, 0, 0, 0, 0, 0])
            self.builder.AddNamedSystem("ff_wrench_XYZ", self.ff_wrench_XYZ)
        else:
            self.ff_force_XYZ = ctrl.aux.HTNtoXYZ()
            self.ff_torque_XYZ = ConstantVectorSource([0, 0, 0])
            self.ff_wrench_XYZ = Multiplexer([3,3])
            
            self.ff_force_HT = ConstantVectorSource([0, 0])
            self.ff_force_HTN = Multiplexer([2, 1])
            
            self.builder.AddNamedSystem("ff_force_HT", self.ff_force_HT)
            self.builder.AddNamedSystem("ff_force_HTN", self.ff_force_HTN)
            self.builder.AddNamedSystem("ff_force_XYZ", self.ff_force_XYZ)
            self.builder.AddNamedSystem("ff_torque_XYZ", self.ff_torque_XYZ)
            self.builder.AddNamedSystem("ff_wrench_XYZ", self.ff_wrench_XYZ)

            if self.n_hat_force_compensation_source == \
                    NHatForceCompensationSource.MEASURED:
                self.ff_force_N = ctrl.aux.NormalForceSelector(
                    ll_idx=self.ll_idx,
                    contact_body_idx=self.contact_body_idx,
                    ff_constant_force=1
                )
            elif self.n_hat_force_compensation_source == \
                    NHatForceCompensationSource.CONSTANT:
                self.ff_force_N = ConstantVectorSource([5])
            
            self.builder.AddNamedSystem("ff_force_N", self.ff_force_N)
            self.ff_force_N_Sat = Saturation(min_value=[0],max_value=[50])
            self.builder.AddNamedSystem("ff_force_N_Sat", self.ff_force_N_Sat)

 
    def _add_ctrl_systems(self):
        if self.ctrl_paradigm == CtrlParadigm.INVERSE_DYNAMICS:
            self._add_inverse_dynamics_ctrl()
        elif self.ctrl_paradigm == CtrlParadigm.KINEMATIC:
            self._add_kinematic_ctrl()
        elif self.ctrl_paradigm == CtrlParadigm.IMPEDANCE:
            self._add_impedance_ctrl()
        
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

    def _context_specific_init(self):
        self.diagram_context = self.diagram.CreateDefaultContext()
        manipulator.data['set_positions'](
            self.diagram, self.diagram_context, self.plant,
            self.manipulator_instance)
        self.paper.set_positions(self.diagram, self.diagram_context)


    def _connect(self, out_data, inp_data):
        # Get output port
        if isinstance(out_data, str):
            output_port = self._get_system(out_data).get_output_port()
        elif isinstance(out_data, OutputPort):
            output_port = out_data
        else:
            output_sys = self._get_system(out_data[0])
            if isinstance(out_data[1], int):
                output_port = output_sys.get_output_port(out_data[1])
            else:
                output_port = output_sys.GetOutputPort(out_data[1])

        # Get input port
        if isinstance(inp_data, str):
            input_port = self._get_system(inp_data).get_input_port()
        elif isinstance(inp_data, InputPort):
            input_port = inp_data
        else:
            input_sys = self._get_system(inp_data[0])
            if isinstance(inp_data[1], int):
                input_port = input_sys.get_input_port(inp_data[1])
            else:
                input_port = input_sys.GetInputPort(inp_data[1])
        self.builder.Connect(output_port, input_port)

    def _connect_all_inputs(self, out_sys_name, inp_sys_name,
            skipped_ports=set()):
        inp_sys = self._get_system(inp_sys_name)
        for i in range(inp_sys.num_input_ports()):
            # Hypothetically, we could go by index. But, I want to be robust
            # to them having different orders (although they should always
            # have the same names for parts)
            name = inp_sys.get_input_port(i).get_name()
            if name in skipped_ports:
                continue
            self._connect([out_sys_name, name], [inp_sys_name, name])

    def _connect_all_outputs(self, out_sys_name, in_sys_name,
            skipped_ports=set()):
        out_sys = self._get_system(out_sys_name)
        for i in range(out_sys.num_output_ports()):
            # Hypothetically, we could go by index. But, I want to be robust
            # to them having different orders (although they should always
            # have the same names for parts)
            name = out_sys.get_output_port(i).get_name()
            if name in skipped_ports:
                continue
            self._connect([out_sys_name, name], [in_sys_name, name])

    def _get_system(self, sys_name):
        systems = self.builder.GetMutableSystems()
        for sys in systems:
            if sys.get_name() == sys_name:
                return sys
        raise KeyError(f"No system in builder with name \"{sys_name}\"")

    def _wire(self):
        # Set up self.vision
        self._connect(["plant", "body_poses"], ["vision", "poses"])
        self._connect(["plant", "spatial_velocities"], ["vision", "vels"])

        # Set up self.vision processor
        self._connect_all_outputs("vision", "vis_proc")

        # Set up self.proprioception
        self._connect(
            self.plant.get_state_output_port(self.manipulator_instance),
            ["prop", "state"]
        )

        # Set up logger
        self._connect(["plant", "body_poses"], ["log", "poses"])
        self._connect(["plant", "spatial_velocities"], ["log", "vels"])
        self._connect(["plant", "spatial_accelerations"], ["log", "accs"])
        self._connect(["plant", "contact_results"], ["log", "contact_results"])
        self._connect(["plant", "reaction_forces"], ["log", "joint_forces"])
        self._connect(self.plant.get_generalized_acceleration_output_port(
                self.manipulator_instance), ["log", "manipulator_accs"])
        self._connect(self.plant.get_state_output_port(
                self.manipulator_instance), ["log", "state"])
        self._connect(["prop", "tau_g"], ["log", "tau_g"])
        self._connect(["prop", "M"], ["log", "M"])
        self._connect(["prop", "Cv"], ["log", "Cv"])
        self._connect(["prop", "J"], ["log", "J"])
        self._connect(["prop", "Jdot_qdot"], ["log", "Jdot_qdot"])
        self._connect(["vis_proc", "in_contact"], ["log", "in_contact"])
        self._connect("joint_centering_ctrl",
            ["log", "joint_centering_torque"])

        # Set up visualization
        if self.meshcat is not None:
            self._connect(["vision", "pose_M_translational"],
                ["end_effector_frame_vis", "pos"])
            self._connect(["vision", "pose_M_rotational"],
                ["end_effector_frame_vis", "rot"])
            self._connect(["vision", "pose_L_translational"],
                ["link_frame_vis", "pos"])
            self._connect(["vision", "pose_L_rotational"],
                ["link_frame_vis", "rot"])

        if (self.ctrl_paradigm == CtrlParadigm.INVERSE_DYNAMICS):
            skipped_ports = {"T_hat", "N_hat", "in_contact", "v_LN", "v_MN"}
            self._connect_all_outputs(
                "vis_proc", "fold_ctrl", skipped_ports=skipped_ports)

            self._connect_all_outputs(
                "prop", "fold_ctrl", skipped_ports={"q", "v"})

            self._connect("joint_centering_ctrl",
                ["fold_ctrl", "joint_centering_torque"])
            
            if self.meshcat is not None:
                self._connect(
                    ["fold_ctrl", "HTNd"], ["desired_position_XYZ", "HTN"])
                self._connect(
                    ["vis_proc", "T_hat"], ["desired_position_XYZ", "T_hat"])
                self._connect(
                    ["vis_proc", "N_hat"], ["desired_position_XYZ", "N_hat"])
                self._connect(
                    ["vision", "pose_L_translational"],
                    ["desired_pos_adder", 0])
                self._connect(
                    ["desired_position_XYZ", "XYZ"],
                    ["desired_pos_adder", 1])
                self._connect("desired_pos_adder", ["desired_pos_vis", "pos"])
                self._connect(
                    ["fold_ctrl", "rot_XYZd"], ["desired_pos_vis", "rot"])
        elif (self.ctrl_paradigm == CtrlParadigm.KINEMATIC):
            self._connect_all_inputs("vis_proc", "fold_ctrl",
                skipped_ports={"J_translational", "J_rotational"})
            self._connect(
                ["prop", "J_translational"], ["fold_ctrl", "J_translational"])
            self._connect(
                ["prop", "J_rotational"], ["fold_ctrl", "J_rotational"])
        elif (self.ctrl_paradigm == CtrlParadigm.IMPEDANCE):
            for k in ["pose_M_translational", "pose_M_rotational",
                    "vel_M_translational", "vel_M_rotational"]:
                self._connect(["vision", k], ["fold_ctrl", k])
            for k in ["M", "J", "Jdot_qdot", "Cv"]:
                self._connect(["prop", k], ["fold_ctrl", k])
            self._connect("K_gen", ["fold_ctrl", "K"])
            self._connect("D_gen", ["fold_ctrl", "D"])
            self._connect(["setpoint_gen", "x0"], ["fold_ctrl", "x0"])
            self._connect(["setpoint_gen", "dx0"], ["fold_ctrl", "dx0"])
            if self.n_hat_force_compensation_source != NHatForceCompensationSource.NONE:
                self._connect("ff_force_HTN", ["ff_force_XYZ", "HTN"])
                self._connect(["vis_proc", "T_hat"], ["ff_force_XYZ", "T_hat"])
                self._connect(["vis_proc", "N_hat"], ["ff_force_XYZ", "N_hat"])
                self._connect("ff_torque_XYZ", ["ff_wrench_XYZ", 0])
                self._connect("ff_force_XYZ", ["ff_wrench_XYZ", 1])
                self._connect("ff_force_HT", ["ff_force_HTN", 0])
                self._connect("ff_force_N", "ff_force_N_Sat")
                self._connect("ff_force_N_Sat", ["ff_force_HTN", 1])

                if self.n_hat_force_compensation_source == \
                        NHatForceCompensationSource.MEASURED:
                    self._connect(["plant", "contact_results"], "ff_force_N")
            self._connect("ff_wrench_XYZ", ["fold_ctrl", "feedforward_wrench"])
            
            self._connect(["setpoint_gen", "x0"], "demux_setpoint")
            if self.meshcat is not None:
                self._connect(["fold_ctrl", "adjusted_x0_pos"],
                    ["setpoint_vis", "pos"])
                self._connect(["fold_ctrl", "adjusted_x0_rot"],
                    ["setpoint_vis", "rot"])
            
            if type(self.setpoint_gen) is \
                    ctrl.impedance_generators.setpoint_generators.link_feedback.LinkFeedbackSetpointGenerator:
                self._connect_all_inputs("vision", "setpoint_gen")

        # Controller connections
        ## Controller mux
        self._connect(["vis_proc", "in_contact"],
            ["ctrl_selector", "in_contact"])
        self._connect(["fold_ctrl", "tau_out"],
            ["ctrl_selector", "contact_ctrl"])
        if self.ctrl_paradigm == CtrlParadigm.INVERSE_DYNAMICS:
            self._connect("pre_contact_ctrl",
                ["ctrl_selector", "pre_contact_ctrl"])
        else:
            self._connect(["fold_ctrl", "tau_out"],
                ["ctrl_selector", "pre_contact_ctrl"])
        self._connect("ctrl_selector", ["log", "tau_ctrl"])
        self._connect(["fold_ctrl", "tau_out"], ["log", "tau_contact_ctrl"])

        ## Wire pre contact control
        self._connect(["prop", "J"], ["pre_contact_ctrl", "J"])
        self._connect(["vision", "vel_M_translational"], ["pre_contact_ctrl", "v_M"])

        ## Set up joint centering control
        self._connect_all_inputs("prop", "joint_centering_ctrl")

        ## Wire gravity compensation + final adder
        self._connect(["prop", "tau_g"], ["tau_g_gain", 0])
        self._connect("joint_centering_ctrl", ["adder", 0])
        self._connect("ctrl_selector", ["adder", 1])
        self._connect("tau_g_gain", ["adder", 2])

        self._connect("adder", ["plant", "panda_actuation"])
        self._connect("adder", ["log", "tau_out"])

        # Visualization and logging
        self.logger = LogVectorOutput(self.log_wrapper.get_output_port(), self.builder)

        if self.meshcat is not None:
            meshcat_params = MeshcatVisualizerParams()
            self.vis = MeshcatVisualizerCpp.AddToBuilder(self.builder, self.scene_graph.get_query_output_port(), self.meshcat, meshcat_params)

            self.end_effector_frame_vis.set_animation(self.vis.get_mutable_recording())
            self.link_frame_vis.set_animation(self.vis.get_mutable_recording())
            if self.ctrl_paradigm == CtrlParadigm.INVERSE_DYNAMICS:
                self.desired_pos_vis.set_animation(self.vis.get_mutable_recording())
            elif self.ctrl_paradigm == CtrlParadigm.IMPEDANCE:
                self.setpoint_vis.set_animation(self.vis.get_mutable_recording())

        # Build diagram
        self.diagram = self.builder.Build()
