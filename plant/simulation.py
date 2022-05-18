import pydrake
# TODO: pull these out
from pydrake.all import (
    BodyIndex, Adder, ConstantVectorSource, Multiplexer, Demultiplexer, Gain,
    LogVectorOutput, MeshcatVisualizerParams, MeshcatVisualizerCpp, Saturation,
    OutputPort, InputPort, DiscreteTimeDelay
)

import numpy as np

from dataclasses import dataclass
import enum
import re
import copy

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
import ctrl.impedance_generators.setpoint_generators.force_feedback
import ctrl.aux
import sim_exceptions

from log_wrapper import LogWrapper

import constants

import visualization

import config

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

default_port_noise_map = {
    "pose_L_rotational": 0,
    "pose_L_translational": 0,
    "vel_L_rotational": 0,
    "vel_L_translational": 0,
    "pose_M_rotational": 0,
    "pose_M_translational": 0,
    "vel_M_rotational": 0,
    "vel_M_translational": 0,
    "q": 0,
    "v": 0,
    "Fn": 0,
    "F": 0
}

# Enums
class CtrlParadigm(enum.Enum):
    INVERSE_DYNAMICS = enum.auto()
    KINEMATIC = enum.auto()
    IMPEDANCE = enum.auto()


class ImpedanceType(enum.Enum):
    OFFLINE_TRAJ = enum.auto()
    LINK_FB = enum.auto()
    FORCE_FB = enum.auto()
    NONE = enum.auto


class NHatForceCompensationSource(enum.Enum):
    PURE_FN = enum.auto()
    CONTACT_FORCE = enum.auto()
    CONSTANT = enum.auto()
    NONE = enum.auto()


class Simulation:
    # ============================ PUBLIC FUNCTIONS ===========================
    def __init__(self,
            ctrl_paradigm: CtrlParadigm, impedance_type: ImpedanceType,
            n_hat_force_compensation_source: NHatForceCompensationSource,
            num_links: config.NumLinks, DT, TSPAN,
            sim_params=None, ctrl_params=None, meshcat=None,
            impedance_stiffness=None, exit_when_folded=False, const_ff_Fn=5,
            timeout=None, noise=default_port_noise_map, impedance_scale=1,
            model_impedance_delay=False, use_true_hinge_diameter=False):
        # System parameters
        # Sim is used for simulation. Ctrl is used for everything else
        if sim_params is None:
            self.sim_sys_consts = constants.nominal_sys_consts(num_links)
        else: 
            self.sim_sys_consts = sim_params
        if ctrl_params is None:
            self.ctrl_sys_consts = copy.deepcopy(self.sim_sys_consts)
        else: 
            self.ctrl_sys_consts = ctrl_params

        # Controller parameters
        self.ctrl_paradigm = ctrl_paradigm
        self.impedance_type = impedance_type
        # TODO: This needs a better name
        self.n_hat_force_compensation_source = n_hat_force_compensation_source
        
        # Order is [theta_x, theta_y, theta_z, x, y, z]
        if impedance_stiffness is None:
            self.impedance_stiffness = [4, 4, 4, 40, 40, 40]
        else:
            self.impedance_stiffness = impedance_stiffness
        self.impedance_stiffness = np.array(self.impedance_stiffness)*impedance_scale
        # TODO: better name?
        self.const_ff_Fn = const_ff_Fn

        # Noise parameters
        self.noise = copy.deepcopy(noise)
        for k, v in default_port_noise_map.items():
            if not (k in self.noise.keys()):
                self.noise[k] = v
        assert len(self.noise.keys() - default_port_noise_map.keys()) == 0

        # Other settings
        self.meshcat = meshcat
        self.TSPAN = TSPAN
        self.num_links = num_links
        self.model_impedance_delay = model_impedance_delay

        # Set up logs
        self.tau_O_log = [0]
        self.F_OT_log = [0]
        self.F_ON_log = [0]

        # Verify controller config is valid
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
                self.builder, time_step=DT)
        self.plant.set_stiction_tolerance(constants.v_stiction)
        self.plant.set_penetration_allowance(0.001)

        self._add_mbp_bodies(use_true_hinge_diameter)

        # Init sim_sys_consts
        self.sim_sys_consts.I_L = self.plant.get_body(
            BodyIndex(self.paper.link_idxs[-1])).default_rotational_inertia(
                ).CalcPrincipalMomentsOfInertia()[0]
        self.sim_sys_consts.v_stiction = constants.v_stiction
        self.sim_sys_consts.g = self.plant.gravity_field().gravity_vector()[-1]*-1
        # Init ctrl_sys_consts
        self.ctrl_sys_consts.I_L = self.plant.get_body(
            BodyIndex(self.paper.link_idxs[-1])).default_rotational_inertia(
                ).CalcPrincipalMomentsOfInertia()[0]
        self.ctrl_sys_consts.v_stiction = constants.v_stiction
        self.ctrl_sys_consts.g = self.plant.gravity_field().gravity_vector()[-1]*-1

        contact_body = self.plant.GetBodyByName(
            manipulator.data["contact_body_name"])
        self.ll_idx = self.paper.link_idxs[-1]
        self.contact_body_idx = int(contact_body.index())

        self._add_non_ctrl_systems(
            timeout=timeout, exit_when_folded=exit_when_folded)
        self._add_ctrl_systems(DT)
        self._wire(DT)
        self._context_specific_init()


    def get_viz_str(self):
        # Regex processing
        ## Remove objects we don't want to visualize
        objects_to_remove = [
            "VectorLogSink",
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

        return viz_str

    def exit_cleanly(self, simulator):
        if self.meshcat is not None:
            self.vis.StopRecording()
            self.vis.PublishRecording()
        
        return self.logger.FindLog(simulator.get_context())


    def run_sim(self, clean_exit=True):
        """
        Run the actual simulation and return the log.

        :param clean_exit: Whether or not to catch and print (but otherwise
                           ignore) any errors that happen during the sim
        """
        # Finalize simulation and visualization
        self.exit_system.set_start_time()
        simulator = pydrake.systems.analysis.Simulator(
            self.diagram, self.diagram_context)
        simulator.Initialize()
        if self.meshcat is not None:
            self.vis.StartRecording()

        if clean_exit:
            try:
                simulator.AdvanceTo(self.TSPAN)
            except sim_exceptions.SimTaskComplete as e:
                print("Successful run!")
                self.success = True
                self.exit_message = repr(e)
                return self.exit_cleanly(simulator)
            except BaseException as e:
                # Has to be BaseException to get KeyboardInterrupt
                print(type(e))
                print(e)
                self.success = False
                self.exit_message = repr(e)
                return self.exit_cleanly(simulator)
        else:
            simulator.AdvanceTo(self.TSPAN)

        if self.meshcat is not None:
            self.vis.StopRecording()
            self.vis.PublishRecording()

        return self.logger.FindLog(simulator.get_context())


    # =========================== PRIVATE FUNCTIONS ===========================

    def _add_mbp_bodies(self, use_true_hinge_diameter):
        """
        Initialize and add to builder systems any systems that introduce new
        bodies to the multibody plant, so this needs to be called before
        `plant.Finalize()`, which is called at the end of the function.
        """
        # Pedestal
        pedestal_instance = pedestal.AddPedestal(self.plant, self.num_links, h_L=self.sim_sys_consts.h_L)

        # Manipulator
        self.manipulator_instance = manipulator.data["add_plant_function"](
            plant=self.plant, m_M=self.sim_sys_consts.m_M,
            r=self.sim_sys_consts.r, mu=self.sim_sys_consts.mu,
            scene_graph=self.scene_graph)

        # Paper
        self.paper = Paper(self.plant, self.scene_graph,
            default_joint_angle=0,
            k_J=self.sim_sys_consts.k_J, b_J=self.sim_sys_consts.b_J,
            m_L=self.sim_sys_consts.m_L, w_L=self.sim_sys_consts.w_L,
            h_L=self.sim_sys_consts.h_L, mu=self.sim_sys_consts.mu,
            num_links=self.num_links,
            use_true_hinge_diameter=use_true_hinge_diameter)
        self.paper.weld_paper_edge(pedestal_instance)

        self.plant.Finalize()

 
    def _add_non_ctrl_systems(self, timeout, exit_when_folded):
        """
        Initialize and add to builder systems that don't add bodies to the
        plant (so they can be called after `plant.Finalize()`) that are also
        not controllers.
        """
        self.builder.AddNamedSystem(
            "q_noise",
            ctrl.aux.NoiseGenerator(
            manipulator.data['nq'], self.noise['q'])
        )
        self.builder.AddNamedSystem(
            "v_noise",
            ctrl.aux.NoiseGenerator(
            manipulator.data['nq'], self.noise['v'])
        )
        self.builder.AddNamedSystem(
            "qv_noise",
            Multiplexer([manipulator.data['nq'], manipulator.data['nq']])
        )
        self.builder.AddNamedSystem(
            "qv_noise_adder",
            Adder(2, 2*manipulator.data['nq'])
        )
        self.builder.AddNamedSystem("qv_noise_demux", Demultiplexer(
            [manipulator.data['nq'], manipulator.data['nq']]))
        # Proprioception
        self.builder.AddNamedSystem(
            "prop",
            perception.proprioception.ProprioceptionSystem(
                m_M=self.ctrl_sys_consts.m_M,
                r=self.ctrl_sys_consts.r,
                mu=self.ctrl_sys_consts.mu
            )
        )

        # Vision
        X_LJ_L = self.paper.joints[0].frame_on_child(
            ).GetFixedPoseInBodyFrame()
        self.vision_processor = perception.vision.VisionProcessor(
            self.ctrl_sys_consts, X_LJ_L=X_LJ_L)
        self.builder.AddNamedSystem("vis_proc", self.vision_processor)
        vision_sys = perception.vision.VisionSystem(
            ll_idx=self.ll_idx,
            contact_body_idx=self.contact_body_idx
        )
        self.builder.AddNamedSystem("vision_no_noise", vision_sys)
        for i in range(vision_sys.num_output_ports()):
            name = vision_sys.get_output_port(i).get_name()
            noise = self.noise[name]
            self.builder.AddNamedSystem(
                name + "_noise",
                ctrl.aux.NoiseGenerator(3,noise)
            )
            self.builder.AddNamedSystem(
                name + "_w_noise",
                Adder(2,3)
            )

        # TODO: reorganize noise
        # Force
        self.builder.AddNamedSystem(
            "measured_contact_force",
            ctrl.aux.NormalForceSelector(
                ll_idx=self.ll_idx,
                contact_body_idx=self.contact_body_idx,
                paper=self.paper
            )
        )

        # Logger
        self.log_wrapper = LogWrapper(
            self.plant.num_bodies(),
            self.contact_body_idx,
            self.paper,
            self.plant,
            tau_O_log=self.tau_O_log,
            F_OT_log=self.F_OT_log,
            F_ON_log=self.F_ON_log
        )
        self.builder.AddNamedSystem("log", self.log_wrapper)

        self.builder.AddNamedSystem(
            "any_contact_calc", 
            ctrl.aux.AnyContactsCalculator(
                self.paper.link_idxs, self.contact_body_idx))
        # Exit system
        link_z = X_LJ_L.translation()[-1]
        z_thresh_tol = self.sim_sys_consts.h_L/10
        z_thresh_offset = 2*link_z + z_thresh_tol
        z_lockdown_thresh_offset = z_thresh_offset + self.sim_sys_consts.h_L
        self.exit_system = ctrl.aux.ExitSystem(
            ll_idx=self.ll_idx,
            paper=self.paper,
            exit_when_folded=exit_when_folded,
            timeout=timeout,
            z_thresh_offset=z_thresh_offset,
            z_lockdown_thresh_offset=z_lockdown_thresh_offset
        )
        self.builder.AddNamedSystem("exit_system", self.exit_system)

        if self.meshcat is not None:
            # Create visualization
            meshcat_params = MeshcatVisualizerParams()
            self.vis = MeshcatVisualizerCpp.AddToBuilder(
                self.builder, self.scene_graph.get_query_output_port(),
                self.meshcat, meshcat_params)

            # End effector visualization
            ee_frame_vis = visualization.FrameVisualizer(
                name="end_effector__no_noise", meshcat=self.meshcat)
            self.builder.AddNamedSystem("end_effector_frame_vis", ee_frame_vis)
            ee_frame_vis.set_animation(self.vis.get_mutable_recording())

            # Link visualization
            link_frame_vis = visualization.FrameVisualizer(
                name="last_link", meshcat=self.meshcat)
            self.builder.AddNamedSystem("link_frame_vis", link_frame_vis)
            link_frame_vis.set_animation(self.vis.get_mutable_recording())

    def _add_inverse_dynamics_ctrl(self):
        options = {
            'model_friction': False,
            'measure_joint_wrench': True,
        }
        self.fold_ctrl = ctrl.inverse_dynamics.InverseDynamicsController(
            sys_consts=self.ctrl_sys_consts, options=options,
            num_links=self.num_links,
            tau_O_log=self.tau_O_log,
            F_OT_log=self.F_OT_log,
            F_ON_log=self.F_ON_log)

        if self.meshcat is not None:
            self.builder.AddNamedSystem(
                "desired_position_XYZ", ctrl.aux.HTNtoXYZ())
            self.builder.AddNamedSystem("desired_pos_adder", Adder(2, 3))

            desired_pos_vis = visualization.FrameVisualizer(
                name="desired position", meshcat=self.meshcat, opacity=0.3)
            self.builder.AddNamedSystem("desired_pos_vis", desired_pos_vis)
            desired_pos_vis.set_animation(self.vis.get_mutable_recording())


    def _add_kinematic_ctrl(self):
        self.fold_ctrl = ctrl.kinematic_ctrlr.KinematicController(
            sys_consts=self.ctrl_sys_consts)


    def _add_impedance_ctrl(self):
        self._add_impedance_n_hat_force_compensation()

        if self.impedance_type == ImpedanceType.OFFLINE_TRAJ:
            setpoint_gen = ctrl.impedance_generators.setpoint_generators.\
                    offline_loader.OfflineTrajLoader(num_links=self.num_links,
                        speed_factor=1)
        elif self.impedance_type == ImpedanceType.LINK_FB:
            setpoint_gen = ctrl.impedance_generators.setpoint_generators.\
                    link_feedback.LinkFeedbackSetpointGenerator(
                        sys_consts=self.ctrl_sys_consts,
                        num_links=self.num_links)
        elif self.impedance_type == ImpedanceType.FORCE_FB:
            k_ = self.impedance_stiffness[-1]
            dist_offset = self.const_ff_Fn/k_
            setpoint_gen = ctrl.impedance_generators.setpoint_generators.\
                    force_feedback.ForceFeedbackSetpointGenerator(
                        sys_consts=self.ctrl_sys_consts,
                        contact_body_idx=self.contact_body_idx,
                        num_links=self.num_links,
                        dist_offset=dist_offset)
            self.setpoint_gen = setpoint_gen
        
        self.fold_ctrl = ctrl.cartesian_impedance.CartesianImpedanceController(
            sys_consts=self.ctrl_sys_consts)
        
        self.builder.AddNamedSystem("K_gen",
            ConstantVectorSource(self.impedance_stiffness))
        self.builder.AddNamedSystem("D_gen",
            ConstantVectorSource(2*np.sqrt(self.impedance_stiffness)))
        self.builder.AddNamedSystem("setpoint_gen", setpoint_gen)
        self.builder.AddNamedSystem("demux_setpoint", Demultiplexer([3,3]))
        if self.meshcat is not None:
            self.setpoint_vis = visualization.FrameVisualizer(
                name="impedance_setpoint", meshcat=self.meshcat, opacity=0.3)
            self.builder.AddNamedSystem("setpoint_vis", self.setpoint_vis)
            self.setpoint_vis.set_animation(self.vis.get_mutable_recording())


    def _add_impedance_n_hat_force_compensation(self):
        """
        If we are using impedance control, initialize and add systems related
        to generating normal compensation.
        """
        if self.n_hat_force_compensation_source == \
                NHatForceCompensationSource.NONE:
            self.builder.AddNamedSystem("ff_wrench_XYZ",
                ConstantVectorSource([0, 0, 0, 0, 0, 0]))
        elif self.n_hat_force_compensation_source == \
                NHatForceCompensationSource.CONTACT_FORCE:
            self.builder.AddNamedSystem(
                "ff_torque_XYZ", ConstantVectorSource([0, 0, 0]))
            self.builder.AddNamedSystem("ff_wrench_XYZ", Multiplexer([3,3]))
            self.builder.AddNamedSystem(
                "F_noise",
                ctrl.aux.NoiseGenerator(3, self.noise["F"])
            )
            self.builder.AddNamedSystem(
                "F_with_noise",
                Adder(2, 3)
            )
        else:
            self.builder.AddNamedSystem("ff_force_HT",
                ConstantVectorSource([0, 0]))
            self.builder.AddNamedSystem("ff_force_HTN", Multiplexer([2, 1]))
            self.builder.AddNamedSystem("ff_force_XYZ", ctrl.aux.HTNtoXYZ())
            self.builder.AddNamedSystem(
                "ff_torque_XYZ", ConstantVectorSource([0, 0, 0]))
            self.builder.AddNamedSystem("ff_wrench_XYZ", Multiplexer([3,3]))

            const_ff_Fn_src = ConstantVectorSource([self.const_ff_Fn])
            if (self.n_hat_force_compensation_source == \
                    NHatForceCompensationSource.PURE_FN):
                self.builder.AddNamedSystem("const_ff_Fn_src", const_ff_Fn_src)
                self.builder.AddNamedSystem(
                    "measured_ff_Fn_noise",
                    ctrl.aux.NoiseGenerator(1, self.noise["Fn"])
                )
                self.builder.AddNamedSystem(
                    "measured_ff_Fn_w_noise",
                    Adder(2, 1)
                )
                ff_force_N = Adder(2, 1)
            elif self.n_hat_force_compensation_source == \
                    NHatForceCompensationSource.CONSTANT:
                ff_force_N = const_ff_Fn_src
            self.builder.AddNamedSystem("ff_force_N", ff_force_N)
            self.builder.AddNamedSystem("ff_force_N_Sat",
                Saturation(min_value=[0],max_value=[500]))

 
    def _add_ctrl_systems(self, DT):
        if self.ctrl_paradigm == CtrlParadigm.INVERSE_DYNAMICS:
            self._add_inverse_dynamics_ctrl()
        elif self.ctrl_paradigm == CtrlParadigm.KINEMATIC:
            self._add_kinematic_ctrl()
        elif self.ctrl_paradigm == CtrlParadigm.IMPEDANCE:
            self._add_impedance_ctrl()
        
        self.builder.AddNamedSystem("fold_ctrl", self.fold_ctrl)
        self.builder.AddNamedSystem("adder",
            Adder(3, manipulator.data['nq']))
        self.builder.AddNamedSystem("tau_g_gain",
            Gain(-1, manipulator.data['nq']))
        self.builder.AddNamedSystem("joint_centering_ctrl",
            ctrl.aux.JointCenteringCtrl())

        # TODO: clean this up
        if self.ctrl_paradigm == CtrlParadigm.INVERSE_DYNAMICS or self.impedance_type == ImpedanceType.FORCE_FB:
            self.builder.AddNamedSystem(
                "pre_contact_ctrl__setpoint_gen",
                ctrl.impedance_generators.setpoint_generators.\
                        offline_loader.OfflineTrajLoader(
                            num_links=self.num_links, speed_factor=10)
            )
            self.builder.AddNamedSystem(
                "pre_contact_ctrl",
                ctrl.cartesian_impedance.CartesianImpedanceController(
                sys_consts=self.ctrl_sys_consts)
            )
            _impedance_stiffness = [40, 40, 40, 400, 400, 400]
            self.builder.AddNamedSystem("pre_contact_ctrl__K_gen",
                ConstantVectorSource(_impedance_stiffness))
            self.builder.AddNamedSystem("pre_contact_ctrl__D_gen",
                ConstantVectorSource(2*np.sqrt(_impedance_stiffness)))
            self.builder.AddNamedSystem("pre_contact_ctrl__ff_Fn",
                ConstantVectorSource([0, 0, 0, 0, 0, 0]))
            self.builder.AddNamedSystem("fake_lockdown_signal", ConstantVectorSource([0]))

        self.builder.AddNamedSystem("ctrl_selector", ctrl.aux.CtrlSelector())

        if DT > 0:
            self.builder.AddNamedSystem("delay",
                DiscreteTimeDelay(DT, 1, manipulator.data['nq']))
            if self.model_impedance_delay:
                if self.ctrl_paradigm == CtrlParadigm.IMPEDANCE:
                    self.builder.AddNamedSystem("x0_delay",
                        DiscreteTimeDelay(0.01, 1, 6))
                    self.builder.AddNamedSystem("F_delay",
                        DiscreteTimeDelay(0.01, 1, 6))


    def _context_specific_init(self):
        self.diagram_context = self.diagram.CreateDefaultContext()
        manipulator.data['set_positions'](
            self.diagram, self.diagram_context, self.plant,
            self.manipulator_instance, self.num_links)
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

    def _wire(self, DT):
        # Set up vision
        self._connect(["plant", "body_poses"], ["vision_no_noise", "poses"])
        self._connect(["plant", "spatial_velocities"],
            ["vision_no_noise", "vels"])

        # Set up vision processor
        vision_sys = self._get_system("vision_no_noise")
        for i in range(vision_sys.num_output_ports()):
            name = vision_sys.get_output_port(i).get_name()
            self._connect(["vision_no_noise", name], [name + "_w_noise", 0])
            self._connect(name + "_noise", [name + "_w_noise", 1])
            self._connect(name + "_w_noise", ["vis_proc", name])

        # Set up proprioception
        self._connect("q_noise", ["qv_noise", 0])
        self._connect("v_noise", ["qv_noise", 1])
        self._connect("qv_noise", ["qv_noise_adder", 0])
        self._connect(self.plant.get_state_output_port(
            self.manipulator_instance), ["qv_noise_adder", 1])
        self._connect("qv_noise_adder", ["prop", "state"])

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

        self._connect(["plant", "contact_results"], "any_contact_calc")
        self._connect(["plant", "body_poses"],["exit_system", "poses"])
        self._connect("any_contact_calc", ["exit_system", "in_contact"])
        self._connect("any_contact_calc", ["log", "any_links_in_contact"])
        self._connect(["vis_proc", "theta_L"], ["exit_system", "theta_L"])
        self._connect(["vis_proc", "d_theta_L"], ["exit_system", "d_theta_L"])
        if self.impedance_type != ImpedanceType.NONE:
            self._connect("exit_system", ["fold_ctrl", "lockdown_signal"])
        self._connect("exit_system", ["log", "alive_signal"])

        # Set up visualization
        if self.meshcat is not None:
            self._connect(["vision_no_noise", "pose_M_translational"],
                ["end_effector_frame_vis", "pos"])
            self._connect(["vision_no_noise", "pose_M_rotational"],
                ["end_effector_frame_vis", "rot"])
            self._connect(["vision_no_noise", "pose_L_translational"],
                ["link_frame_vis", "pos"])
            self._connect(["vision_no_noise", "pose_L_rotational"],
                ["link_frame_vis", "rot"])

        # Force feedback
        self._connect(["plant", "contact_results"],
            "measured_contact_force")

        if (self.ctrl_paradigm == CtrlParadigm.INVERSE_DYNAMICS):
            skipped_ports = {"in_contact", "v_LN", "v_MN"}
            self._connect_all_outputs(
                "vis_proc", "fold_ctrl", skipped_ports=skipped_ports)

            self._connect(
                "pose_L_rotational_w_noise",
                ["fold_ctrl", "pose_L_rotational"]
            )
            self._connect(
                "pose_L_translational_w_noise",
                ["fold_ctrl", "pose_L_translational"]
            )

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
                # This should be w/o noise, because we're just converting frames
                # TODO: double check
                self._connect(
                    ["vision_no_noise", "pose_L_translational"],
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
                self._connect(k + "_w_noise", ["fold_ctrl", k])
            for k in ["M", "J", "Jdot_qdot", "Cv"]:
                self._connect(["prop", k], ["fold_ctrl", k])
            self._connect("K_gen", ["fold_ctrl", "K"])
            self._connect("D_gen", ["fold_ctrl", "D"])
            if self.model_impedance_delay:
                self._connect(["setpoint_gen", "x0"], "x0_delay")
                self._connect("x0_delay", ["fold_ctrl", "x0"])
            else:
                self._connect(["setpoint_gen", "x0"], ["fold_ctrl", "x0"])
            self._connect(["setpoint_gen", "dx0"], ["fold_ctrl", "dx0"])
            if self.n_hat_force_compensation_source == \
                    NHatForceCompensationSource.CONTACT_FORCE:
                self._connect("ff_torque_XYZ", ["ff_wrench_XYZ", 0])
                self._connect(
                    ["measured_contact_force", "contact_force"],
                    ["F_with_noise", 0])
                self._connect(
                    "F_noise",
                    ["F_with_noise", 1])
                self._connect(
                    "F_with_noise",
                    ["ff_wrench_XYZ", 1])
            elif (self.n_hat_force_compensation_source == \
                    NHatForceCompensationSource.PURE_FN) or \
                    (self.n_hat_force_compensation_source == \
                    NHatForceCompensationSource.CONSTANT):
                self._connect("ff_force_HTN", ["ff_force_XYZ", "HTN"])
                self._connect(["vis_proc", "T_hat"], ["ff_force_XYZ", "T_hat"])
                self._connect(["vis_proc", "N_hat"], ["ff_force_XYZ", "N_hat"])
                self._connect("ff_torque_XYZ", ["ff_wrench_XYZ", 0])
                self._connect("ff_force_XYZ", ["ff_wrench_XYZ", 1])
                self._connect("ff_force_HT", ["ff_force_HTN", 0])
                self._connect("ff_force_N", "ff_force_N_Sat")
                self._connect("ff_force_N_Sat", ["ff_force_HTN", 1])

                if self.n_hat_force_compensation_source == \
                        NHatForceCompensationSource.PURE_FN:
                    self._connect(["measured_contact_force", "F_N"],
                        ["measured_ff_Fn_w_noise", 0])
                    self._connect("measured_ff_Fn_noise",
                        ["measured_ff_Fn_w_noise", 1])
                    self._connect("measured_ff_Fn_w_noise", ["ff_force_N", 0])
                    self._connect("const_ff_Fn_src", ["ff_force_N", 1])
            if self.model_impedance_delay:
                self._connect("ff_wrench_XYZ", "F_delay")
                self._connect("F_delay", ["fold_ctrl", "feedforward_wrench"])
            else:
                self._connect("ff_wrench_XYZ", ["fold_ctrl", "feedforward_wrench"])
            
            self._connect(["setpoint_gen", "x0"], "demux_setpoint")
            if self.meshcat is not None:
                self._connect(["fold_ctrl", "adjusted_x0_pos"],
                    ["setpoint_vis", "pos"])
                self._connect(["fold_ctrl", "adjusted_x0_rot"],
                    ["setpoint_vis", "rot"])

            if self.impedance_type == ImpedanceType.LINK_FB:
                self._connect("pose_L_translational_w_noise",
                    ["setpoint_gen", "pose_L_translational"])
                self._connect("pose_L_rotational_w_noise",
                    ["setpoint_gen", "pose_L_rotational"])
            elif self.impedance_type == ImpedanceType.FORCE_FB:
                self._connect("qv_noise_adder", "qv_noise_demux")
                self._connect(["qv_noise_demux", 0], ["setpoint_gen", "q"])
                self._connect(
                    ["measured_contact_force", "contact_force"],
                    ["setpoint_gen", "F"])

        # Controller connections
        ## Controller mux
        self._connect(["vis_proc", "in_contact"],
            ["ctrl_selector", "in_contact"])
        self._connect(["fold_ctrl", "tau_out"],
            ["ctrl_selector", "contact_ctrl"])
        if self.ctrl_paradigm == CtrlParadigm.INVERSE_DYNAMICS or self.impedance_type == ImpedanceType.FORCE_FB:
            self._connect("pre_contact_ctrl__ff_Fn", ["pre_contact_ctrl", "feedforward_wrench"])
            for k in ["pose_M_translational", "pose_M_rotational",
                    "vel_M_translational", "vel_M_rotational"]:
                self._connect(k + "_w_noise", ["pre_contact_ctrl", k])
            for k in ["M", "J", "Jdot_qdot", "Cv"]:
                self._connect(["prop", k], ["pre_contact_ctrl", k])
            self._connect("pre_contact_ctrl__K_gen", ["pre_contact_ctrl", "K"])
            self._connect("pre_contact_ctrl__D_gen", ["pre_contact_ctrl", "D"])
            self._connect(
                ["pre_contact_ctrl__setpoint_gen", "x0"],
                ["pre_contact_ctrl", "x0"]
            )
            self._connect(
                ["pre_contact_ctrl__setpoint_gen", "dx0"],
                ["pre_contact_ctrl", "dx0"]
            )
            self._connect(["pre_contact_ctrl", "tau_out"],
                ["ctrl_selector", "pre_contact_ctrl"])
            self._connect("fake_lockdown_signal", ["pre_contact_ctrl", "lockdown_signal"])
        else:
            self._connect(["fold_ctrl", "tau_out"],
                ["ctrl_selector", "pre_contact_ctrl"])
        self._connect("ctrl_selector", ["log", "tau_ctrl"])
        self._connect(["fold_ctrl", "tau_out"], ["log", "tau_contact_ctrl"])

        ## Set up joint centering control
        self._connect_all_inputs("prop", "joint_centering_ctrl")

        ## Wire gravity compensation + final adder
        self._connect(["prop", "tau_g"], ["tau_g_gain", 0])
        self._connect("joint_centering_ctrl", ["adder", 0])
        self._connect("ctrl_selector", ["adder", 1])
        self._connect("tau_g_gain", ["adder", 2])

        if DT > 0:
            self._connect("adder", "delay")
            self._connect("delay", ["plant", "panda_actuation"])
        else:
            self._connect("adder", ["plant", "panda_actuation"])

        self._connect("adder", ["log", "tau_out"])

        # Visualization and logging
        self.logger = LogVectorOutput(
            self.log_wrapper.get_output_port(), self.builder)

        # Build diagram
        self.diagram = self.builder.Build()
