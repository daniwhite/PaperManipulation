"""Contains paper modeling class as well as paper-related constants."""

# Standard imports
import numpy as np

# Drake imports
import pydrake
from pydrake.all import RigidTransform, RotationMatrix, LinearBushingRollPitchYaw
from pydrake.multibody.tree import BodyIndex, SpatialInertia, UnitInertia

# Imports of other project files
import constants

PAPER_WIDTH = 11*constants.IN_TO_M
PAPER_DEPTH = 8.5*constants.IN_TO_M
PAPER_HEIGHT = 0.01

JOINT_TYPES = {"NATURAL", "FIXED", "DRIVEN"}


class Paper:
    """Model of paper dynamics."""
    name = "paper"
    width = PAPER_WIDTH
    depth = PAPER_DEPTH
    height = PAPER_HEIGHT

    # Source:
    # https://www.jampaper.com/paper-weight-chart.asp
    true_height = height  # 0.0097e-3
    density = 80/1000

    # Source:
    # https://smartech.gatech.edu/bitstream/handle/1853/5562/jones_ar.pdf
    # http://www.mate.tue.nl/mate/pdfs/10509.pdf
    youngs_modulus = 6*1e9  # Convert to n/m^2

    def __init__(self, plant, scene_graph, num_links, joint_type, default_joint_angle=-np.pi/60,
                 damping=1e-5, stiffness=1e-3):
        if joint_type not in JOINT_TYPES:
            raise ValueError("joint_type not in {}".format(JOINT_TYPES))
        else:
            self.joint_type = joint_type
        # Initialize parameters
        self.plant = plant
        self.scene_graph = scene_graph
        self.num_links = num_links
        self.mu = constants.FRICTION
        self.default_joint_angle = default_joint_angle
        self.link_width = self.width/self.num_links
        self.link_mass = 0.1  # self.density*self.width*self.width/self.num_links

        if self.joint_type == "NATURAL":
            self.damping = damping
            self.stiffness = stiffness
            self.torque_stiffness_constants = np.zeros([3, 1])
            self.torque_stiffness_constants[0] = stiffness
            self.torque_damping_constants = np.zeros([3, 1])
            self.torque_damping_constants[0] = damping
            self.force_stiffness_constants = np.zeros([3, 1])
            self.force_damping_constants = np.zeros([3, 1])

            # This is hypothetically how we should be able to derive the stiffness
            # of the links...but it produces way too large values.
            # L = self.width/num_links
            # I = self.depth*self.true_height**3/12
            # # Stiffness = 3*(youngs modulus)*I/(Length)
            # physical_stiffness_N_p_m = 3*self.youngs_modulus*I/L**3
            # physical_N_p_rad = physical_stiffness_N_p_m*L**2

            # Use this stiffness only if simulating with a shorter DT
            # self.stiffness = physical_N_p_rad

        # Body indices of each link. Used to index into log outputs, or use BodyIndex to get the
        # body with plant.get_body
        self.link_idxs = []
        for link_num in range(self.num_links):
            # Initialize bodies and instances
            paper_instance = self.plant.AddModelInstance(
                self.name + str(link_num))
            paper_body = self.plant.AddRigidBody(
                self.name + "_body",
                paper_instance,
                SpatialInertia(mass=self.link_mass,
                               # CoM at origin of body frame
                               p_PScm_E=np.array([0., 0., 0.]),
                               # Default moment of inertia for a solid box
                               G_SP_E=UnitInertia.SolidBox(
                                   self.width, self.link_width, self.true_height))
            )

            if self.plant.geometry_source_is_registered():
                # Set a box with link dimensions for collision geometry
                self.plant.RegisterCollisionGeometry(
                    paper_body,
                    RigidTransform(),  # Pose in body frame
                    pydrake.geometry.Box(
                        self.width, self.link_width, self.height),  # Actual shape
                    self.name + "_body", pydrake.multibody.plant.CoulombFriction(
                        self.mu, self.mu)  # Friction parameters
                )

                # Set Set a box with link dimensions for visual geometry
                self.plant.RegisterVisualGeometry(
                    paper_body,
                    RigidTransform(),
                    pydrake.geometry.Box(
                        self.width, self.link_width, self.height),
                    self.name + "_body",
                    [0, 1, 0, 1])  # 0.9, 0.9, 0.9, 1.0])  # RGBA color

            # Operations between adjacent links
            if link_num > 0:
                # Get bodies
                paper1_body = self.plant.get_body(
                    BodyIndex(self.link_idxs[-1]))
                paper2_body = self.plant.GetBodyByName(
                    "paper_body", paper_instance)

                # Set up joints
                paper1_hinge_frame = pydrake.multibody.tree.FixedOffsetFrame(
                    "paper_hinge_frame",
                    paper1_body,
                    RigidTransform(RotationMatrix(), [0,
                                                      self.link_width/2+constants.EPSILON/2,
                                                      0.5*self.height]))
                self.plant.AddFrame(paper1_hinge_frame)
                paper2_hinge_frame = pydrake.multibody.tree.FixedOffsetFrame(
                    "paper_hinge_frame",
                    paper2_body,
                    RigidTransform(RotationMatrix(), [0,
                                                      (-self.link_width/2 +
                                                       constants.EPSILON/2),
                                                      0.5*self.height]))
                self.plant.AddFrame(paper2_hinge_frame)

                if self.joint_type == "FIXED":
                    if isinstance(default_joint_angle, list):
                        raise NotImplementedError

                    RT = RigidTransform()
                    RT.set_rotation(RotationMatrix.MakeXRotation(
                        self.default_joint_angle))

                    joint = self.plant.AddJoint(pydrake.multibody.tree.WeldJoint(
                        "paper_hinge",
                        paper1_hinge_frame,
                        paper2_hinge_frame,
                        RT))
                else:
                    joint = self.plant.AddJoint(pydrake.multibody.tree.RevoluteJoint(
                        "paper_hinge",
                        paper1_hinge_frame,
                        paper2_hinge_frame,
                        [1, 0, 0]))

                    if isinstance(default_joint_angle, list):
                        joint.set_default_angle(
                            self.default_joint_angle[link_num])
                    else:
                        joint.set_default_angle(self.default_joint_angle)

                    # Set up joint angle
                    self.plant.AddForceElement(LinearBushingRollPitchYaw(
                        paper1_hinge_frame, paper2_hinge_frame,
                        self.torque_stiffness_constants,
                        self.torque_damping_constants,
                        self.force_stiffness_constants,
                        self.force_damping_constants))

                # Ignore collisions between adjacent links
                geometries = self.plant.CollectRegisteredGeometries(
                    [paper1_body, paper2_body])
                self.scene_graph.ExcludeCollisionsWithin(geometries)

            self.link_idxs.append(int(paper_body.index()))

    def post_finalize_steps(self, builder):
        """
        Any initialization steps that need to be done after `Finalize` is called.
        Currently, just initializes controllers for paper joints if `joint_type` is `DRIVEN`.
        """
        if self.joint_type != "DRIVEN":
            return
        raise NotImplementedError
        # stuff copied over from fab8b02
        # PROGRAMMING: Implement speed controllers at joints
        # self.paper_ctrlrs = []
        # for paper_instance in self.link_instances[1:]:
        #     paper_ctrlr = PidController(kp=[[self.stiffness]], ki=[
        #                                 [0]], kd=[[self.damping]])
        #     builder.AddSystem(paper_ctrlr)
        #     self.paper_ctrlrs.append(paper_ctrlr)
        #     builder.Connect(paper_ctrlr.get_output_port(),
        #                     self.plant.get_actuation_input_port(paper_instance))
        #     builder.Connect(self.plant.get_state_output_port(
        #         paper_instance), paper_ctrlr.get_input_port_estimated_state())

    def context_dependent_steps(self, diagram, diagram_context):
        """
        Any initialization steps require a context.
        """
        # for paper_ctrlr in self.paper_ctrlrs:
        #     paper_ctrlr_context = diagram.GetMutableSubsystemContext(
        #         paper_ctrlr, diagram_context)
        #     paper_ctrlr.get_input_port_desired_state().FixValue(
        #         paper_ctrlr_context, [0, 0])

    def get_free_edge_idx(self):
        """
        Returns the model instance corresponding to the edge of the paper which
        is not affixed to the pedestal.
        """
        return self.link_idxs[-1]

    def weld_paper_edge(self, pedestal_width, pedestal_height):
        """
        Fixes an edge of the paper to the pedestal
        """
        # Fix paper to object
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.get_body(BodyIndex(self.link_idxs[0])).body_frame(),
            RigidTransform(RotationMatrix(
            ), [0, -(pedestal_width/2-self.width/4), pedestal_height+self.height/2])
        )
