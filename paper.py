# Standard imports
import numpy as np

# Drake imports
import pydrake
from pydrake.all import RigidTransform, RotationMatrix, LinearBushingRollPitchYaw
from pydrake.multibody.tree import SpatialInertia, UnitInertia

# Imports of other project files
import constants

PAPER_WIDTH = 11*constants.IN_TO_M
PAPER_DEPTH = 8.5*constants.IN_TO_M
PAPER_HEIGHT = 0.0005


class Paper:
    """Model of paper dynamics."""
    name = "paper"
    width = PAPER_WIDTH
    depth = PAPER_DEPTH
    height = PAPER_HEIGHT

    # Source:
    # https://www.jampaper.com/paper-weight-chart.asp
    true_height = 0.0097e-3
    density = 80/1000

    # Source:
    # https://smartech.gatech.edu/bitstream/handle/1853/5562/jones_ar.pdf
    # http://www.mate.tue.nl/mate/pdfs/10509.pdf
    youngs_modulus = 6*1e9  # Convert to n/m^2

    def __init__(self, plant, scene_graph, num_links, mu=5.0, default_joint_angle=-np.pi/60, damping=1e-5,
                 stiffness=1e-3):
        # Initialize parameters
        self.plant = plant
        self.scene_graph = scene_graph
        self.num_links = num_links
        self.mu = constants.FRICTION
        self.default_joint_angle = default_joint_angle
        self.damping = damping
        self.stiffness = stiffness
        self.torque_stiffness_constants = np.zeros([3, 1])
        self.torque_stiffness_constants[0] = stiffness
        self.torque_damping_constants = np.zeros([3, 1])
        self.torque_damping_constants[0] = damping
        self.force_stiffness_constants = np.zeros([3, 1])
        self.force_damping_constants = np.zeros([3, 1])

        self.link_width = self.width/self.num_links
        self.link_mass = self.density*self.width*self.width/self.num_links

        L = self.width/num_links
        I = self.depth*self.true_height**3/12
        # Stiffness = 3*(youngs modulus)*I/(Length)
        physical_stiffness_N_p_m = 3*self.youngs_modulus*I/L**3
        physical_N_p_rad = physical_stiffness_N_p_m*L**2

        # Use this stiffness only if simulating with a shorter DT
        # self.stiffness = physical_N_p_rad

        self.link_instances = []
        for link_num in range(self.num_links):
            # Initalize bodies and instances
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
                    [0.9, 0.9, 0.9, 1.0])  # RBGA color

            # Operations between adjacent links
            if link_num > 0:
                # Get bodies
                paper1_body = self.plant.GetBodyByName(
                    "paper_body", self.link_instances[-1])
                paper2_body = self.plant.GetBodyByName(
                    "paper_body", paper_instance)

                # Set up joint actuators
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

                joint = self.plant.AddJoint(pydrake.multibody.tree.RevoluteJoint(
                    "paper_hinge",
                    paper1_hinge_frame,
                    paper2_hinge_frame,
                    [1, 0, 0]))

                if type(default_joint_angle) is list:
                    joint.set_default_angle(self.default_joint_angle[link_num])
                else:
                    joint.set_default_angle(self.default_joint_angle)

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

            self.link_instances.append(paper_instance)

    def get_free_edge_instance(self):
        return self.link_instances[-2]

    def weld_paper_edge(self, pedestal_width, pedestal_height):
        # Fix paper to object
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetBodyByName(
                "paper_body", self.link_instances[0]).body_frame(),
            RigidTransform(RotationMatrix(
            ), [0, -(pedestal_width/2-self.width/4), pedestal_height+self.height/2])
        )
