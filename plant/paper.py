"""Contains paper modeling class as well as paper-related constants."""

# Standard imports
import numpy as np

# Drake imports
import pydrake
from pydrake.all import RigidTransform, RotationMatrix, CollisionFilterDeclaration
from pydrake.multibody.tree import BodyIndex, SpatialInertia, UnitInertia, RevoluteSpring

# Imports of other project files
import constants
from constants import THIN_PLYWOOD_THICKNESS, PLYWOOD_LENGTH

# Other imports
from enum import Enum

class NumLinks(Enum):
    TWO = 2
    FOUR = 4

PAPER_X_DIM = PLYWOOD_LENGTH
PAPER_Z_DIM = THIN_PLYWOOD_THICKNESS

settling_time = 2
class Paper:
    """Model of paper dynamics."""
    name = "paper"
    x_dim = PAPER_X_DIM
    z_dim = PAPER_Z_DIM
    density = 500

    hinge_diameter = (3/32)*constants.IN_TO_M

    def __init__(self, plant, scene_graph, num_links:NumLinks, default_joint_angle=-np.pi/60,
                 damping=1e-5, stiffness=1e-3):
        # Drake objects
        self.plant = plant
        self.scene_graph = scene_graph

        # Geometric and physical quantities
        self.num_links = num_links
        # PROGRAMMING: Refactor constants.FRICTION to be constants.mu
        self.mu = constants.FRICTION
        self.default_joint_angle = default_joint_angle
        self.link_width = PLYWOOD_LENGTH/2
        self.y_dim = self.link_width * num_links.value
        self.link_mass = self.x_dim*self.y_dim*self.z_dim*self.density

        # Lists of internal Drake objects
        self.link_idxs = []
        self.joints = []

        self.damping = damping
        self.stiffness = stiffness
        self.instance = self.plant.AddModelInstance(self.name)
        for link_num in range(self.num_links.value):
            # Initialize bodies and instances
            
            paper_body = self.plant.AddRigidBody(
                self.name + "_body" + str(link_num),
                self.instance,
                SpatialInertia(mass=self.link_mass,
                               # CoM at origin of body frame
                               p_PScm_E=np.array([0., 0., 0.]),
                               # Default moment of inertia for a solid box
                               G_SP_E=UnitInertia.SolidBox(
                                   self.x_dim, self.link_width, self.z_dim))
            )

            if self.plant.geometry_source_is_registered():
                # Set a box with link dimensions for collision geometry
                self.plant.RegisterCollisionGeometry(
                    paper_body,
                    RigidTransform(),  # Pose in body frame
                    pydrake.geometry.Box(
                        self.x_dim, self.link_width, self.z_dim),  # Actual shape
                    self.name + "_body" + str(link_num), pydrake.multibody.plant.CoulombFriction(
                        self.mu, self.mu)  # Friction parameters
                )

                # Set Set a box with link dimensions for visual geometry
                self.plant.RegisterVisualGeometry(
                    paper_body,
                    RigidTransform(),
                    pydrake.geometry.Box(
                        self.x_dim, self.link_width, self.z_dim),
                    self.name + "_body" + str(link_num),
                    [0, 1, 0, 1])  # RGBA color

            # Operations between adjacent links
            if link_num > 0:
                # Get bodies
                paper1_body = self.plant.get_body(
                    BodyIndex(self.link_idxs[-1]))
                paper2_body = self.plant.GetBodyByName(
                    self.name + "_body" + str(link_num), self.instance)

                # Set up joints
                paper1_hinge_frame = pydrake.multibody.tree.FixedOffsetFrame(
                    "paper_hinge_frame",
                    paper1_body,
                    RigidTransform(RotationMatrix(),
                        [
                            0,
                            (self.link_width+self.hinge_diameter)/2,
                            (self.z_dim+self.hinge_diameter)/2
                        ])
                )
                self.plant.AddFrame(paper1_hinge_frame)
                paper2_hinge_frame = pydrake.multibody.tree.FixedOffsetFrame(
                    "paper_hinge_frame",
                    paper2_body,
                    RigidTransform(RotationMatrix(),
                        [
                            0,
                            -(self.link_width+self.hinge_diameter)/2,
                            (self.z_dim+self.hinge_diameter)/2
                        ])
                )
                self.plant.AddFrame(paper2_hinge_frame)

                joint = self.plant.AddJoint(pydrake.multibody.tree.RevoluteJoint(
                    "paper_hinge_" + str(link_num),
                    paper1_hinge_frame,
                    paper2_hinge_frame,
                    [1, 0, 0],
                    damping=damping))

                if isinstance(default_joint_angle, list):
                    joint.set_default_angle(
                        self.default_joint_angle[link_num])
                else:
                    joint.set_default_angle(self.default_joint_angle)

                self.plant.AddForceElement(
                    RevoluteSpring(joint, 0, self.stiffness))
                self.joints.append(joint)
                # Ignore collisions between adjacent links
                geometries = self.plant.CollectRegisteredGeometries(
                    [paper1_body, paper2_body])
                self.scene_graph.collision_filter_manager().Apply(
                    CollisionFilterDeclaration()
                        .ExcludeWithin(geometries))
            self.link_idxs.append(int(paper_body.index()))

    def weld_paper_edge(self, pedestal_y_dim, pedestal_z_dim):
        """
        Fixes an edge of the paper to the pedestal
        """
        # Fix paper to object
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.get_body(BodyIndex(self.link_idxs[0])).body_frame(),
            RigidTransform(RotationMatrix(
            ), [0, -(pedestal_y_dim/2-self.y_dim/4), pedestal_z_dim+self.z_dim/2])
        )
