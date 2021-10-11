"""Contains paper modeling class as well as paper-related constants."""

# Standard imports
import numpy as np

# Drake imports
import pydrake
from pydrake.all import RigidTransform, RotationMatrix
from pydrake.multibody.tree import BodyIndex, SpatialInertia, UnitInertia, RevoluteSpring

# Imports of other project files
import constants

PAPER_WIDTH = 20*constants.IN_TO_M
PAPER_DEPTH = 8.5*constants.IN_TO_M
PAPER_HEIGHT = 0.01


class Paper:
    """Model of paper dynamics."""
    name = "paper"
    width = PAPER_WIDTH
    depth = PAPER_DEPTH
    height = PAPER_HEIGHT

    def __init__(self, plant, scene_graph, num_links, default_joint_angle=-np.pi/60,
                 damping=1e-5, stiffness=1e-3):
        # Drake objects
        self.plant = plant
        self.scene_graph = scene_graph

        # Geometric and physical quantities
        self.num_links = num_links
        # PROGRAMMING: Refactor constants.FRICTION to be constants.mu
        self.mu = constants.FRICTION
        self.default_joint_angle = default_joint_angle
        self.link_width = self.width/self.num_links
        self.link_mass = 0.1

        # Lists of internal Drake objects
        self.link_idxs = []
        self.joints = []

        self.damping = damping
        self.stiffness = stiffness
        self.instance = self.plant.AddModelInstance(self.name)
        for link_num in range(self.num_links):
            # Initialize bodies and instances
            
            paper_body = self.plant.AddRigidBody(
                self.name + "_body" + str(link_num),
                self.instance,
                SpatialInertia(mass=self.link_mass,
                               # CoM at origin of body frame
                               p_PScm_E=np.array([0., 0., 0.]),
                               # Default moment of inertia for a solid box
                               G_SP_E=UnitInertia.SolidBox(
                                   self.width, self.link_width, self.height))
            )

            if self.plant.geometry_source_is_registered():
                # Set a box with link dimensions for collision geometry
                self.plant.RegisterCollisionGeometry(
                    paper_body,
                    RigidTransform(),  # Pose in body frame
                    pydrake.geometry.Box(
                        self.width, self.link_width, self.height),  # Actual shape
                    self.name + "_body" + str(link_num), pydrake.multibody.plant.CoulombFriction(
                        self.mu, self.mu)  # Friction parameters
                )

                # Set Set a box with link dimensions for visual geometry
                self.plant.RegisterVisualGeometry(
                    paper_body,
                    RigidTransform(),
                    pydrake.geometry.Box(
                        self.width, self.link_width, self.height),
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
                    RigidTransform(RotationMatrix(), [0,
                                                      self.link_width/2,
                                                      0]))
                self.plant.AddFrame(paper1_hinge_frame)
                paper2_hinge_frame = pydrake.multibody.tree.FixedOffsetFrame(
                    "paper_hinge_frame",
                    paper2_body,
                    RigidTransform(RotationMatrix(), [0,
                                                      (-self.link_width/2 +
                                                       0),
                                                      0]))
                self.plant.AddFrame(paper2_hinge_frame)

                joint = self.plant.AddJoint(pydrake.multibody.tree.RevoluteJoint(
                    "paper_hinge",
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
                self.scene_graph.ExcludeCollisionsWithin(geometries)
            self.link_idxs.append(int(paper_body.index()))

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
