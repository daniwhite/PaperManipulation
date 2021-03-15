"""Functions for creating and controlling the finger manipulator."""

# Standard imports
import constants
import pedestal
from pydrake.multibody.tree import SpatialInertia, UnitInertia
import numpy as np

# Drake imports
import pydrake
from pydrake.all import RigidTransform, SpatialVelocity, ProximityProperties, ContactResults

RADIUS = 0.01
VOLUME = (4/3)*RADIUS**3*np.pi
MASS = VOLUME*1e3  # Assume finger is made of water

# Finger initial position
INIT_Y = pedestal.PEDESTAL_WIDTH/2 + RADIUS + constants.EPSILON
INIT_Z = 0


def AddFinger(plant):
    """Adds the manipulator."""
    finger = plant.AddModelInstance("finger")

    # Add false bodies for control joints
    empty_inertia = SpatialInertia(0, [0, 0, 0], UnitInertia(0, 0, 0))
    plant.AddRigidBody("false_body1", finger, empty_inertia)
    plant.AddRigidBody("false_body2", finger, empty_inertia)

    # Initialize finger body
    finger_body = plant.AddRigidBody(
        "finger_body", finger,
        pydrake.multibody.tree.SpatialInertia(
            mass=MASS,
            p_PScm_E=np.array([0., 0., 0.]),
            G_SP_E=pydrake.multibody.tree.UnitInertia(1.0, 1.0, 1.0)))

    # Register geometry
    if plant.geometry_source_is_registered():
        col_geom = plant.RegisterCollisionGeometry(
            finger_body, RigidTransform(),
            pydrake.geometry.Sphere(RADIUS),
            "finger_body",
            pydrake.multibody.plant.CoulombFriction(constants.FRICTION, constants.FRICTION))
        plant.RegisterVisualGeometry(
            finger_body,
            RigidTransform(),
            pydrake.geometry.Sphere(RADIUS),
            "finger_body",
            [.9, .5, .5, 1.0])  # Color

    # Add control inputs for the three DOF
    # Linear y control
    finger_y = plant.AddJoint(pydrake.multibody.tree.PrismaticJoint(
        "finger_y",
        plant.world_frame(),
        plant.GetFrameByName("false_body1"), [0, 1, 0], -1, 1))
    plant.AddJointActuator("finger_y", finger_y)
    finger_y.set_default_translation(INIT_Y)
    # Linear z control
    finger_z = plant.AddJoint(pydrake.multibody.tree.PrismaticJoint(
        "finger_z",
        plant.GetFrameByName("false_body1"),
        plant.GetFrameByName("false_body2"), [0, 0, 1], -1, 1))
    finger_z.set_default_translation(INIT_Z)
    plant.AddJointActuator("finger_z", finger_z)
    # Rotational x control
    finger_x = plant.AddJoint(pydrake.multibody.tree.RevoluteJoint(
        "finger_x",
        plant.GetFrameByName("false_body2"),
        plant.GetFrameByName("finger_body"),
        [1, 0, 0],
        damping=0
    ))
    finger_x.set_default_angle(0)
    plant.AddJointActuator("finger_x", finger_x)

    return finger, finger_body, col_geom


class FingerController(pydrake.systems.framework.LeafSystem):
    """Base class for implementing a controller at the finger."""

    def __init__(self, finger_idx, ll_idx):
        pydrake.systems.framework.LeafSystem.__init__(self)

        self.DeclareAbstractInputPort(
            "poses", pydrake.common.value.AbstractValue.Make([RigidTransform(), RigidTransform()]))
        self.DeclareAbstractInputPort(
            "vels", pydrake.common.value.AbstractValue.Make([SpatialVelocity(), SpatialVelocity()]))
        self.DeclareAbstractInputPort(
            "contact_results",
            pydrake.common.value.AbstractValue.Make(ContactResults()))
        self.DeclareVectorOutputPort(
            "finger_actuation", pydrake.systems.framework.BasicVector(3),
            self.CalcOutput)

        self.finger_idx = finger_idx
        self.ll_idx = ll_idx
        self.debug = {}
        self.debug['times'] = []

    def GetForces(self, poses, vels, contact_point, slip_speed, pen_depth, N_hat):
        """
        Should be overloaded to return [Fy, Fz, tau] to move manipulator (not including gravity
        compensation.)
        """
        raise NotImplementedError()

    def CalcOutput(self, context, output):
        """
        Feeds inputs to controller and packs outputs. Also add gravity compensation.
        """

        # Get inputs
        poses = self.get_input_port(0).Eval(context)
        vels = self.get_input_port(1).Eval(context)

        contact_results = self.get_input_port(2).Eval(context)
        contact_point = None
        slip_speed = None
        pen_depth = None
        N_hat = None
        for i in range(contact_results.num_point_pair_contacts()):
            point_pair_contact_info = \
                contact_results.point_pair_contact_info(i)

            a_idx = int(point_pair_contact_info.bodyA_index())
            b_idx = int(point_pair_contact_info.bodyB_index())

            if ((a_idx == self.ll_idx) and (b_idx == self.finger_idx) or
                    (a_idx == self.finger_idx) and (b_idx == self.ll_idx)):
                contact_point = point_pair_contact_info.contact_point()
                slip_speed = point_pair_contact_info.slip_speed()
                pen_point_pair = point_pair_contact_info.point_pair()
                pen_depth = pen_point_pair.depth
                # PROGRAMMING: check sign
                N_hat = np.expand_dims(pen_point_pair.nhat_BA_W, 1)

        self.debug['times'].append(context.get_time())

        [fy, fz, tau] = self.GetForces(
            poses, vels, contact_point, slip_speed, pen_depth, N_hat)
        fz += MASS*constants.g

        output.SetFromVector([fy, fz, tau])


class BlankController(FingerController):
    """Fold paper with feedback on position of the past link"""

    # Making these parameters keywords means that
    def __init__(self, finger_idx, ll_idx):
        super().__init__(finger_idx, ll_idx)

    def GetForces(self, poses, vels, contact_point, slip_speed, pen_depth, N_hat):
        return [0, 0, 0]
