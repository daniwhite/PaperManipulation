"""Functions for creating and controlling the finger manipulator."""

# Standard imports
import constants
import pedestal
from pydrake.multibody.tree import SpatialInertia, UnitInertia, JacobianWrtVariable
from pydrake.all import BasicVector
import numpy as np
from collections import defaultdict

# Drake imports
import pydrake
from pydrake.all import FindResourceOrThrow, RigidTransform, RotationMatrix


def AddArm(plant, scene_graph=None):
    """
    Creates the panda arm.
    """
    parser = pydrake.multibody.parsing.Parser(plant, scene_graph)
    arm_instance = parser.AddModelFromFile(FindResourceOrThrow(
        "drake/manipulation/models/franka_description/urdf/panda_arm.urdf"))

    # Weld pedestal to world
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("panda_link0", arm_instance),
        RigidTransform(RotationMatrix(), [0, pedestal.PEDESTAL_DEPTH*3, 0])
    )

    finger = plant.AddModelInstance("finger")

    RADIUS = 0.03
    VOLUME = (4/3)*RADIUS**3*np.pi
    MASS = VOLUME*1e3  # Assume finger is made of water

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

    plant.WeldFrames(
        plant.GetFrameByName("panda_link8", arm_instance),
        finger_body.body_frame(),
        RigidTransform(RotationMatrix(), [0, 0, 0])
    )

    for i in range(9):
        panda_body = plant.GetBodyByName("panda_link" + str(i), arm_instance)
        geometries = plant.CollectRegisteredGeometries(
            [panda_body, finger_body])
        scene_graph.ExcludeCollisionsWithin(geometries)
    return arm_instance


class ArmForceController(pydrake.systems.framework.LeafSystem):
    """Base class for implementing a controller at the finger."""

    def __init__(self, plant, q_idxs):
        pydrake.systems.framework.LeafSystem.__init__(self)

        self.plant = plant
        self.arm_instance = self.plant.GetModelInstanceByName("panda")
        # TODO: fix
        self.nq_arm = 7  # self.plant.num_positions(arm_instance)
        self.q_idxs = q_idxs

        self.DeclareVectorInputPort("q", BasicVector(self.nq_arm*2))
        # self.DeclareAbstractInputPort(
        #     "poses", pydrake.common.value.AbstractValue.Make([RigidTransform(), RigidTransform()]))
        # self.DeclareAbstractInputPort(
        #     "vels", pydrake.common.value.AbstractValue.Make([SpatialVelocity(), SpatialVelocity()]))
        # self.DeclareAbstractInputPort(
        #     "contact_results",
        #     pydrake.common.value.AbstractValue.Make(ContactResults()))
        self.DeclareVectorOutputPort(
            "arm_actuation", pydrake.systems.framework.BasicVector(
                self.nq_arm),
            self.CalcOutput)

        self.debug = defaultdict(list)

    def post_finalize_steps(self):
        self.plant_context = self.plant.CreateDefaultContext()

    # , poses, vels, contact_point, slip_speed, pen_depth, N_hat):
    def GetForces(self):
        """
        Should be overloaded to return [Fy, Fz] to move manipulator (not including gravity
        compensation.)
        """
        # raise NotImplementedError()
        return np.array([[10, 0]]).T

    def CalcOutput(self, context, output):
        self.debug['times'].append(context.get_time())
        # This input put is already restricted to the arm, but it includes both q and v
        q = self.get_input_port().Eval(context)[:self.nq_arm]
        self.plant.SetPositions(self.plant_context, self.arm_instance, q)

        finger_body = self.plant.GetBodyByName("finger_body")

        J_raw = self.plant.CalcJacobianTranslationalVelocity(
            self.plant_context,
            JacobianWrtVariable.kQDot,
            finger_body.body_frame(),
            [0, 0, 0],
            self.plant.world_frame(),
            self.plant.world_frame())

        J = J_raw[1:, self.q_idxs]

        forces = self.GetForces()
        out = J.T@forces
        out = out.flatten()

        grav_all = self.plant.CalcGravityGeneralizedForces(
            self.plant_context)
        grav = grav_all[self.q_idxs]
        out -= grav
        output.SetFromVector(out)
