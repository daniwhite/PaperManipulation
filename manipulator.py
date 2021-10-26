# Drake imports
import pydrake
from pydrake.multibody.tree import SpatialInertia, UnitInertia, JacobianWrtVariable
from pydrake.all import BasicVector, MultibodyPlant, ContactResults, SpatialVelocity, SpatialForce, FindResourceOrThrow, RigidTransform, RotationMatrix, AngleAxis, RollPitchYaw
from pydrake.all import (
    MathematicalProgram, Solve, eq, le, ge,
)
import constants
import pedestal

import numpy as np

RADIUS = 0.01
VOLUME = (4/3)*RADIUS**3*np.pi
MASS = VOLUME*1e3  # Assume finger is made of water

INIT_Y = pedestal.PEDESTAL_WIDTH/2 + RADIUS + constants.EPSILON
INIT_Z = 0


# class ManipulatorPlant:
#     def __init__(self, contact_body_name) -> None:
#         self.contact_body_name = contact_body_name

#     def addManipulator(self, plant, scene_graph=None):
#         raise NotImplementedError
    
#     def get_contact_body_name(self):
#         return self.contact_body_name

# class SpherePlant(ManipulatorPlant):
#     def __init__(self, contact_body_name="sphere_body") -> None:
#         super().__init__(contact_body_name)
    
#     def addManipulator(self, plant, scene_graph=None):
#         pass

# class ArmPlant(ManipulatorPlant):
#     def __init__(self, contact_body_name="panda_leftfinger") -> None:
#         super().__init__(contact_body_name)

def setArmPositions(diagram, diagram_context, plant, manipulator_instance):
    q0 = np.zeros(7)
    q0[0] = -np.pi/2
    q0[1] = 0# 1.1
    q0[3] = np.pi/2
    q0[5] = np.pi/2#3*np.pi/2 + 0.1
    q0[6] = -np.pi/4
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
    plant.SetPositions(plant_context, manipulator_instance, q0)

def addArm(plant, scene_graph=None):
    """
    Creates the panda arm.
    """
    parser = pydrake.multibody.parsing.Parser(plant, scene_graph)
    arm_instance = parser.AddModelFromFile("panda_arm_hand.urdf")

    # Weld pedestal to world
    jnt = plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("panda_link0", arm_instance),
        RigidTransform(RotationMatrix().MakeZRotation(np.pi), [0, 0.65, 0])
    )
    # Weld fingers (offset matches original urdf)
    plant.WeldFrames(
        plant.GetFrameByName("panda_hand", arm_instance),
        plant.GetFrameByName("panda_leftfinger", arm_instance),
        RigidTransform(RotationMatrix(), [0, 0, 0.0584])
    )
    plant.WeldFrames(
        plant.GetFrameByName("panda_hand", arm_instance),
        plant.GetFrameByName("panda_rightfinger", arm_instance),
        RigidTransform(RotationMatrix(), [0, 0, 0.0584])
    )

    return arm_instance

def setSpherePositions(diagram, diagram_context, plant, manipulator_instance):
    pass

def addSphere(plant, scene_graph=None):
    """Adds the manipulator."""
    sphere = plant.AddModelInstance("sphere")

    # Initialize sphere body
    sphere_body = plant.AddRigidBody(
        "sphere_body", sphere,
        pydrake.multibody.tree.SpatialInertia(
            mass=MASS,
            p_PScm_E=np.array([0., 0., 0.]),
            G_SP_E=pydrake.multibody.tree.UnitInertia(1.0, 1.0, 1.0)))

    # Initialize false bodies
    empty_inertia = SpatialInertia(0, [0, 0, 0], UnitInertia(0, 0, 0))
    for i in range(5):
        # Add false bodies for control joints
        false_body = plant.AddRigidBody("false_body{}".format(i), sphere, empty_inertia)
        # plant.WeldFrames(
        #     plant.world_frame(),
        #     false_body.body_frame(),
        #     RigidTransform()
        # )

    # Register geometry
    if plant.geometry_source_is_registered():
        col_geom = plant.RegisterCollisionGeometry(
            sphere_body, RigidTransform(),
            pydrake.geometry.Sphere(RADIUS),
            "sphere_body",
            pydrake.multibody.plant.CoulombFriction(constants.FRICTION, constants.FRICTION))
        plant.RegisterVisualGeometry(
            sphere_body,
            RigidTransform(),
            pydrake.geometry.Sphere(RADIUS),
            "sphere_body",
            [.9, .5, .5, 1.0])  # Color

    # jnt = plant.AddJoint(pydrake.multibody.tree.Joint(
    #     "sphere_joint",
    #     plant.world_frame(),
    #     sphere_body))
    # plant.AddJointActuator("sphere_actuator", jnt)

    # Linear x control
    sphere_x_translation = plant.AddJoint(pydrake.multibody.tree.PrismaticJoint(
        "sphere_x_translation",
        plant.world_frame(),
        plant.GetFrameByName("false_body0"), [1, 0, 0], -1, 1))
    plant.AddJointActuator("sphere_x_translation", sphere_x_translation)
    sphere_x_translation.set_default_translation(0)
    # Linear y control
    sphere_y_translation = plant.AddJoint(pydrake.multibody.tree.PrismaticJoint(
        "sphere_y_translation",
        plant.GetFrameByName("false_body0"),
        plant.GetFrameByName("false_body1"), [0, 1, 0], -1, 1))
    plant.AddJointActuator("sphere_y_translation", sphere_y_translation)
    sphere_y_translation.set_default_translation(INIT_Y)
    # Linear z control
    sphere_z_translation = plant.AddJoint(pydrake.multibody.tree.PrismaticJoint(
        "sphere_z",
        plant.GetFrameByName("false_body1"),
        plant.GetFrameByName("false_body2"), [0, 0, 1], -1, 1))
    sphere_z_translation.set_default_translation(INIT_Z)
    plant.AddJointActuator("sphere_z", sphere_z_translation)
    # Rotational x control
    sphere_x_rotation = plant.AddJoint(pydrake.multibody.tree.RevoluteJoint(
        "sphere_x_rotation",
        plant.GetFrameByName("false_body2"),
        plant.GetFrameByName("false_body3"),
        [1, 0, 0],
        damping=0
    ))
    sphere_x_rotation.set_default_angle(0)
    plant.AddJointActuator("sphere_x_rotation", sphere_x_rotation)
    # Rotational y control
    sphere_y_rotation = plant.AddJoint(pydrake.multibody.tree.RevoluteJoint(
        "sphere_y_rotation",
        plant.GetFrameByName("false_body3"),
        plant.GetFrameByName("false_body4"),
        [0, 1, 0],
        damping=0
    ))
    sphere_y_rotation.set_default_angle(0)
    plant.AddJointActuator("sphere_y_rotation", sphere_y_rotation)
    # Rotational z control
    sphere_z_rotation = plant.AddJoint(pydrake.multibody.tree.RevoluteJoint(
        "sphere_z_rotation",
        plant.GetFrameByName("false_body4"),
        plant.GetFrameByName("sphere_body"),
        [0, 0, 1],
        damping=0
    ))
    sphere_z_rotation.set_default_angle(0)
    plant.AddJointActuator("sphere_z_rotation", sphere_z_rotation)

    return sphere

expected_keys = {
    "contact_body_name",
    "add_plant_function",
    "set_positions",
    "nq"
}

arm_data = {
    "contact_body_name": "panda_leftfinger",
    "add_plant_function": addArm,
    "set_positions": setArmPositions,
    "nq": 7,
}

sphere_data = {
    "contact_body_name": "sphere_body",
    "add_plant_function": addSphere,
    "set_positions": setSpherePositions,
    "nq": 6,
}


assert arm_data.keys() == expected_keys
assert sphere_data.keys() == expected_keys

data = sphere_data
