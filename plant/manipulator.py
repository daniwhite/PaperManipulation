# Drake imports
import pydrake
from pydrake.multibody.tree import SpatialInertia, UnitInertia
from pydrake.all import RigidTransform, RotationMatrix, Mesh, \
    CollisionFilterDeclaration

import numpy as np

# Imports of other project files
from constants import IN_TO_M, EPSILON
import config

RADIUS = 0.05
VOLUME = (4/3)*RADIUS**3*np.pi
MASS = EPSILON

# For the panda, the end effector is always at least a hemisphere, but may be
# more than that. This variable keeps track of home much of the second/lower
# hemisphere to include, where 0 indicates just a hemisphere, and 1 indicates a
# full sphere.
fraction_of_lower_hemisphere = 0.5 # Only applies if not using mesh
USE_MESH = False
USE_BOX = False

def setArmPositions(diagram, diagram_context, plant, manipulator_instance):
    q0 = np.zeros(7)
    q0[0] = np.pi/2
    q0[1] = 0 #0.6
    q0[3] = -np.pi+0.4
    q0[5] = 1.5*np.pi+0.2
    q0[6] = -3*np.pi/4
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
    plant.SetPositions(plant_context, manipulator_instance, q0)

def addPrimitivesEndEffector(plant, scene_graph, sphere_body, arm_instance):
    end_effector_RT = RigidTransform(
        p=[0,-fraction_of_lower_hemisphere*RADIUS,0]
    )
    if plant.geometry_source_is_registered():
        plant.RegisterCollisionGeometry(
            sphere_body,
            end_effector_RT,
            pydrake.geometry.Sphere(RADIUS),
            "sphere_body",
            pydrake.multibody.plant.CoulombFriction(1, 1))
        plant.RegisterVisualGeometry(
            sphere_body,
            end_effector_RT,
            pydrake.geometry.Sphere(RADIUS),
            "sphere_body",
            [.9, .5, .5, 1.0])  # Color

    if USE_BOX:
        partial_radius = RADIUS * (
            1 - fraction_of_lower_hemisphere**2)**0.5
        box_side = partial_radius*2
        box_height = RADIUS*(1-fraction_of_lower_hemisphere)
        box = pydrake.geometry.Box(
            box_side, box_side, box_height)
        box_body = plant.AddRigidBody(
            "box_body", arm_instance,
            pydrake.multibody.tree.SpatialInertia(
                mass=MASS,
                p_PScm_E=np.array([0., 0., 0.]),
                G_SP_E=pydrake.multibody.tree.UnitInertia(1.0, 1.0, 1.0)))
        if plant.geometry_source_is_registered():
            plant.RegisterCollisionGeometry(
                box_body,
                RigidTransform(),
                box,
                "box_body",
                pydrake.multibody.plant.CoulombFriction(1, 1))
            plant.RegisterVisualGeometry(
                box_body,
                RigidTransform(),
                box,
                "box_body",
                [.9, .5, .5, 1.0])  # Color

            geometries = plant.CollectRegisteredGeometries(
                [
                    box_body,
                    sphere_body,
                    plant.GetBodyByName("panda_link8")
                ])
            scene_graph.collision_filter_manager().Apply(
                CollisionFilterDeclaration()
                    .ExcludeWithin(geometries))
        plant.WeldFrames(
            plant.GetFrameByName("panda_link8", arm_instance),
            plant.GetFrameByName("box_body", arm_instance),
            RigidTransform(
                R=RotationMatrix.MakeZRotation(np.pi/4),
                p=[0,0,0.065-box_height/2]
        ))


def addMeshEndEffector(plant, scene_graph, sphere_body):
    # Initialize sphere body
    partial_sphere_mesh = Mesh("models/partial_sphere.obj")
    if plant.geometry_source_is_registered():
        plant.RegisterCollisionGeometry(
            sphere_body,
            RigidTransform(R=RotationMatrix.MakeXRotation(np.pi/2)),
            partial_sphere_mesh,
            "sphere_body",
            pydrake.multibody.plant.CoulombFriction(1, 1))
        plant.RegisterVisualGeometry(
            sphere_body,
            RigidTransform(R=RotationMatrix.MakeXRotation(np.pi/2)),
            partial_sphere_mesh,
            "sphere_body",
            [.9, .5, .5, 1.0])  # Color

        geometries = plant.CollectRegisteredGeometries(
            [
                sphere_body,
                plant.GetBodyByName("panda_link8")
            ])
        scene_graph.collision_filter_manager().Apply(
            CollisionFilterDeclaration()
                .ExcludeWithin(geometries))


def addArm(plant, scene_graph=None):
    """
    Creates the panda arm.
    """
    parser = pydrake.multibody.parsing.Parser(plant, scene_graph)
    arm_instance = parser.AddModelFromFile("models/panda_arm.urdf")


    panda_offset = 0
    if config.num_links == config.NumLinks.TWO:
        panda_offset = IN_TO_M*22
    elif config.num_links == config.NumLinks.FOUR:
        panda_offset = IN_TO_M*40
    # Weld panda to world
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("panda_link0", arm_instance),
        RigidTransform(RotationMatrix().MakeZRotation(np.pi), [
            0,
            panda_offset,
            0
        ])
    )

    sphere_body = plant.AddRigidBody(
        "sphere_body", arm_instance,
        pydrake.multibody.tree.SpatialInertia(
            mass=MASS,
            p_PScm_E=np.array([0., 0., 0.]),
            G_SP_E=pydrake.multibody.tree.UnitInertia(1.0, 1.0, 1.0)))

    if USE_MESH:
        addMeshEndEffector(plant, scene_graph, sphere_body)
    else:
        addPrimitivesEndEffector(plant, scene_graph, sphere_body, arm_instance)

    X_P_S = RigidTransform(
        RotationMatrix.MakeZRotation(np.pi/4).multiply(
            RotationMatrix.MakeXRotation(-np.pi/2)),
        [0, 0, 0.065]
    ) # Roughly aligns axes with world axes
    plant.WeldFrames(
        plant.GetFrameByName("panda_link8", arm_instance),
        plant.GetFrameByName("sphere_body", arm_instance),
        X_P_S
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
            pydrake.multibody.plant.CoulombFriction(1, 1))
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
    sphere_y_translation.set_default_translation(0.2)
    # Linear z control
    sphere_z_translation = plant.AddJoint(pydrake.multibody.tree.PrismaticJoint(
        "sphere_z",
        plant.GetFrameByName("false_body1"),
        plant.GetFrameByName("false_body2"), [0, 0, 1], -1, 1))
    sphere_z_translation.set_default_translation(0.25)
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
    "contact_body_name": "sphere_body",
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

data = arm_data
