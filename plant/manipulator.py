# Drake imports
import pydrake
from pydrake.multibody.tree import SpatialInertia, UnitInertia
from pydrake.all import RigidTransform, RotationMatrix, Mesh, \
    CollisionFilterDeclaration

import numpy as np

# Imports of other project files
from constants import IN_TO_M, EPSILON
import config

BOX_MASS = EPSILON

# For the panda, the end effector is always at least a hemisphere, but may be
# more than that. This variable keeps track of home much of the second/lower
# hemisphere to include, where 0 indicates just a hemisphere, and 1 indicates a
# full sphere.
fraction_of_lower_hemisphere = 0.5 # Only applies if not using mesh
dist_between_panda_hand_and_edge = 2*IN_TO_M
USE_MESH = False
USE_BOX = False

neutral_q = [
    0.00010707791913362024,
    -0.7853274663373043,
    6.881470455973932e-05,
    -2.356330685498422,
    -0.00014061049952148694,
    1.5709792686628385,
    0.7856590871546001
]

panda_x_offset = 0.25
panda_offset = 0
if config.num_links == config.NumLinks.TWO:
    panda_offset = IN_TO_M*22
elif config.num_links == config.NumLinks.FOUR:
    panda_offset = IN_TO_M*22
X_W_panda = RigidTransform(RotationMatrix().MakeZRotation(-np.pi/2), [
    panda_x_offset,
    panda_offset,
    0
])

def setArmPositions(diagram, diagram_context, plant, manipulator_instance):
    q0 = np.zeros(7)
    q0[0] = -np.pi/2 # -np.pi/12 -> closer to physical
    q0[1] = 0
    q0[2] = np.pi/2
    q0[3] = -np.pi
    q0[4] = np.pi
    q0[5] = np.pi/4 #1.5*np.pi+0.2
    q0[6] = -3*np.pi/4-np.pi/2

    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
    plant.SetPositions(plant_context, manipulator_instance, q0)

def addPrimitivesEndEffector(
        plant, scene_graph, main_end_effector_body, arm_instance, r, mu):
    end_effector_RT = RigidTransform(
        p=[0,0,0]
    )

    # Create box body, if using it
    if USE_BOX:
        # Set up box dimensions
        partial_radius = r * (
            1 - fraction_of_lower_hemisphere**2)**0.5
        box_side = partial_radius*2
        box_height = r*(1-fraction_of_lower_hemisphere)

        # Create body
        box = pydrake.geometry.Box(
            box_side, box_side, box_height)
        box_body = plant.AddRigidBody(
            "box_body", arm_instance,
            pydrake.multibody.tree.SpatialInertia(
                mass=BOX_MASS,
                p_PScm_E=np.array([0., 0., 0.]),
                G_SP_E=pydrake.multibody.tree.UnitInertia(1.0, 1.0, 1.0)))

    if plant.geometry_source_is_registered():
        # Collision geometry for sphere
        plant.RegisterCollisionGeometry(
            main_end_effector_body,
            end_effector_RT,
            pydrake.geometry.Sphere(r),
            "main_end_effector_body",
            pydrake.multibody.plant.CoulombFriction(mu, mu))
        
        # Set up collision filter group
        geometry_list = [
            main_end_effector_body,
            plant.GetBodyByName("panda_link8"),
            plant.GetBodyByName("panda_hand")
        ]

        # Collision geometry for box
        if USE_BOX:
            plant.RegisterCollisionGeometry(
                box_body,
                RigidTransform(),
                box,
                "box_body",
                pydrake.multibody.plant.CoulombFriction(mu, mu))
            geometry_list.append(box_body)
        
        # Create filters
        geometries = plant.CollectRegisteredGeometries(geometry_list)
        scene_graph.collision_filter_manager().Apply(
            CollisionFilterDeclaration().ExcludeWithin(geometries))

        # Visual geometry for sphere
        plant.RegisterVisualGeometry(
            main_end_effector_body,
            end_effector_RT,
            pydrake.geometry.Sphere(r),
            "main_end_effector_body",
            [.9, .5, .5, 1.0])  # Color

        # Visual geometry for box
        if USE_BOX:
            plant.RegisterVisualGeometry(
                box_body,
                RigidTransform(),
                box,
                "box_body",
                [.9, .5, .5, 1.0])  # Color

    if USE_BOX:
        # Weld box (but not sphere) to arm
        plant.WeldFrames(
            plant.GetFrameByName("panda_link8", arm_instance),
            plant.GetFrameByName("box_body", arm_instance),
            RigidTransform(
                R=RotationMatrix.MakeZRotation(np.pi/4),
                p=[0,0,0.065-box_height/2]
        ))


def addMeshEndEffector(plant, scene_graph, main_end_effector_body, mu):
    partial_sphere_mesh = Mesh("models/partial_sphere.obj")
    if plant.geometry_source_is_registered():
        # Collision geometry
        plant.RegisterCollisionGeometry(
            main_end_effector_body,
            RigidTransform(R=RotationMatrix.MakeXRotation(np.pi/2)),
            partial_sphere_mesh,
            "main_end_effector_body",
            pydrake.multibody.plant.CoulombFriction(mu, mu))

        geometries = plant.CollectRegisteredGeometries(
            [
                main_end_effector_body,
                plant.GetBodyByName("panda_link8"),
                plant.GetBodyByName("panda_hand")
            ])
        scene_graph.collision_filter_manager().Apply(
            CollisionFilterDeclaration()
                .ExcludeWithin(geometries))

        # Visual geometry
        plant.RegisterVisualGeometry(
            main_end_effector_body,
            RigidTransform(R=RotationMatrix.MakeXRotation(np.pi/2)),
            partial_sphere_mesh,
            "main_end_effector_body",
            [.9, .5, .5, 1.0])  # Color


def addArm(plant, m_M, r, mu, scene_graph=None):
    """
    Creates the panda arm.
    """

    # =========================== ARM INITIALIZATION ==========================
    # Load arm
    parser = pydrake.multibody.parsing.Parser(plant, scene_graph)
    # arm_instance = parser.AddModelFromFile("models/panda_arm.urdf")
    arm_instance = parser.AddModelFromFile("models/panda_arm_collision_only.urdf")

    # Weld to world (position depends on number of links)
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("panda_link0", arm_instance),
        X_W_panda
    )

    # ====================== END EFFECTOR INITIALIZATION ======================
    # Exclude collisions between additional panda links
    if plant.geometry_source_is_registered():
        geometries = plant.CollectRegisteredGeometries(
            [
                plant.GetBodyByName("panda_link6"),
                plant.GetBodyByName("panda_link8")
            ])
        scene_graph.collision_filter_manager().Apply(
            CollisionFilterDeclaration()
                .ExcludeWithin(geometries))

    # Create the main body for the end effector, setting inertial properties
    main_end_effector_body = plant.AddRigidBody(
        "main_end_effector_body", arm_instance,
        pydrake.multibody.tree.SpatialInertia(
            mass=m_M,
            p_PScm_E=np.array([0., 0., 0.]),
            G_SP_E=pydrake.multibody.tree.UnitInertia(1.0, 1.0, 1.0)))

    # Set collision + visual geometry
    if USE_MESH:
        addMeshEndEffector(plant, scene_graph, main_end_effector_body, mu)
    else:
        addPrimitivesEndEffector(
            plant, scene_graph, main_end_effector_body, arm_instance, r, mu)

    # Weld
    half_panda_hand_height = 0.065
    sphere_partiality_deduction = fraction_of_lower_hemisphere*r
    X_P_S = RigidTransform(
        RotationMatrix().MakeZRotation(-np.pi/4),
        [
            0,
            0,
            (
                half_panda_hand_height \
                + dist_between_panda_hand_and_edge \
                + r - sphere_partiality_deduction
            )]
    ) # Roughly aligns axes with world axes
    plant.WeldFrames(
        plant.GetFrameByName("panda_link8", arm_instance),
        plant.GetFrameByName("main_end_effector_body", arm_instance),
        X_P_S
    )

    return arm_instance

def setSpherePositions(diagram, diagram_context, plant, manipulator_instance):
    pass

def addSphere(plant, m_M, r, mu, scene_graph=None):
    """Adds the manipulator."""
    sphere = plant.AddModelInstance("sphere")

    # Initialize sphere body
    sphere_body = plant.AddRigidBody(
        "sphere_body", sphere,
        pydrake.multibody.tree.SpatialInertia(
            mass=m_M,
            p_PScm_E=np.array([0., 0., 0.]),
            G_SP_E=pydrake.multibody.tree.UnitInertia(1.0, 1.0, 1.0)))

    # Initialize false bodies
    empty_inertia = SpatialInertia(0, [0, 0, 0], UnitInertia(0, 0, 0))
    for i in range(5):
        # Add false bodies for control joints
        plant.AddRigidBody("false_body{}".format(i), sphere, empty_inertia)

    # Register geometry
    if plant.geometry_source_is_registered():
        plant.RegisterCollisionGeometry(
            sphere_body, RigidTransform(),
            pydrake.geometry.Sphere(r),
            "sphere_body",
            pydrake.multibody.plant.CoulombFriction(mu, mu))
        plant.RegisterVisualGeometry(
            sphere_body,
            RigidTransform(),
            pydrake.geometry.Sphere(r),
            "sphere_body",
            [.9, .5, .5, 1.0])  # Color

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
    "contact_body_name": "main_end_effector_body",
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

# Rotation between link position and nominal manipulator position
RotX_L_Md = 0

assert arm_data.keys() == expected_keys
assert sphere_data.keys() == expected_keys

data = arm_data
