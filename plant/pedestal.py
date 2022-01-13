"""Parameters and functions related with the pedestal."""

import numpy as np

import pydrake
from pydrake.all import RigidTransform, RotationMatrix
from pydrake.multibody.tree import SpatialInertia, UnitInertia

from constants import THICK_PLYWOOD_THICKNESS, PLYWOOD_LENGTH, IN_TO_M

bump_z = 5*IN_TO_M
PEDESTAL_X_DIM = PLYWOOD_LENGTH
PEDESTAL_Y_DIM = PLYWOOD_LENGTH/2
PEDESTAL_Z_DIM = PLYWOOD_LENGTH + bump_z

def AddPedestal(plant):
    """
    Creates the pedestal.
    """

    pedestal_instance = plant.AddModelInstance("pedestal_base")

    bodies = []
    names = ["left", "right"]
    x_positions = [
        (PLYWOOD_LENGTH - THICK_PLYWOOD_THICKNESS)/2,
        -(PLYWOOD_LENGTH - THICK_PLYWOOD_THICKNESS)/2
    ]

    for name, x_position in zip(names, x_positions):
        full_name = "pedestal_" + name + "_body"
        body =  plant.AddRigidBody(
            full_name,
            pedestal_instance,
            SpatialInertia(mass=1, # Doesn't matter because it's welded
                            p_PScm_E=np.array([0., 0., 0.]),
                            G_SP_E=UnitInertia.SolidBox(
                                THICK_PLYWOOD_THICKNESS,
                                PEDESTAL_Y_DIM,
                                PEDESTAL_Z_DIM)
        ))

        if plant.geometry_source_is_registered():
            plant.RegisterCollisionGeometry(
                body,
                RigidTransform(),
                pydrake.geometry.Box(
                    THICK_PLYWOOD_THICKNESS,
                    PEDESTAL_Y_DIM,
                    PLYWOOD_LENGTH
                ),
                full_name,
                pydrake.multibody.plant.CoulombFriction(1, 1)
            )

            plant.RegisterVisualGeometry(
                body,
                RigidTransform(),
                pydrake.geometry.Box(
                    THICK_PLYWOOD_THICKNESS,
                    PEDESTAL_Y_DIM,
                    PLYWOOD_LENGTH
                ),
                full_name,
                [0.4, 0.4, 0.4, 1])  # RGBA color

            plant.WeldFrames(
                plant.world_frame(),
                plant.GetFrameByName(full_name, pedestal_instance),
                RigidTransform(RotationMatrix(), [
                    x_position,
                    0,
                    PLYWOOD_LENGTH/2+ bump_z
                ]
            ))
    
    # Add bump at the bottom
    full_name = "pedestal_bottom_body"
    body =  plant.AddRigidBody(
        full_name,
        pedestal_instance,
        SpatialInertia(mass=1, # Doesn't matter because it's welded
                        p_PScm_E=np.array([0., 0., 0.]),
                        G_SP_E=UnitInertia.SolidBox(
                            PEDESTAL_X_DIM,
                            PEDESTAL_Y_DIM,
                            bump_z)
    ))

    if plant.geometry_source_is_registered():
        plant.RegisterCollisionGeometry(
            body,
            RigidTransform(),
            pydrake.geometry.Box(
                PEDESTAL_X_DIM,
                PEDESTAL_Y_DIM,
                bump_z
            ),
            full_name,
            pydrake.multibody.plant.CoulombFriction(1, 1)
        )

        plant.RegisterVisualGeometry(
            body,
            RigidTransform(),
            pydrake.geometry.Box(
                PEDESTAL_X_DIM,
                PEDESTAL_Y_DIM,
                bump_z
            ),
            full_name,
            [0.4, 0.4, 0.4, 1])  # RGBA color

        plant.WeldFrames(
            plant.world_frame(),
            plant.GetFrameByName(full_name, pedestal_instance),
            RigidTransform(RotationMatrix(), [
                0,
                0,
                bump_z/2
            ]
        ))

    return pedestal_instance
