"""Parameters and functions related with the pedestal."""

import numpy as np

import pydrake
from pydrake.all import RigidTransform, RotationMatrix
from pydrake.multibody.tree import SpatialInertia, UnitInertia
import config

import constants
from constants import \
    THICK_PLYWOOD_THICKNESS, PLYWOOD_LENGTH, IN_TO_M

PEDESTAL_BASE_Z_DIM = 5*IN_TO_M
PEDESTAL_Y_DIM = PLYWOOD_LENGTH
PEDESTAL_Z_DIM = PLYWOOD_LENGTH + PEDESTAL_BASE_Z_DIM

pedestal_base_name = "pedestal_bottom_body"

def AddPedestal(plant, num_links:config.NumLinks, weld_base=True):
    """
    Creates the pedestal.
    """
    PEDESTAL_X_OFFSET, PEDESTAL_Y_OFFSET, PEDESTAL_Z_OFFSET = np.load(
        "pedestal_xyz_{}_links.npz".format(num_links.value))['pedestal_xyz']
    pedestal_x_dim = constants.PEDESTAL_X_DIM(num_links)

    pedestal_instance = plant.AddModelInstance("pedestal_base")

    # Add bump at the bottom
    bottom_name = pedestal_base_name
    bottom_body_mass = 1 if weld_base else 1e9
    body =  plant.AddRigidBody(
        bottom_name,
        pedestal_instance,
        SpatialInertia(mass=bottom_body_mass,
                        p_PScm_E=np.array([0., 0., 0.]),
                        G_SP_E=UnitInertia.SolidBox(
                            pedestal_x_dim,
                            PEDESTAL_Y_DIM,
                            PEDESTAL_BASE_Z_DIM)
    ))

    if plant.geometry_source_is_registered():
        plant.RegisterCollisionGeometry(
            body,
            RigidTransform(),
            pydrake.geometry.Box(
                pedestal_x_dim,
                PEDESTAL_Y_DIM,
                PEDESTAL_BASE_Z_DIM
            ),
            bottom_name,
            pydrake.multibody.plant.CoulombFriction(1, 1)
        )

        plant.RegisterVisualGeometry(
            body,
            RigidTransform(),
            pydrake.geometry.Box(
                pedestal_x_dim,
                PEDESTAL_Y_DIM,
                PEDESTAL_BASE_Z_DIM
            ),
            bottom_name,
            [0.4, 0.4, 0.4, 1])  # RGBA color

        if weld_base:
            plant.WeldFrames(
                plant.world_frame(),
                plant.GetFrameByName(bottom_name, pedestal_instance),
                RigidTransform(RotationMatrix.MakeZRotation(np.pi), [
                    PEDESTAL_X_OFFSET,
                    PEDESTAL_Y_OFFSET,
                    PEDESTAL_Z_OFFSET
                ]
            ))

    names = ["left", "right"]
    y_positions = [
        (PLYWOOD_LENGTH - THICK_PLYWOOD_THICKNESS)/2,
        -(PLYWOOD_LENGTH - THICK_PLYWOOD_THICKNESS)/2
    ]

    for name, y_position in zip(names, y_positions):
        full_name = "pedestal_" + name + "_body"
        body =  plant.AddRigidBody(
            full_name,
            pedestal_instance,
            SpatialInertia(mass=1, # Doesn't matter because it's welded
                            p_PScm_E=np.array([0., 0., 0.]),
                            G_SP_E=UnitInertia.SolidBox(
                                pedestal_x_dim,
                                THICK_PLYWOOD_THICKNESS,
                                PEDESTAL_Z_DIM)
        ))

        if plant.geometry_source_is_registered():
            plant.RegisterCollisionGeometry(
                body,
                RigidTransform(),
                pydrake.geometry.Box(
                    pedestal_x_dim,
                    THICK_PLYWOOD_THICKNESS,
                    PLYWOOD_LENGTH
                ),
                full_name,
                pydrake.multibody.plant.CoulombFriction(1, 1)
            )

            plant.RegisterVisualGeometry(
                body,
                RigidTransform(),
                pydrake.geometry.Box(
                    pedestal_x_dim,
                    THICK_PLYWOOD_THICKNESS,
                    PLYWOOD_LENGTH
                ),
                full_name,
                [0.4, 0.4, 0.4, 1])  # RGBA color

            plant.WeldFrames(
                plant.GetFrameByName(bottom_name, pedestal_instance),
                plant.GetFrameByName(full_name, pedestal_instance),
                RigidTransform(RotationMatrix(), [
                    0,
                    y_position,
                    PLYWOOD_LENGTH/2+ PEDESTAL_BASE_Z_DIM/2
                ]
            ))


    return pedestal_instance
