"""Parameters and functions related with the pedestal."""

import pydrake
from pydrake.all import RigidTransform, RotationMatrix

PEDESTAL_WIDTH = 0.15
PEDESTAL_HEIGHT = 0.6
PEDESTAL_DEPTH = 0.3
# TODO: these should actually be reflected in the box


def AddPedestal(plant):
    """
    Creates the pedestal.
    """
    # Parse pedestal model
    parser = pydrake.multibody.parsing.Parser(plant)
    pedestal_instance = parser.AddModelFromFile("models/pedestal.sdf")

    # Weld pedestal to world
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("pedestal_base", pedestal_instance),
        RigidTransform(RotationMatrix(), [0, 0, PEDESTAL_HEIGHT/2])
    )

    return pedestal_instance
