import pydrake
from pydrake.all import RigidTransform, RotationMatrix

PEDESTAL_WIDTH = 0.225
PEDESTAL_HEIGHT = 0.1
PEDESTAL_DEPTH = 0.3


def AddPedestal(plant):
    # Parse pedestal model
    parser = pydrake.multibody.parsing.Parser(plant)
    pedestal_instance = parser.AddModelFromFile("pedestal.sdf")

    # Weld pedestal to world
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("pedestal_base", pedestal_instance),
        RigidTransform(RotationMatrix(), [0, 0, PEDESTAL_HEIGHT/2])
    )

    return pedestal_instance
