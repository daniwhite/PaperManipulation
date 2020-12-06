import constants
import pydrake
from pydrake.all import RigidTransform, RotationMatrix, PoseBundle, RollPitchYaw
from pydrake.multibody.tree import SpatialInertia, UnitInertia
import numpy as np


class BodyPoseWrapper(pydrake.systems.framework.LeafSystem):
    def __init__(self, num_bodies):
        pydrake.systems.framework.LeafSystem.__init__(self)
        self._size = num_bodies*6

        self.DeclareAbstractInputPort(
            "inp", pydrake.common.value.AbstractValue.Make([RigidTransform(), RigidTransform()]))
        self.DeclareVectorOutputPort(
            "out", pydrake.systems.framework.BasicVector(num_bodies*6),
            self.CalcOutput)

    def CalcOutput(self, context, output):
        inp = self.get_input_port(0).Eval(context)
        out = []
        for RT in inp:
            out += list(RT.translation())
            out += list(RollPitchYaw(RT.rotation()).vector())
        output.SetFromVector(out)
