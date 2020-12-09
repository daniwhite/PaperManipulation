import constants
import pydrake
from pydrake.all import RigidTransform, RotationMatrix, PoseBundle, RollPitchYaw, SpatialVelocity
from pydrake.multibody.tree import SpatialInertia, UnitInertia
import numpy as np


class LogWrapper(pydrake.systems.framework.LeafSystem):
    def __init__(self, num_bodies):
        pydrake.systems.framework.LeafSystem.__init__(self)
        self._size = num_bodies*12

        self.DeclareAbstractInputPort(
            "poses", pydrake.common.value.AbstractValue.Make([RigidTransform(), RigidTransform()]))
        self.DeclareAbstractInputPort(
            "vels", pydrake.common.value.AbstractValue.Make([SpatialVelocity(), SpatialVelocity()]))
        self.DeclareVectorOutputPort(
            "out", pydrake.systems.framework.BasicVector(num_bodies*12),
            self.CalcOutput)

    def CalcOutput(self, context, output):
        out = []
        poses = self.get_input_port(0).Eval(context)
        vels = self.get_input_port(1).Eval(context)
        for pose, vel in zip(poses, vels):
            out += list(pose.translation())
            rot_vec = RollPitchYaw(pose.rotation()).vector()
            # _deg*np.pi/180
            rot_vec[1] = pose.rotation().ToAngleAxis().angle()
            if sum(pose.rotation().ToAngleAxis().axis()) < 0:
                rot_vec[1] *= -1
            # print(pose.rotation().ToAngleAxis().axis())
            out += list(rot_vec)
            out += list(vel.translational())
            out += list(vel.rotational())
        output.SetFromVector(out)
