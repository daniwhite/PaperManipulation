"""Defines wrapper class to pack inputs for a logger."""
# Drake imports
import pydrake
from pydrake.all import RigidTransform, RollPitchYaw, SpatialVelocity, SpatialAcceleration


class LogWrapper(pydrake.systems.framework.LeafSystem):
    """
    Wrapper system that converts RigidTransform and SpatialVelocity inputs into vector output so it
    can be used easily with a logger.
    """

    def __init__(self, num_bodies):
        pydrake.systems.framework.LeafSystem.__init__(self)
        self.entries_per_body = 3*6
        self._size = num_bodies*self.entries_per_body

        self.DeclareAbstractInputPort(
            "poses", pydrake.common.value.AbstractValue.Make([RigidTransform(), RigidTransform()]))
        self.DeclareAbstractInputPort(
            "vels", pydrake.common.value.AbstractValue.Make([SpatialVelocity(), SpatialVelocity()]))
        self.DeclareAbstractInputPort(
            "accs", pydrake.common.value.AbstractValue.Make([SpatialAcceleration(), SpatialAcceleration()]))
        self.DeclareVectorOutputPort(
            "out", pydrake.systems.framework.BasicVector(
                num_bodies*self.entries_per_body),
            self.CalcOutput)

    def CalcOutput(self, context, output):
        out = []
        poses = self.get_input_port(0).Eval(context)
        vels = self.get_input_port(1).Eval(context)
        accs = self.get_input_port(2).Eval(context)
        for pose, vel, acc in zip(poses, vels, accs):
            out += list(pose.translation())
            rot_vec = RollPitchYaw(pose.rotation()).vector()

            # AngleAxis is more convenient because of where it wraps around
            rot_vec[1] = pose.rotation().ToAngleAxis().angle()
            if sum(pose.rotation().ToAngleAxis().axis()) < 0:
                rot_vec[1] *= -1

            out += list(rot_vec)
            out += list(vel.translational())
            out += list(vel.rotational())
            out += list(acc.translational())
            out += list(acc.rotational())
        output.SetFromVector(out)
