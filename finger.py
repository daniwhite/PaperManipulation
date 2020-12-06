import constants
import pydrake
from pydrake.all import RigidTransform, RotationMatrix, SpatialVelocity
from pydrake.multibody.tree import SpatialInertia, UnitInertia
import numpy as np


def AddFinger(plant, init_x, init_z):
    """Adds the manipulator."""
    radius = constants.FINGER_RADIUS
    finger = plant.AddModelInstance("finger")

    # Add false body at the origin that the finger
    plant.AddRigidBody("false_body", finger, SpatialInertia(
        0, [0, 0, 0], UnitInertia(0, 0, 0)))

    # Initialize finger body
    finger_body = plant.AddRigidBody("body", finger,
                                     pydrake.multibody.tree.SpatialInertia(
                                         mass=constants.FINGER_MASS,
                                         p_PScm_E=np.array([0., 0., 0.]),
                                         G_SP_E=pydrake.multibody.tree.UnitInertia(1.0, 1.0, 1.0)))

    # Register geometry
    shape = pydrake.geometry.Sphere(radius)
    if plant.geometry_source_is_registered():
        plant.RegisterCollisionGeometry(
            finger_body, RigidTransform(), shape, "body", pydrake.multibody.plant.CoulombFriction(
                constants.FRICTION, constants.FRICTION))
        plant.RegisterVisualGeometry(
            finger_body, RigidTransform(), shape, "body", [.9, .5, .5, 1.0])

    # Add control joins for x and z movement
    finger_x = plant.AddJoint(pydrake.multibody.tree.PrismaticJoint(
        "finger_x",
        plant.world_frame(),
        plant.GetFrameByName("false_body"), [1, 0, 0], -1, 1))
    plant.AddJointActuator("finger_x", finger_x)
    finger_x.set_default_translation(init_x)
    finger_z = plant.AddJoint(pydrake.multibody.tree.PrismaticJoint(
        "finger_z",
        plant.GetFrameByName("false_body"),
        plant.GetFrameByName("body"), [0, 0, 1], -1, 1))
    finger_z.set_default_translation(init_z)
    plant.AddJointActuator("finger_z", finger_z)

    return finger


class FingerController(pydrake.systems.framework.LeafSystem):
    """Base class for implementing a controller at the finger."""

    def __init__(self, plant, finger_idx):
        pydrake.systems.framework.LeafSystem.__init__(self)
        self._plant = plant

        self.DeclareAbstractInputPort(
            "poses", pydrake.common.value.AbstractValue.Make([RigidTransform(), RigidTransform()]))
        self.DeclareAbstractInputPort(
            "vels", pydrake.common.value.AbstractValue.Make([SpatialVelocity(), SpatialVelocity()]))
        self.DeclareVectorOutputPort(
            "finger_actuation", pydrake.systems.framework.BasicVector(2),
            self.CalcOutput)

        self.finger_idx = finger_idx

    def GetForces(self, poses, vels):
        raise NotImplementedError()

    def CalcOutput(self, context, output):
        # Get inputs
        g = self._plant.gravity_field().gravity_vector()[[0, 2]]
        poses = self.get_input_port(0).Eval(context)
        vels = self.get_input_port(1).Eval(context)

        fx, fz = self.GetForces(poses, vels)
        output.SetFromVector(-constants.FINGER_MASS*g + [fx, fz])


class PDFinger(FingerController):
    """Set up PD controller for finger."""

    def __init__(self, plant, finger_idx, pts, tspan_per_segment=5, kx=5, kz=5, dx=0.01, dz=0.01):
        super().__init__(plant, finger_idx)

        # Generate trajectory
        self.xs = []
        self.zs = []
        self.xdots = [0]
        self.zdots = [0]

        for segment_i in range(len(pts)-1):
            start_pt = pts[segment_i]
            end_pt = pts[segment_i+1]
            for prog_frac in np.arange(0, tspan_per_segment, constants.DT)/tspan_per_segment:
                new_x = start_pt[0] + (end_pt[0] - start_pt[0])*prog_frac
                self.xs.append(new_x)
                new_z = start_pt[1] + (end_pt[1] - start_pt[1])*prog_frac
                self.zs.append(new_z)

        for i in range(len(self.xs)-1):
            self.xdots.append((self.xs[i+1]-self.xs[i])/constants.DT)
            self.zdots.append((self.zs[i+1]-self.zs[i])/constants.DT)

        # Set gains
        self.kx = kx
        self.kz = kz
        self.dx = dx
        self.dz = dz

        # For keeping track of place in trajectory
        self.idx = 0

    def GetForces(self, poses, vels):
        # Unpack values
        x = poses[self.finger_idx].translation()[0]
        z = poses[self.finger_idx].translation()[2]
        xdot = vels[self.finger_idx].translational()[0]
        zdot = vels[self.finger_idx].translational()[2]

        if self.idx < len(self.xs):
            fx = self.kx*(self.xs[self.idx] - x) + \
                self.dx*(self.xdots[self.idx] - xdot)
            fz = self.kz*(self.zs[self.idx] - z) + \
                self.dz*(self.zdots[self.idx] - zdot)
        else:
            fx = self.kx*(self.xs[-1] - x) + self.dx*(-xdot)
            fz = self.kz*(self.zs[-1] - z) + self.dz*(- zdot)
        self.idx += 1

        return fx, fz
