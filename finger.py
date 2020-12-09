import constants
import pydrake
from pydrake.all import RigidTransform, RotationMatrix, SpatialVelocity, RollPitchYaw
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


class EdgeController(FingerController):

    def __init__(self, plant, finger_idx, ll_idx, K=None, D=None, F_Nd=None, d_d=None, w_l=None, h_l=None):
        super().__init__(plant, finger_idx)

        self.ll_idx = ll_idx
        self.K = K
        self.D = D
        self.F_Nd = F_Nd
        self.d_d = d_d
        self.last_d = 0
        self.last_theta_sign = None

        self.w_l = w_l
        self.h_l = h_l
        self.r = constants.FINGER_RADIUS
        self.theta_crossed_thresh = False

        self.debug = {}
        self.debug['Fx'] = []
        self.debug['Fz'] = []
        self.debug['theta_y'] = []
        self.debug['FN'] = []
        self.debug['FP'] = []
        self.debug['d'] = []
        self.debug['delta_d'] = []
        self.debug['ddot'] = []
        self.debug['impulse'] = []
        self.debug['dp'] = []
        self.debug['dn'] = []

    def GetForces(self, poses, vels):
        # Unpack values
        x_m = poses[self.finger_idx].translation()[0]
        z_m = poses[self.finger_idx].translation()[2]
        xdot_m = vels[self.finger_idx].translational()[0]
        zdot_m = vels[self.finger_idx].translational()[2]
        x_l = poses[self.ll_idx].translation()[0]
        z_l = poses[self.ll_idx].translation()[2]
        xdot_l = vels[self.ll_idx].translational()[0]
        zdot_l = vels[self.ll_idx].translational()[2]
        minus_theta_y = poses[self.ll_idx].rotation().ToAngleAxis().angle()
        if sum(poses[self.ll_idx].rotation().ToAngleAxis().axis()) < 0:
            minus_theta_y *= -1
        minus_omega_y = vels[self.ll_idx].rotational()[1]

        # Fix angle convention
        theta_y = -minus_theta_y
        omega_y = -minus_omega_y

        # d = np.sin(theta_y)*(z_m-z_l) + np.cos(theta_y)*(x_m-x_l)
        x_edge = x_l+(self.w_l/2)*np.cos(theta_y)+(self.h_l/2)*np.sin(theta_y)
        z_edge = z_l+(self.w_l/2)*np.sin(theta_y)-(self.h_l/2)*np.cos(theta_y)
        x_contact = x_m + self.r*np.sin(theta_y)
        z_contact = z_m + self.r*np.cos(theta_y)
        d = np.sqrt((x_edge - x_contact)**2 + (z_edge - z_contact)
                    ** 2)

        # (x_p, z_p) = projected position of manipulator onto link
        x_p = np.cos(theta_y)*(np.cos(theta_y)*(x_m-x_edge) +
                               np.sin(theta_y)*(z_m-z_edge))+x_edge
        z_p = np.sin(theta_y)*(np.cos(theta_y)*(x_m-x_m) +
                               np.sin(theta_y)*(z_m-z_edge))+z_edge

        # dp = parallel distance
        dp = np.sqrt((x_edge - x_p)**2 + (z_edge - z_p) ** 2) - \
            constants.FINGER_RADIUS
        self.debug['dp'].append(dp)
        dn = np.sqrt((x_m - x_p)**2 + (z_m - z_p) ** 2) - \
            constants.FINGER_RADIUS
        self.debug['dn'].append(dn)
        d = dp

        ddot = (d-self.last_d)/constants.DT
        self.last_d = d
        #  np.sin(theta_y)*((zdot_m-zdot_l)-omega_y*(x_m-x_l)) \
        #     + np.cos(theta_y)*((xdot_m-xdot_l)-omega_y*(z_m-z_l))

        # d_n =

        # Calculate intermediate forces
        F_Pd = -(self.K*(self.d_d-d) - self.D*ddot)  # F_P = towards edge
        F_Nd = self.F_Nd if (d-self.d_d) < 0.01 else 0.01

        Fx = np.cos(theta_y)*F_Pd - np.sin(theta_y)*F_Nd
        Fz = np.sin(theta_y)*F_Pd + np.cos(theta_y)*F_Nd

        theta_sign = np.sign(np.cos(theta_y))
        if theta_y > 0:
            self.theta_crossed_thresh = 0
        # if self.theta_crossed_thresh and np.abs(theta_y - np.pi/2) < constants.EPSILON:
        if self.last_theta_sign is not None and theta_sign != self.last_theta_sign:
            z_impulse = -constants.FINGER_MASS*zdot_m
            z_impulse /= constants.DT
            Fz += z_impulse
            x_impulse = -constants.FINGER_MASS*xdot_m
            x_impulse /= constants.DT
            Fx += x_impulse
            self.debug['impulse'].append(x_impulse + z_impulse)
        else:
            self.debug['impulse'].append(0)
        self.last_theta_sign = theta_sign

        # Update debug traces
        self.debug['Fx'].append(Fx)
        self.debug['Fz'].append(Fz)
        self.debug['theta_y'].append(theta_y)
        self.debug['FN'].append(F_Nd)
        self.debug['FP'].append(F_Pd)
        self.debug['d'].append(d)
        self.debug['delta_d'].append(self.d_d-d)
        self.debug['ddot'].append(ddot)

        return Fx, Fz
