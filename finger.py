"""Functions for creating and controlling the finger manipulator."""

# Standard imports
import numpy as np

# Drake imports
import pydrake
from pydrake.all import \
    RigidTransform, SpatialVelocity, MathematicalProgram, eq, SnoptSolver, AutoDiffXd

from pydrake.multibody.tree import SpatialInertia, UnitInertia, JacobianWrtVariable

# Imports of other project files
import constants
import paper
import pedestal


def AddFinger(plant, init_y, init_z):
    """Adds the manipulator."""
    radius = constants.FINGER_RADIUS
    finger = plant.AddModelInstance("finger")

    # Add false body at the origin that the finger
    plant.AddRigidBody("false_body", finger, SpatialInertia(
        0, [0, 0, 0], UnitInertia(0, 0, 0)))

    # Initialize finger body
    finger_body = plant.AddRigidBody("finger_body", finger,
                                     pydrake.multibody.tree.SpatialInertia(
                                         mass=constants.FINGER_MASS,
                                         p_PScm_E=np.array([0., 0., 0.]),
                                         G_SP_E=pydrake.multibody.tree.UnitInertia(1.0, 1.0, 1.0)))

    # Register geometry
    shape = pydrake.geometry.Sphere(radius)
    if plant.geometry_source_is_registered():
        plant.RegisterCollisionGeometry(
            finger_body, RigidTransform(),
            shape,
            "finger_body",
            pydrake.multibody.plant.CoulombFriction(constants.FRICTION, constants.FRICTION))
        plant.RegisterVisualGeometry(
            finger_body, RigidTransform(), shape, "finger_body", [.9, .5, .5, 1.0])

    # Add control joins for y and z movement
    finger_y = plant.AddJoint(pydrake.multibody.tree.PrismaticJoint(
        "finger_y",
        plant.world_frame(),
        plant.GetFrameByName("false_body"), [0, 1, 0], -1, 1))
    plant.AddJointActuator("finger_y", finger_y)
    finger_y.set_default_translation(init_y)
    finger_z = plant.AddJoint(pydrake.multibody.tree.PrismaticJoint(
        "finger_z",
        plant.GetFrameByName("false_body"),
        plant.GetFrameByName("finger_body"), [0, 0, 1], -1, 1))
    finger_z.set_default_translation(init_z)
    plant.AddJointActuator("finger_z", finger_z)

    return finger, finger_body


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
        """
        Should be overloaded to return [Fy, Fz] to move manipulator (not including gravity
        compensation.)
        """
        raise NotImplementedError()

    def CalcOutput(self, context, output):
        # Get inputs
        g = self._plant.gravity_field().gravity_vector()[[0, 2]]
        poses = self.get_input_port(0).Eval(context)
        vels = self.get_input_port(1).Eval(context)

        fy, fz = self.GetForces(poses, vels)
        output.SetFromVector(-constants.FINGER_MASS*g + [fy, fz])


class BlankController(FingerController):
    """Fold paper with feedback on position of the past link"""

    # Making these parameters keywords means that
    def __init__(self, plant, finger_idx):
        super().__init__(plant, finger_idx)

    def GetForces(self, poses, vels):
        return [0, 0]


class PDFinger(FingerController):
    """Set up PD controller for finger."""

    def __init__(self, plant, finger_idx, pts, tspan_per_segment=5, ky=5, kz=5, dy=0.01, dz=0.01):
        super().__init__(plant, finger_idx)

        # Generate trajectory
        self.ys = []
        self.zs = []
        self.ydots = [0]
        self.zdots = [0]

        for segment_i in range(len(pts)-1):
            start_pt = pts[segment_i]
            end_pt = pts[segment_i+1]
            for prog_frac in np.arange(0, tspan_per_segment, constants.DT)/tspan_per_segment:
                new_y = start_pt[0] + (end_pt[0] - start_pt[0])*prog_frac
                self.ys.append(new_y)
                new_z = start_pt[1] + (end_pt[1] - start_pt[1])*prog_frac
                self.zs.append(new_z)

        for i in range(len(self.ys)-1):
            self.ydots.append((self.ys[i+1]-self.ys[i])/constants.DT)
            self.zdots.append((self.zs[i+1]-self.zs[i])/constants.DT)

        # Set gains
        self.ky = ky
        self.kz = kz
        self.dy = dy
        self.dz = dz

        # For keeping track of place in trajectory
        self.idx = 0

    def GetForces(self, poses, vels):
        # Unpack values
        y = poses[self.finger_idx].translation()[1]
        z = poses[self.finger_idx].translation()[2]
        ydot = vels[self.finger_idx].translational()[1]
        zdot = vels[self.finger_idx].translational()[2]

        if self.idx < len(self.ys):
            fy = self.ky*(self.ys[self.idx] - y) + \
                self.dy*(self.ydots[self.idx] - ydot)
            fz = self.kz*(self.zs[self.idx] - z) + \
                self.dz*(self.zdots[self.idx] - zdot)
        else:
            fy = self.ky*(self.ys[-1] - y) + self.dy*(-ydot)
            fz = self.kz*(self.zs[-1] - z) + self.dz*(- zdot)
        self.idx += 1

        return fy, fz


class EdgeController(FingerController):
    """Fold paper with feedback on position of the past link"""

    # Making these parameters keywords means that
    def __init__(self, plant, paper_, finger_idx, F_Nd, debug=False):
        super().__init__(plant, finger_idx)
        self.paper = paper_

        # Control parameters
        # self.K = K
        self.F_Nd = F_Nd
        # self.d_d = d_d

        # PROGRAMMING: Reintroduce paper parameters

        # Initialize debug dict if necessary
        if debug:
            self.debug = {}
            self.debug['N_hats'] = []
            self.debug['T_hats'] = []
            self.debug['theta_xs'] = []
            self.debug['omega_xs'] = []
            self.debug['F_Gs'] = []
            self.debug['F_GTs'] = []
            self.debug['F_GNs'] = []
            self.debug['F_Os'] = []
            self.debug['F_OTs'] = []
            self.debug['F_ONs'] = []
            self.debug['F_CNs'] = []
            self.debug['F_CTs'] = []
            self.debug['F_Ms'] = []
            self.debug['d_Ns'] = []
            self.debug['d_Ts'] = []
            self.debug['F_Ns'] = []
        else:
            self.debug = None

    def GetForces(self, poses, vels):
        ll_idx = self.paper.get_free_edge_idx()

        # Unpack rotation
        # AngleAxis is more convenient because of where it wraps around
        theta_x = poses[ll_idx].rotation().ToAngleAxis().angle()
        if sum(poses[ll_idx].rotation().ToAngleAxis().axis()) < 0:
            theta_x *= -1
        omega_x = vels[ll_idx].rotational()[0]

        # Rotation matrix, for convenience
        R = poses[ll_idx].rotation()
        R_inv = R.inverse()

        y_hat = np.array([[0, 1, 0]]).T
        z_hat = np.array([[0, 0, 1]]).T
        T_hat = R@y_hat
        N_hat = R@z_hat

        T_proj_mat = T_hat@(T_hat.T)
        N_proj_mat = N_hat@(N_hat.T)

        g = self._plant.gravity_field().gravity_vector()

        # Calculate distances
        manipulator_p = np.array([poses[self.finger_idx].translation()[0:3]]).T
        link_p = np.array([poses[ll_idx].translation()[0:3]]).T
        # Origin is at end of link, on near side
        link_p -= self.paper.height/2 * N_hat
        link_p += self.paper.link_width/2 * T_hat
        M_d = manipulator_p - link_p
        C_d = R_inv@M_d
        _, d_T, d_N = C_d.flatten()  # Flatten required to make sure these are scalars

        # Calculate forces
        M_F_G = self.paper.link_mass*g
        tau_O = -(self.paper.stiffness*theta_x +
                  self.paper.damping*omega_x)
        lever_arm = self.paper.link_width - np.abs(d_T)
        M_F_O = tau_O/lever_arm
        M_F_O *= N_hat

        # Change frames
        C_F_G = R_inv@M_F_G
        _, F_GT, F_GN = C_F_G.flatten()
        C_F_O = R_inv@M_F_O
        _, F_OT, F_ON = C_F_O.flatten()
        # Centripetal force
        F_OT -= self.paper.link_mass * \
            (self.paper.link_width/2)*(omega_x/(2*np.pi))**2

        # Calculate controller N hat force
        m_M = constants.FINGER_MASS
        m_L = self.paper.link_mass
        F_CN = self.F_Nd*(m_L+m_M)/m_L - F_ON - F_GN

        # Calculate resulting normal force
        F_N = (m_L*F_CN-m_M*(F_ON+F_GN))/(m_L+m_M)

        F_CT_min = -2*constants.FRICTION*F_N
        F_CT_min += F_OT
        F_CT_min += F_GT
        F_CT_max = 2*constants.FRICTION*F_N
        F_CT_max += F_OT
        F_CT_max += F_GT

        # PROGRAMMING: Eventually, we should max sure we apply some minimum F_N, even if it exceeds the force control target

        # F_CT = -F_OT-F_GT
        F_CT = (F_CT_min + F_CT_max)/2  # Average to be robust
        # If we are not in contact, apply no tangential force.
        if d_N > constants.FINGER_RADIUS + paper.PAPER_HEIGHT/2 + constants.EPSILON:
            F_CT = 0
            # F_CN = self.F_Nd

        # Convert to manipulator frame
        F_C = np.array([[0, F_CT, F_CN]]).T
        F_M = R@F_C

        if self.debug is not None:
            self.debug['N_hats'].append(N_hat)
            self.debug['T_hats'].append(T_hat)
            self.debug['theta_xs'].append(theta_x)
            self.debug['omega_xs'].append(omega_x)
            self.debug['F_Gs'].append(M_F_G)
            self.debug['F_GTs'].append(F_GT)
            self.debug['F_GNs'].append(F_GN)
            self.debug['F_Os'].append(M_F_O)
            self.debug['F_OTs'].append(F_OT)
            self.debug['F_ONs'].append(F_ON)
            self.debug['F_CNs'].append(F_CN)
            self.debug['F_CTs'].append(F_CT)
            self.debug['F_Ms'].append(F_M)
            self.debug['d_Ns'].append(d_N)
            self.debug['d_Ts'].append(d_T)
            self.debug['F_Ns'].append(F_N)

        return F_M.flatten()[1:]


class OptimizationController(FingerController):
    """Fold paper with feedback on position of the past link"""

    # Making these parameters keywords means that
    def __init__(self, plant, paper_, finger_idx, ll_idx):  # avoids name collision with module
        super().__init__(plant, finger_idx)

        self.finger_idx = finger_idx
        self.ll_idx = ll_idx
        self.paper = paper_

        self.plant = plant

        self.forces = [0, 0]
        self.idx = 0

        # time steps in the trajectory optimization
        self.opt_dt = 1000*constants.DT
        self.T = int(constants.TSPAN/(self.opt_dt))
        self.nf = 2
        self.nq = None  # PROGRAMMING: Can I set nq in finger init?

    def get_jacobian(self, context):
        """
        Get Jacobian from actuator torques/forces to end effector forces.
        Should end up being trivial in this case.
        """

        # get reference frames for the given leg and the ground
        finger_frame = self.plant.GetBodyByName("finger_body").body_frame()
        world_frame = self.plant.world_frame()

        # compute Jacobian matrix
        J = self.plant.CalcJacobianTranslationalVelocity(
            context,
            JacobianWrtVariable(0),
            finger_frame,
            [0, 0, 0],
            world_frame,
            world_frame
        )

        # discard y components since we are in 2D
        return J[[0, 2]]

    def constrain_end_effector(self, variables):
        """
        Function used to contrain the "end effector" position in optimization.
        """

        # configuration, velocity, acceleration, force
        assert variables.size == 3 * self.nq + self.nf
        split_at = [self.nq, 2 * self.nq, 3 * self.nq]
        q, qd, _, _ = np.split(  # pylint: disable=unbalanced-tuple-unpacking
            variables, split_at)

        context = self.plant.CreateDefaultContext()
        self.plant.SetPositions(context, q)
        self.plant.SetVelocities(context, qd)

        pose = self.plant.EvalBodyPoseInWorld(
            context, self.plant.GetBodyByName("paper_body", self.ll_idx))
        x_pos = pose.translation()[0]
        z_pos = pose.translation()[2]
        y_rot = pydrake.math.RollPitchYaw_[
            AutoDiffXd](pose.rotation()).vector()[1]

        return [x_pos, z_pos, y_rot]

    def manipulator_equations(self, variables):
        """
        Evaluate manipulator dynamics to see if everything balances.
        """

        # configuration, velocity, acceleration, force
        assert variables.size == 3 * self.nq + self.nf
        split_at = [self.nq, 2 * self.nq, 3 * self.nq]
        q, qd, qdd, f = np.split(  # pylint: disable=unbalanced-tuple-unpacking
            variables, split_at)

        # set state
        context = self.plant.CreateDefaultContext()
        self.plant.SetPositions(context, q)
        self.plant.SetVelocities(context, qd)

        # matrices for the manipulator equations
        M = self.plant.CalcMassMatrixViaInverseDynamics(context)
        Cv = self.plant.CalcBiasTerm(context)
        tauG = self.plant.CalcGravityGeneralizedForces(context)

        # Jacobian of the stance foot
        J = self.get_jacobian(context)

        # return violation of the manipulator equations
        return M.dot(qdd) + Cv - tauG - J.T.dot(f)

    def optimize(self, init_positions):
        """
        Run optimizer to generate control inputs.
        """

        self.plant = self.plant.ToAutoDiffXd()
        self.nq = self.plant.num_positions()

        # minimum and maximum time interval is seconds
        h_min = self.opt_dt*(1 - 1e-9)
        h_max = self.opt_dt*(1 + 1e-9)

        # initialize program
        prog = MathematicalProgram()

        # vector of the time intervals
        # (distances between the T + 1 break points)
        h = prog.NewContinuousVariables(self.T, name='h')

        # system configuration, generalized velocities, and accelerations
        q = prog.NewContinuousVariables(rows=self.T+1, cols=self.nq, name='q')
        qd = prog.NewContinuousVariables(
            rows=self.T+1, cols=self.nq, name='qd')
        qdd = prog.NewContinuousVariables(
            rows=self.T, cols=self.nq, name='qdd')

        # stance-foot force
        f = prog.NewContinuousVariables(rows=self.T, cols=2, name='f')

        # lower and upper bound on the time steps for all t
        prog.AddBoundingBoxConstraint([h_min] * self.T, [h_max] * self.T, h)

        # link the configurations, velocities, and accelerations
        # uses implicit Euler method, https://en.wikipedia.org/wiki/Backward_Euler_method
        for t in range(self.T):
            prog.AddConstraint(eq(q[t+1], q[t] + h[t] * qd[t+1]))
            prog.AddConstraint(eq(qd[t+1], qd[t] + h[t] * qdd[t]))

        # manipulator equations for all t (implicit Euler)
        for t in range(self.T):
            variables = np.concatenate((q[t+1], qd[t+1], qdd[t], f[t]))
            prog.AddConstraint(self.manipulator_equations,
                               lb=[0]*self.nq, ub=[0]*self.nq, vars=variables)

        # Constrain to initial position
        prog.AddConstraint(eq(q[0], init_positions))
        prog.AddConstraint(eq(qd[0], np.zeros_like(init_positions)))

        # Constrain final position
        variables = np.concatenate((q[-1], qd[-1], qdd[-1], f[-1]))
        prog.AddConstraint(self.constrain_end_effector,
                           lb=[-pedestal.PEDESTAL_DEPTH/2,
                               pedestal.PEDESTAL_HEIGHT + paper.PAPER_HEIGHT/2-constants.EPSILON,
                               -constants.EPSILON],
                           ub=[pedestal.PEDESTAL_DEPTH/2,
                               pedestal.PEDESTAL_HEIGHT + paper.PAPER_HEIGHT/2+2*constants.EPSILON,
                               constants.EPSILON],
                           vars=variables)
        prog.AddConstraint(q[-1][0],
                           -pedestal.PEDESTAL_DEPTH / 2,
                           pedestal.PEDESTAL_DEPTH/2)
        prog.AddConstraint(q[-1][2],
                           pedestal.PEDESTAL_HEIGHT + paper.PAPER_HEIGHT/2-constants.EPSILON,
                           pedestal.PEDESTAL_DEPTH/2)

        solver = SnoptSolver()
        self.result = solver.Solve(  # pylint: disable=attribute-defined-outside-init
            prog)

        if not self.result.is_success():
            raise RuntimeError("Optimization failed!")

        self.forces = self.result.GetSolution(f)

    def GetForces(self, poses, vels):
        idx = int(self.idx*constants.DT/self.opt_dt)
        if idx < self.T:
            Fx, Fz = self.forces[idx, :]
        else:
            Fx, Fz = [0, 0]

        self.idx += 1
        return Fx, Fz
