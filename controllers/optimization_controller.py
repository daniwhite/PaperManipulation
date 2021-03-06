# Standard imports
import pedestal
import paper
import constants
import finger
from pydrake.multibody.tree import JacobianWrtVariable
import numpy as np

# Drake imports
import pydrake
from pydrake.all import MathematicalProgram, eq, SnoptSolver, AutoDiffXd


class OptimizationController(finger.FingerController):
    """Fold paper with feedback on position of the past link"""

    # Making these parameters keywords means that
    # avoids name collision with module
    def __init__(self, plant, paper_, finger_idx, ll_idx, v_stiction):
        super().__init__(plant, finger_idx)

        self.finger_idx = finger_idx
        self.ll_idx = ll_idx
        self.paper = paper_

        self.plant = plant
        self.v_stiction = v_stiction

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
        q = prog.NewContinuousVariables(
            rows=self.T+1, cols=self.nq, name='q')
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

    def GetForces(self, poses, vels, contact_point):
        idx = int(self.idx*constants.DT/self.opt_dt)
        if idx < self.T:
            Fx, Fz = self.forces[idx, :]
        else:
            Fx, Fz = [0, 0]

        self.idx += 1
        return Fx, Fz
