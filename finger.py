"""Functions for creating and controlling the finger manipulator."""

# Standard imports
import pedestal
import paper
import constants
from pydrake.multibody.tree import SpatialInertia, UnitInertia, JacobianWrtVariable
import numpy as np

# Drake imports
import pydrake
from pydrake.all import \
    RigidTransform, SpatialVelocity, MathematicalProgram, eq, SnoptSolver, AutoDiffXd, BodyIndex, ProximityProperties, ContactResults


# Imports of other project files


def AddFinger(plant, init_y, init_z):
    """Adds the manipulator."""
    radius = constants.FINGER_RADIUS
    finger = plant.AddModelInstance("finger")

    # Add false body at the origin that the finger
    plant.AddRigidBody("false_body1", finger, SpatialInertia(
        0, [0, 0, 0], UnitInertia(0, 0, 0)))
    plant.AddRigidBody("false_body2", finger, SpatialInertia(
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
        proximity_props = ProximityProperties()
        proximity_props.AddProperty(
            "material",
            "coulomb_friction",
            pydrake.multibody.plant.CoulombFriction(constants.FRICTION, constants.FRICTION))
        col_geom = plant.RegisterCollisionGeometry(
            finger_body, RigidTransform(),
            shape,
            "finger_body",
            proximity_props)
        plant.RegisterVisualGeometry(
            finger_body, RigidTransform(), shape, "finger_body", [.9, .5, .5, 1.0])

    # Add control joins for y and z movement
    finger_y = plant.AddJoint(pydrake.multibody.tree.PrismaticJoint(
        "finger_y",
        plant.world_frame(),
        plant.GetFrameByName("false_body1"), [0, 1, 0], -1, 1))
    plant.AddJointActuator("finger_y", finger_y)
    finger_y.set_default_translation(init_y)
    finger_z = plant.AddJoint(pydrake.multibody.tree.PrismaticJoint(
        "finger_z",
        plant.GetFrameByName("false_body1"),
        plant.GetFrameByName("false_body2"), [0, 0, 1], -1, 1))
    finger_z.set_default_translation(init_z)
    plant.AddJointActuator("finger_z", finger_z)
    finger_x = plant.AddJoint(pydrake.multibody.tree.RevoluteJoint(
        "finger_x",
        plant.GetFrameByName("false_body2"),
        plant.GetFrameByName("finger_body"),
        [1, 0, 0],
        damping=0
    ))
    finger_x.set_default_angle(0)
    plant.AddJointActuator("finger_x", finger_x)

    return finger, finger_body, col_geom


class FingerController(pydrake.systems.framework.LeafSystem):
    """Base class for implementing a controller at the finger."""

    def __init__(self, plant, finger_idx, ll_idx):
        pydrake.systems.framework.LeafSystem.__init__(self)
        self._plant = plant

        self.DeclareAbstractInputPort(
            "poses", pydrake.common.value.AbstractValue.Make([RigidTransform(), RigidTransform()]))
        self.DeclareAbstractInputPort(
            "vels", pydrake.common.value.AbstractValue.Make([SpatialVelocity(), SpatialVelocity()]))
        self.DeclareAbstractInputPort(
            "contact_results",
            pydrake.common.value.AbstractValue.Make(ContactResults()))
        self.DeclareVectorOutputPort(
            "finger_actuation", pydrake.systems.framework.BasicVector(3),
            self.CalcOutput)

        self.finger_idx = finger_idx
        self.ll_idx = ll_idx
        self.debug = {}
        self.debug['times'] = []

    def GetForces(self, poses, vels, contact_point):
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

        contact_results = self.get_input_port(2).Eval(context)
        contact_point = None
        slip_speed = None
        pen_depth = None
        for i in range(contact_results.num_point_pair_contacts()):
            point_pair_contact_info = \
                contact_results.point_pair_contact_info(i)

            a_idx = int(point_pair_contact_info.bodyA_index())
            b_idx = int(point_pair_contact_info.bodyB_index())

            if ((a_idx == self.ll_idx) and (b_idx == self.finger_idx) or
                    (a_idx == self.finger_idx) and (b_idx == self.ll_idx)):
                contact_point = point_pair_contact_info.contact_point()
                slip_speed = point_pair_contact_info.slip_speed()
                pen_point_pair = point_pair_contact_info.point_pair()
                pen_depth = pen_point_pair.depth

        self.debug['times'].append(context.get_time())

        [fy, fz, tau] = self.GetForces(
            poses, vels, contact_point, slip_speed, pen_depth)
        #
        # fy, fz = self.GetForces(poses, vels, contact_point)
        # tau = 0
        out = np.concatenate(
            ([fy, fz] - constants.FINGER_MASS*g, [tau]))
        output.SetFromVector(out)


class BlankController(FingerController):
    """Fold paper with feedback on position of the past link"""

    # Making these parameters keywords means that
    def __init__(self, plant, finger_idx):
        super().__init__(plant, finger_idx)

    def GetForces(self, poses, vels, contact_point):
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

    def GetForces(self, poses, vels, contact_point):
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
    def __init__(self, plant, paper_, finger_idx, ll_idx, F_Nd, v_stiction, debug=False):
        super().__init__(plant, finger_idx, ll_idx)
        self.paper = paper_

        # Control parameters
        # self.K = K
        self.F_Nd = F_Nd
        # self.d_d = d_d

        self.bar_d_T = None
        self.last_d_T = 0
        self.last_d_N = 0
        self.v_stiction = v_stiction

        # PROGRAMMING: Reintroduce paper parameters

        # Initialize debug dict if necessary
        if debug:
            self.debug['N_hats'] = []
            self.debug['T_hats'] = []
            self.debug['F_GTs'] = []
            # self.debug['F_GNs'] = []
            self.debug['F_CNs'] = []
            self.debug['F_CTs'] = []
            self.debug['taus'] = []
            # self.debug['d_Ts'] = []
            # self.debug['d_Ns'] = []

            self.debug['F_GNs'] = []
            self.debug['I_Ls'] = []
            self.debug['a_LNds'] = []
            self.debug['d_Ns'] = []
            self.debug['d_Ts'] = []
            self.debug['d_d_Ns'] = []
            self.debug['d_theta_Ls'] = []
            self.debug['dd_d_Tds'] = []
            self.debug['h_Ls'] = []
            self.debug['m_Ms'] = []
            self.debug['mu_SLs'] = []
            self.debug['mu_SMs'] = []
            self.debug['p_CTs'] = []
            self.debug['p_LTs'] = []
            self.debug['rs'] = []
            self.debug['w_Ls'] = []

            self.debug['v_LNs'] = []
            self.debug['v_MNs'] = []

    def GetForces(self, poses, vels, contact_point, slip_speed, pen_depth):
        # Directions
        R = poses[self.ll_idx].rotation()
        y_hat = np.array([[0, 1, 0]]).T
        z_hat = np.array([[0, 0, 1]]).T
        T_hat = R@y_hat
        N_hat = R@z_hat

        T_proj_mat = T_hat@(T_hat.T)
        N_proj_mat = N_hat@(N_hat.T)

        # Helper functions
        def get_T_proj(vec):
            T_vec = np.matmul(T_proj_mat, vec)
            T_mag = np.linalg.norm(T_vec)
            T_sgn = np.sign(np.matmul(T_hat.T, T_vec))
            T = T_mag.flatten()*T_sgn.flatten()
            return T

        def get_N_proj(vec):
            N_vec = np.matmul(N_proj_mat, vec)
            N_mag = np.linalg.norm(N_vec)
            N_sgn = np.sign(np.matmul(N_hat.T, N_vec))
            N = N_mag.flatten()*N_sgn.flatten()
            return N

        def step5(x):
            '''Python version of MultibodyPlant::StribeckModel::step5 method'''
            x3 = x * x * x
            return x3 * (10 + x * (6 * x - 15))

        def stribeck(us, uk, v):
            '''
            Python version of MultibodyPlant::StribeckModel::ComputeFrictionCoefficient

            From
            https://github.com/RobotLocomotion/drake/blob/b09e40db4b1c01232b22f7705fb98aa99ef91f87/multibody/plant/images/stiction.py
            '''
            u = uk
            if v < 1:
                u = us * step5(v)
            elif (v >= 1) and (v < 3):
                u = us - (us - uk) * step5((v - 1) / 2)
            return u

        if contact_point is None:
            F_CN = 10
            F_CT = 0
            tau_M = 0

            F_GT = np.nan
            F_GN = np.nan
            d_T = np.nan
            d_N = np.nan

            F_GN = np.nan
            I_L = np.nan
            a_LNd = np.nan
            d_N = np.nan
            d_T = np.nan
            d_d_N = np.nan
            d_theta_L = np.nan
            dd_d_Td = np.nan
            h_L = np.nan
            m_M = np.nan
            mu_SL = np.nan
            mu_SM = np.nan
            p_CT = np.nan
            p_LT = np.nan
            r = np.nan
            w_L = np.nan
            v_LN = np.nan
            v_MN = np.nan
        else:
            pen_vec = pen_depth*N_hat

            # Constants
            w_L = self.paper.link_width
            I_L = self.paper.plant.get_body(
                BodyIndex(self.paper.get_free_edge_idx())).default_rotational_inertia().CalcPrincipalMomentsOfInertia()[0]
            I_M = self.paper.plant.get_body(BodyIndex(
                self.finger_idx)).default_rotational_inertia().CalcPrincipalMomentsOfInertia()[0]
            h_L = paper.PAPER_HEIGHT
            r = constants.FINGER_RADIUS
            mu = constants.FRICTION
            m_M = constants.FINGER_MASS
            m_L = self.paper.link_mass

            # Positions
            p_C = np.array([contact_point]).T
            p_CT = get_T_proj(p_C)
            # p_CN = get_N_proj(p_C)
            p_L = np.array([poses[self.ll_idx].translation()[0:3]]).T
            p_LT = get_T_proj(p_L)
            # p_LN = get_N_proj(p_L)
            # p_M = np.array([poses[self.finger_idx].translation()[0:3]]).T
            p_LLE = N_hat * -h_L/2 + T_hat * w_L/2
            p_LE = p_L + p_LLE
            d = p_C - p_LE + pen_vec/2
            d_T = get_T_proj(d)
            d_N = get_N_proj(d)
            # p_MConM = p_C - p_M
            # p_MConMN = get_N_proj(p_MConM)
            # p_LConL = p_C - p_L
            # p_LConLN = get_N_proj(p_LConL)
            # r_T = get_T_proj(p_C - p_L)

            # Velocities
            d_theta_L = vels[self.ll_idx].rotational()[0]
            d_theta_M = vels[self.finger_idx].rotational()[0]

            v_L = np.array([vels[self.ll_idx].translational()[0:3]]).T
            v_LN = get_N_proj(v_L)
            v_LT = get_T_proj(v_L)
            v_M = np.array([vels[self.finger_idx].translational()[0:3]]).T
            v_MN = get_N_proj(v_M)
            v_MT = get_T_proj(v_M)
            # v_MN = get_N_proj(v_M)
            # v_WConM = v_M + np.cross(omega_vec_M, p_MConM, axis=0)
            # v_WConMN = get_N_proj(v_WConM)
            d_d_T = -d_theta_L*h_L/2-d_theta_L*r - v_LT + v_MT + d_theta_L*d_N
            d_d_N = -d_theta_L*w_L/2-v_LN+v_MN-d_theta_L*d_T

            # Targets
            dd_d_Nd = 0
            dd_d_Td = 0
            a_LNd = self.F_Nd/self.paper.link_mass
            dd_theta_Md = 0

            # Forces
            # F_N = np.abs((F_FL*h_L*w_L + F_{GN}*w_L**2 - 4*I_L*a_{LNd} - a_{LNd}*m_L*w_L**2)/(2*p_{CT}*w_L - 2*p_{LT}*w_L + w_L**2))
            stribeck_mu = stribeck(mu, mu, slip_speed/self.v_stiction)
            mu_SM = -stribeck_mu * np.sign(d_d_T)
            mu_SL = stribeck_mu * np.sign(d_d_T)

            # Gravity
            g = 9.80665
            F_G = np.array([[0, 0, -self.paper.link_mass*g]]).T
            F_GT = get_T_proj(F_G)
            F_GN = get_N_proj(F_G)

            F_CN = (-F_GN*w_L**2 + 4*I_L*a_LNd + dd_d_Nd*mu_SL*h_L*m_M*w_L + 2*dd_d_Nd*m_M*p_CT*w_L - 2*dd_d_Nd*m_M*p_LT*w_L + dd_d_Nd*m_M*w_L**2 + d_theta_L**2*mu_SL*h_L**2*m_M*w_L/2 + d_theta_L**2*mu_SL*h_L*m_M*r*w_L - d_theta_L**2*mu_SL*h_L*m_M*w_L*d_N + d_theta_L**2*h_L*m_M*p_CT*w_L - d_theta_L**2*h_L*m_M*p_LT*w_L + d_theta_L**2*h_L*m_M*w_L**2/2 + 2*d_theta_L**2*m_M*p_CT*r*w_L - 2*d_theta_L**2*m_M*p_CT*w_L*d_N - 2*d_theta_L**2*m_M*p_LT*r*w_L + 2*d_theta_L**2*m_M *
                    p_LT*w_L*d_N + d_theta_L**2*m_M*r*w_L**2 - d_theta_L**2*m_M*w_L**2*d_N + 2*d_theta_L*d_d_T*mu_SL*h_L*m_M*w_L + 4*d_theta_L*d_d_T*m_M*p_CT*w_L - 4*d_theta_L*d_d_T*m_M*p_LT*w_L + 2*d_theta_L*d_d_T*m_M*w_L**2 + 2*mu_SL*a_LNd*h_L*m_M*w_L + 2*mu_SL*a_LNd*h_L*m_M*d_T + a_LNd*m_L*w_L**2 + 4*a_LNd*m_M*p_CT*w_L + 4*a_LNd*m_M*p_CT*d_T - 4*a_LNd*m_M*p_LT*w_L - 4*a_LNd*m_M*p_LT*d_T + 2*a_LNd*m_M*w_L**2 + 2*a_LNd*m_M*w_L*d_T)/(w_L*(mu_SL*h_L + 2*p_CT - 2*p_LT + w_L))

            F_CT = (F_GN*mu_SM*w_L**2 - 4*I_L*mu_SM*a_LNd + dd_d_Td*mu_SL*h_L*m_M*w_L + 2*dd_d_Td*m_M*p_CT*w_L - 2*dd_d_Td*m_M*p_LT*w_L + dd_d_Td*m_M*w_L**2 - d_theta_L**2*mu_SL*h_L*m_M*w_L**2 - d_theta_L**2*mu_SL*h_L*m_M*w_L*d_T - 2*d_theta_L**2*m_M*p_CT*w_L**2 - 2*d_theta_L**2*m_M*p_CT*w_L*d_T + 2*d_theta_L**2*m_M*p_LT*w_L**2 + 2*d_theta_L**2*m_M*p_LT*w_L*d_T - d_theta_L**2*m_M*w_L**3 - d_theta_L**2*m_M*w_L**2*d_T - 2*d_theta_L*d_d_N*mu_SL*h_L*m_M *
                    w_L - 4*d_theta_L*d_d_N*m_M*p_CT*w_L + 4*d_theta_L*d_d_N*m_M*p_LT*w_L - 2*d_theta_L*d_d_N*m_M*w_L**2 + mu_SL*a_LNd*h_L**2*m_M + 2*mu_SL*a_LNd*h_L*m_M*r - 2*mu_SL*a_LNd*h_L*m_M*d_N - mu_SM*a_LNd*m_L*w_L**2 + 2*a_LNd*h_L*m_M*p_CT - 2*a_LNd*h_L*m_M*p_LT + a_LNd*h_L*m_M*w_L + 4*a_LNd*m_M*p_CT*r - 4*a_LNd*m_M*p_CT*d_N - 4*a_LNd*m_M*p_LT*r + 4*a_LNd*m_M*p_LT*d_N + 2*a_LNd*m_M*r*w_L - 2*a_LNd*m_M*w_L*d_N)/(w_L*(mu_SL*h_L + 2*p_CT - 2*p_LT + w_L))

            tau_M = (-F_GN*mu_SM*r*w_L**2 + 4*I_L*mu_SM*a_LNd*r + I_M*dd_theta_Md*mu_SL*h_L*w_L + 2*I_M*dd_theta_Md*p_CT*w_L - 2 *
                     I_M*dd_theta_Md*p_LT*w_L + I_M*dd_theta_Md*w_L**2 + mu_SM*a_LNd*m_L*r*w_L**2)/(w_L*(mu_SL*h_L + 2*p_CT - 2*p_LT + w_L))
            tau_M = tau_M[0]
            self.last_d_T = d_T
            self.last_d_N = d_N

        F_M = F_CN*N_hat + F_CT*T_hat

        if self.debug is not None:
            self.debug['N_hats'].append(N_hat)
            self.debug['T_hats'].append(T_hat)
            self.debug['F_GTs'].append(F_GT)
            # self.debug['F_GNs'].append(F_GN)
            self.debug['F_CNs'].append(F_CN)
            self.debug['F_CTs'].append(F_CT)
            self.debug['taus'].append(tau_M)
            # self.debug['d_Ts'].append(d_T)
            # self.debug['d_Ns'].append(d_N)

            self.debug['F_GNs'].append(F_GN)
            self.debug['I_Ls'].append(I_L)
            self.debug['a_LNds'].append(a_LNd)
            self.debug['d_Ns'].append(d_N)
            self.debug['d_Ts'].append(d_T)
            self.debug['d_d_Ns'].append(d_d_N)
            self.debug['d_theta_Ls'].append(d_theta_L)
            self.debug['dd_d_Tds'].append(dd_d_Td)
            self.debug['h_Ls'].append(h_L)
            self.debug['m_Ms'].append(m_M)
            self.debug['mu_SLs'].append(mu_SL)
            self.debug['mu_SMs'].append(mu_SM)
            self.debug['p_CTs'].append(p_CT)
            self.debug['p_LTs'].append(p_LT)
            self.debug['rs'].append(r)
            self.debug['w_Ls'].append(w_L)

            self.debug['v_LNs'].append(v_LN)
            self.debug['v_MNs'].append(v_MN)

            # self.debug['p_Cs'].append(p_C)
            # self.debug['ds'].append(d)

        return F_M.flatten()[1], F_M.flatten()[2], tau_M


class OptimizationController(FingerController):
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
