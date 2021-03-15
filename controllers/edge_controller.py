# Standard imports
import finger
import paper
import constants
import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify
import re
from pydrake.all import SpatialForce


class EdgeController(finger.FingerController):
    """Fold paper with feedback on position of the past link"""

    # Making these parameters keywords means that
    def __init__(self, finger_idx, ll_idx, sys_params, jnt_frc_log, debug=False):
        super().__init__(finger_idx, ll_idx)

        self.v_stiction = sys_params['v_stiction']
        self.I_M = sys_params['I_M']
        self.I_L = sys_params['I_L']
        self.w_L = sys_params['w_L']
        self.m_L = sys_params['m_L']
        self.b_J = sys_params['b_J']
        self.k_J = sys_params['k_J']
        self.g = sys_params['g']

        self.d_Td = -0.03  # -0.12
        self.jnt_frc_log = jnt_frc_log
        self.jnt_frc_log.append(SpatialForce(
            np.zeros((3, 1)), np.zeros((3, 1))))

        self.init_math()

        # Initialize debug dict if necessary
        if debug:
            self.debug['F_CNs'] = []
            self.debug['F_CTs'] = []
            self.debug['taus'] = []
            self.debug['dd_d_Nds'] = []
            self.debug['dd_d_Tds'] = []
            self.debug['a_LNds'] = []
            self.debug['dd_theta_Mds'] = []
            self.debug['F_OTs'] = []
            self.debug['F_ONs'] = []
            self.debug['tau_Os'] = []

    def GetForces(self, poses, vels, contact_point, slip_speed, pen_depth, N_hat):
        inputs = {}
        jnt_frcs = self.jnt_frc_log[-1]
        inputs['F_OT'] = F_OT = jnt_frcs.translational()[1]
        inputs['F_ON'] = F_ON = jnt_frcs.translational()[2]
        inputs['tau_O'] = tau_O = jnt_frcs.rotational()[0]

        # Directions
        R = poses[self.ll_idx].rotation()
        y_hat = np.array([[0, 1, 0]]).T
        z_hat = np.array([[0, 0, 1]]).T
        if N_hat is None:
            T_hat = R@y_hat
            N_hat = R@z_hat
        else:
            T_hat = np.matmul(
                np.array([
                    [0,  0, 0],
                    [0,  0, 1],
                    [0, -1, 0],
                ]),
                N_hat)

        T_proj_mat = T_hat@(T_hat.T)
        N_proj_mat = N_hat@(N_hat.T)

        # Helper functions
        def get_T_proj(vec):
            T_vec = np.matmul(T_proj_mat, vec)
            T_mag = np.linalg.norm(T_vec)
            T_sgn = np.sign(np.matmul(T_hat.T, T_vec))
            T = T_mag.flatten()*T_sgn.flatten()
            return T[0]

        def get_N_proj(vec):
            N_vec = np.matmul(N_proj_mat, vec)
            N_mag = np.linalg.norm(N_vec)
            N_sgn = np.sign(np.matmul(N_hat.T, N_vec))
            N = N_mag.flatten()*N_sgn.flatten()
            return N[0]

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

        # Constants
        inputs['w_L'] = w_L = self.w_L
        inputs['h_L'] = h_L = paper.PAPER_HEIGHT
        inputs['r'] = r = finger.RADIUS
        inputs['mu'] = mu = constants.FRICTION
        inputs['m_M'] = finger.MASS
        inputs['m_L'] = self.m_L
        inputs['I_L'] = self.I_L
        inputs['I_M'] = self.I_M
        inputs['b_J'] = self.b_J
        inputs['k_J'] = self.k_J

        # Positions
        p_L = np.array([poses[self.ll_idx].translation()[0:3]]).T
        inputs['p_LT'] = get_T_proj(p_L)
        inputs['p_LN'] = get_N_proj(p_L)

        p_M = np.array([poses[self.finger_idx].translation()[0:3]]).T
        inputs['p_MN'] = get_N_proj(p_M)

        angle_axis = poses[self.ll_idx].rotation().ToAngleAxis()
        theta_L = angle_axis.angle()
        if sum(angle_axis.axis()) < 0:
            theta_L *= -1
        inputs['theta_L'] = theta_L

        p_LLE = N_hat * -h_L/2 + T_hat * w_L/2
        p_LE = p_L + p_LLE

        p_LEM = p_M - p_LE
        p_LEMT = get_T_proj(p_LEM)

        # Velocities
        inputs['d_theta_L'] = d_theta_L = vels[self.ll_idx].rotational()[0]
        inputs['d_theta_M'] = d_theta_M = vels[self.finger_idx].rotational()[
            0]
        omega_vec_L = np.array([[d_theta_L, 0, 0]]).T
        omega_vec_M = np.array([[d_theta_M, 0, 0]]).T

        v_L = np.array([vels[self.ll_idx].translational()[0:3]]).T
        inputs['v_LN'] = v_LN = get_N_proj(v_L)
        inputs['v_LT'] = v_LT = get_T_proj(v_L)

        v_M = np.array([vels[self.finger_idx].translational()[0:3]]).T
        inputs['v_MN'] = v_MN = get_N_proj(v_M)
        inputs['v_MT'] = v_MT = get_T_proj(v_M)

        # Assume for now that link edge is not moving
        # PROGRAMMING: Get rid of this assumption
        v_LEM = v_M
        v_LEMT = get_T_proj(p_LEM)

        if contact_point is None:
            F_CN = 0.1
            F_CT = self.get_pre_contact_F_CT(p_LEMT, v_LEMT)
            tau_M = 0

            # For logging purposes
            a_LNd = np.nan
            dd_d_Nd = np.nan
            dd_d_Td = np.nan
            dd_theta_Md = np.nan
        else:
            pen_vec = pen_depth*N_hat

            # Positions
            p_C = np.array([contact_point]).T
            inputs['p_CT'] = get_T_proj(p_C)
            inputs['p_CN'] = get_N_proj(p_C)

            d = p_C - p_LE + pen_vec/2
            inputs['d_T'] = d_T = get_T_proj(d)
            inputs['d_N'] = d_N = get_N_proj(d)

            p_MConM = p_C - p_M
            p_LConL = p_C - p_L

            # Velocities
            v_WConL = v_L + np.cross(omega_vec_L, p_LConL, axis=0)
            v_WConM = v_M + np.cross(omega_vec_M, p_MConM, axis=0)

            inputs['d_d_T'] = d_d_T = -d_theta_L*h_L/2 - \
                d_theta_L*r - v_LT + v_MT + d_theta_L*d_N
            inputs['d_d_N'] = d_d_N = -d_theta_L*w_L/2-v_LN+v_MN-d_theta_L*d_T

            v_S = np.matmul(T_hat.T, (v_WConM - v_WConL))[0, 0]

            # Targets
            inputs['dd_d_Nd'] = dd_d_Nd = self.get_dd_d_Nd()
            inputs['dd_d_Td'] = dd_d_Td = self.get_dd_d_Td(d_T, d_d_T)
            inputs['a_LNd'] = a_LNd = self.get_a_LNd()
            inputs['dd_theta_Md'] = dd_theta_Md = self.dd_theta_Md()

            # Forces
            stribeck_mu = stribeck(mu, mu, slip_speed/self.v_stiction)
            stribeck_sign_L = np.sign(v_S)
            inputs['mu_SM'] = -stribeck_mu * stribeck_sign_L
            inputs['mu_SL'] = stribeck_mu * stribeck_sign_L

            # Gravity
            F_G = np.array([[0, 0, -self.m_L*constants.g]]).T
            inputs['F_GT'] = get_T_proj(F_G)
            inputs['F_GN'] = get_N_proj(F_G)

            # Pack inputs according to order from algebra
            inps_ = []
            for inp in self.alg_inputs:
                var_name = inp.name
                var_str = self.latex_to_str(var_name)
                inps_.append(inputs[var_str])

            F_CN = self.get_F_CN(inps_)
            F_CT = self.get_F_CT(inps_)
            tau_M = self.get_tau_M(inps_)

        F_M = F_CN*N_hat + F_CT*T_hat

        if self.debug is not None:
            self.debug['F_CNs'].append(F_CN)
            self.debug['F_CTs'].append(F_CT)
            self.debug['taus'].append(tau_M)
            self.debug['dd_d_Nds'].append(dd_d_Nd)
            self.debug['dd_d_Tds'].append(dd_d_Td)
            self.debug['a_LNds'].append(a_LNd)
            self.debug['dd_theta_Mds'].append(dd_theta_Md)
            self.debug['F_OTs'].append(F_OT)
            self.debug['F_ONs'].append(F_ON)
            self.debug['tau_Os'].append(tau_O)

        return F_M.flatten()[1], F_M.flatten()[2], tau_M

    def get_dd_d_Nd(self):
        return 0

    def get_dd_d_Td(self, d_T, d_d_T):
        Kp = 1000000
        Kd = 1000
        return Kp*(self.d_Td - d_T) - Kd*d_d_T

    def get_a_LNd(self):
        return 20

    def dd_theta_Md(self):
        return 0

    def get_pre_contact_F_CT(self, p_LEMT, v_LEMT):
        Kp = 0.1
        Kd = 1
        return Kp*(self.d_Td - p_LEMT) - Kd*v_LEMT

    def latex_to_str(self, sym):
        out = str(sym)
        out = re.sub(r"\\ddot\{([^}]*)\}", r"dd_\1", out)
        out = re.sub(r"\\dot\{([^}]*)\}", r"d_\1", out)
        out = out.replace(r"\ddot", "dd_")
        out = out.replace(r"\dot", "d_")
        out = out.replace(r"{", "").replace(r"}", "").replace("\\", "")
        return out

    def init_math(self):
        self.alg_inputs = []

        # PROGRAMMING: Do this with Drake instead?
        # PROGRAMMING: Can I get some of these lists from sympy?
        # Physical and geometric quantities
        m_L = sp.symbols(r"m_L")
        self.alg_inputs.append(m_L)
        m_M = sp.symbols(r"m_M")
        self.alg_inputs.append(m_M)
        w_L = sp.symbols(r"w_L")
        self.alg_inputs.append(w_L)
        I_L = sp.symbols(r"I_L")
        self.alg_inputs.append(I_L)
        h_L = sp.symbols(r"h_L")
        self.alg_inputs.append(h_L)
        mu = sp.symbols(r"\mu")
        self.alg_inputs.append(mu)
        r = sp.symbols(r"r")
        self.alg_inputs.append(r)
        I_M = sp.symbols(r"I_M")
        self.alg_inputs.append(I_M)

        # Friction coefficients
        mu_SL = sp.symbols(r"\mu_{SL}")
        self.alg_inputs.append(mu_SL)
        mu_SM = sp.symbols(r"\mu_{SM}")
        self.alg_inputs.append(mu_SM)

        # System gains
        b_J = sp.symbols(r"b_J")
        self.alg_inputs.append(b_J)
        k_J = sp.symbols(r"k_J")
        self.alg_inputs.append(k_J)

        # Positions
        p_CN = sp.symbols(r"p_{CN}")
        self.alg_inputs.append(p_CN)
        p_CT = sp.symbols(r"p_{CT}")
        self.alg_inputs.append(p_CT)
        p_MN = sp.symbols(r"p_{MN}")
        self.alg_inputs.append(p_MN)
        p_LN = sp.symbols(r"p_{LN}")
        self.alg_inputs.append(p_LN)
        p_LT = sp.symbols(r"p_{LT}")
        self.alg_inputs.append(p_LT)
        theta_L = sp.symbols(r"\theta_L")
        self.alg_inputs.append(theta_L)
        d_T = sp.symbols(r"d_T")
        self.alg_inputs.append(d_T)
        d_N = sp.symbols(r"d_N")
        self.alg_inputs.append(d_N)

        # Velocities
        v_MN = sp.symbols(r"v_{MN}")
        self.alg_inputs.append(v_MN)
        v_MT = sp.symbols(r"v_{MT}")
        self.alg_inputs.append(v_MT)
        v_LN = sp.symbols(r"v_{LN}")
        self.alg_inputs.append(v_LN)
        v_LT = sp.symbols(r"v_{LT}")
        self.alg_inputs.append(v_LT)
        d_theta_L = sp.symbols(r"\dot\theta_L")
        self.alg_inputs.append(d_theta_L)
        d_theta_M = sp.symbols(r"\dot\theta_M")
        self.alg_inputs.append(d_theta_M)
        d_d_T = sp.symbols(r"\dot{d}_T")
        self.alg_inputs.append(d_d_T)
        d_d_N = sp.symbols(r"\dot{d}_N")
        self.alg_inputs.append(d_d_N)

        # Input forces
        F_GT = sp.symbols(r"F_{GT}")
        self.alg_inputs.append(F_GT)
        F_GN = sp.symbols(r"F_{GN}")
        self.alg_inputs.append(F_GN)
        F_OT, F_ON, tau_O = sp.symbols(r"F_{OT}, F_{ON} \tau_O")
        self.alg_inputs.append(F_OT)
        self.alg_inputs.append(F_ON)
        self.alg_inputs.append(tau_O)

        # Control inputs
        a_LNd = sp.symbols(r"a_{LNd}")
        self.alg_inputs.append(a_LNd)
        dd_d_Nd = sp.symbols(r"\ddot{d}_{Nd}")
        self.alg_inputs.append(dd_d_Nd)
        dd_d_Td = sp.symbols(r"\ddot{d}_{Td}")
        self.alg_inputs.append(dd_d_Td)
        dd_theta_Md = sp.symbols(r"\ddot\theta_{Md}")
        self.alg_inputs.append(dd_theta_Md)

        outputs = [
            a_LT, a_LN, a_MT, a_MN, dd_theta_L, dd_theta_M, F_NM, F_FL, F_FM, dd_d_N, dd_d_T, F_NL, F_CN, F_CT, tau_M
        ] = sp.symbols(
            r"a_{LT} a_{LN} a_{MT} a_{MN} \ddot\theta_L \ddot\theta_M  F_{NM} F_{FL} F_{FM} \ddot{d}_N \ddot{d}_T F_{NL} F_{CN}, F_{CT} \tau_M"
        )
        outputs = list(outputs)

        t = sp.symbols("t")
        theta_L_func = sp.Function(r'\theta_L')(t)
        N_hat = sp.Function(r'\hat N')(theta_L_func)
        T_hat = sp.Function(r'\hat T')(theta_L_func)

        d_T_func = sp.Function(r"d_T")(t)
        d_N_func = sp.Function(r"d_N")(t)
        d_g = d_T_func*T_hat + d_N_func*N_hat

        d_vel_g = sp.diff(d_g, t)

        d_vel_g = d_vel_g.subs(sp.diff(N_hat, t), -
                               sp.diff(theta_L_func, t)*T_hat)
        d_vel_g = d_vel_g.subs(
            sp.diff(T_hat, t), sp.diff(theta_L_func, t)*N_hat)

        d_acc_g = sp.diff(d_vel_g, t)
        d_acc_g = d_acc_g.subs(sp.diff(N_hat, t), -
                               sp.diff(theta_L_func, t)*T_hat)
        d_acc_g = d_acc_g.subs(
            sp.diff(T_hat, t), sp.diff(theta_L_func, t)*N_hat)

        d_acc_cos_g = d_acc_g
        d_acc_cos_g = d_acc_cos_g.subs(sp.diff(theta_L_func, t, t), dd_theta_L)
        d_acc_cos_g = d_acc_cos_g.subs(sp.diff(d_T_func, t, t), dd_d_T)
        d_acc_cos_g = d_acc_cos_g.subs(sp.diff(d_N_func, t, t), dd_d_N)
        d_acc_cos_g = d_acc_cos_g.subs(sp.diff(theta_L_func, t), d_theta_L)
        d_acc_cos_g = d_acc_cos_g.subs(sp.diff(d_T_func, t), d_d_T)
        d_acc_cos_g = d_acc_cos_g.subs(sp.diff(d_N_func, t), d_d_N)
        d_acc_cos_g = d_acc_cos_g.subs(d_T_func, d_T)
        d_acc_cos_g = d_acc_cos_g.subs(d_N_func, d_N)

        dd_d_g_T = d_acc_cos_g.subs(N_hat, 0).subs(T_hat, 1)

        dd_d_g_N = d_acc_cos_g.subs(T_hat, 0).subs(N_hat, 1)

        p_M_func = sp.Function(r"p_M")(t)
        p_L_func = sp.Function(r"p_L")(t)
        v_M = sp.symbols(r"v_M")
        v_L = sp.symbols(r"v_L")
        d_s = (p_M_func + r*N_hat) - (p_L_func + (w_L/2)*T_hat - (h_L/2)*N_hat)

        d_vel_s = sp.diff(d_s, t)
        d_vel_s = d_vel_s.subs(sp.diff(N_hat, t), -
                               sp.diff(theta_L_func, t)*T_hat)
        d_vel_s = d_vel_s.subs(
            sp.diff(T_hat, t), sp.diff(theta_L_func, t)*N_hat)

        d_acc_s = sp.diff(d_vel_s, t)
        d_acc_s = d_acc_s.subs(sp.diff(N_hat, t), -
                               sp.diff(theta_L_func, t)*T_hat)
        d_acc_s = d_acc_s.subs(
            sp.diff(T_hat, t), sp.diff(theta_L_func, t)*N_hat)

        d_acc_cos_s = d_acc_s
        d_acc_cos_s = d_acc_cos_s.subs(sp.diff(theta_L_func, t, t), dd_theta_L)
        d_acc_cos_s = d_acc_cos_s.subs(sp.diff(d_T_func, t, t), dd_d_T)
        d_acc_cos_s = d_acc_cos_s.subs(sp.diff(d_N_func, t, t), dd_d_N)
        d_acc_cos_s = d_acc_cos_s.subs(sp.diff(theta_L_func, t), d_theta_L)
        d_acc_cos_s = d_acc_cos_s.subs(sp.diff(d_T_func, t), d_d_T)
        d_acc_cos_s = d_acc_cos_s.subs(sp.diff(d_N_func, t), d_d_N)
        d_acc_cos_s = d_acc_cos_s.subs(d_T_func, d_T)
        d_acc_cos_s = d_acc_cos_s.subs(d_N_func, d_N)

        dd_d_s_T = d_acc_cos_s.subs(N_hat, 0).subs(T_hat, 1)
        dd_d_s_T = dd_d_s_T.subs(sp.diff(p_M_func, t, t), a_MT)
        dd_d_s_T = dd_d_s_T.subs(sp.diff(p_L_func, t, t), a_LT)
        dd_d_s_T

        dd_d_s_N = d_acc_cos_s.subs(T_hat, 0).subs(N_hat, 1)
        dd_d_s_N = dd_d_s_N.subs(sp.diff(p_M_func, t, t), a_MN)
        dd_d_s_N = dd_d_s_N.subs(sp.diff(p_L_func, t, t), a_LN)
        dd_d_s_N

        nat_eqs = [
            # Link tangential force balance
            [m_L*a_LT, F_FL+F_GT+F_OT],
            # Link normal force balance
            [m_L*a_LN, F_NL + F_GN + F_ON, ],
            # Manipulator tangential force balance
            [m_M*a_MT, F_FM + F_CT, ],
            # Manipulator normal force balance
            [m_M*a_MN, F_NM+F_CN, ],
            # Link moment balance
            [I_L*dd_theta_L, (-w_L/2)*F_ON - (p_CN-p_LN) * \
             F_FL + (p_CT-p_LT)*F_NL + tau_O, ],
            # Manipulator moment balance
            [I_M*dd_theta_M, tau_M-F_FM*(p_CN-p_MN), ],
            # 3rd law normal forces
            [F_NL, -F_NM],
            # Friction relationship L
            [F_FL, mu_SL*F_NL],
            # Friction relationship M
            [F_FM, mu_SM*F_NL],
            # d_T derivative is derivative
            [dd_d_s_T, dd_d_g_T],
            # d_N derivative is derivative
            [dd_d_s_N, dd_d_g_N],
        ]

        art_eqs = [
            [dd_d_N, dd_d_Nd],
            [dd_d_T, dd_d_Td],
            [a_LN, a_LNd],
            [dd_theta_M, dd_theta_Md],
        ]

        env_eqs = nat_eqs + art_eqs

        A = []
        b = []
        for lhs, rhs in env_eqs:
            A_row = []
            b_term = rhs - lhs
            for output_term in outputs:
                try:
                    coeff_L = lhs.coeff(output_term)
                except AttributeError:
                    coeff_L = 0
                try:
                    coeff_R = rhs.coeff(output_term)
                except AttributeError:
                    coeff_R = 0
                coeff = coeff_L - coeff_R
                A_row.append(coeff)
                b_term += coeff * output_term
            A.append(A_row)
            b.append(b_term)
        A = sp.Matrix(A)
        A.simplify()
        b = sp.Matrix([b]).T
        b.simplify()
        x = sp.Matrix([outputs]).T
        x.simplify()

        A_aug = A.row_join(b)
        results = A_aug.rref()[0]
        A_prime = results[:, :-1]
        b_prime = results[:, -1]

        assert A_prime == sp.eye(A_prime.shape[0])

        F_CN_idx = list(x).index(F_CN)
        self.F_CN_exp = b_prime[F_CN_idx]
        F_CT_idx = list(x).index(F_CT)
        self.F_CT_exp = b_prime[F_CT_idx]
        tau_M_idx = list(x).index(tau_M)
        self.tau_M_exp = b_prime[tau_M_idx]

        self.get_F_CN = lambdify([self.alg_inputs], self.F_CN_exp)
        self.get_F_CT = lambdify([self.alg_inputs], self.F_CT_exp)
        self.get_tau_M = lambdify([self.alg_inputs], self.tau_M_exp)

        tau_M_idx = list(x).index(tau_M)
        self.tau_M_exp = b_prime[tau_M_idx]
