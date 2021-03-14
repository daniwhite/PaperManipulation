# Standard imports
import finger
import paper
import constants
import numpy as np
import sympy as sp


class EdgeController(finger.FingerController):
    """Fold paper with feedback on position of the past link"""

    # Making these parameters keywords means that
    def __init__(self, finger_idx, ll_idx, sys_params, debug=False):
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

        self.init_math()

        # Initialize debug dict if necessary
        if debug:
            self.debug['N_hats'] = []
            self.debug['T_hats'] = []
            self.debug['F_GTs'] = []
            self.debug['F_CNs'] = []
            self.debug['F_CTs'] = []
            self.debug['taus'] = []
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
            self.debug['a_LNds'] = []
            self.debug['dd_d_Nds'] = []
            self.debug['dd_d_Tds'] = []
            self.debug['dd_theta_Mds'] = []
            self.debug['p_CNs'] = []
            self.debug['p_CTs'] = []
            self.debug['p_LNs'] = []
            self.debug['p_LTs'] = []
            self.debug['p_MNs'] = []

    def GetForces(self, poses, vels, contact_point, slip_speed, pen_depth, N_hat):
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

        if contact_point is None:
            F_CN = 10
            F_CT = 0
            tau_M = 0

            # For logging purposes
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

            a_LNd = np.nan
            dd_d_Nd = np.nan
            dd_d_Td = np.nan
            dd_theta_Md = np.nan
            p_CN = np.nan
            p_CT = np.nan
            p_LN = np.nan
            p_LT = np.nan
            p_MN = np.nan
        else:
            pen_vec = pen_depth*N_hat

            # Constants
            w_L = self.w_L
            h_L = paper.PAPER_HEIGHT
            r = finger.RADIUS
            mu = constants.FRICTION
            m_M = finger.MASS
            m_L = self.m_L
            I_L = self.I_L
            I_M = self.I_M
            b_J = self.b_J
            k_J = self.k_J

            # Positions
            p_C = np.array([contact_point]).T
            p_CT = get_T_proj(p_C)
            p_CN = get_N_proj(p_C)

            p_L = np.array([poses[self.ll_idx].translation()[0:3]]).T
            p_LT = get_T_proj(p_L)
            p_LN = get_N_proj(p_L)

            angle_axis = poses[self.ll_idx].rotation().ToAngleAxis()
            theta_L = angle_axis.angle()
            if sum(angle_axis.axis()) < 0:
                theta_L *= -1

            p_M = np.array([poses[self.finger_idx].translation()[0:3]]).T
            p_MN = get_N_proj(p_M)

            p_LLE = N_hat * -h_L/2 + T_hat * w_L/2
            p_LE = p_L + p_LLE

            d = p_C - p_LE + pen_vec/2
            d_T = get_T_proj(d)
            d_N = get_N_proj(d)

            p_MConM = p_C - p_M
            p_LConL = p_C - p_L

            # Velocities
            d_theta_L = vels[self.ll_idx].rotational()[0]
            d_theta_M = vels[self.finger_idx].rotational()[0]
            omega_vec_L = np.array([[d_theta_L, 0, 0]]).T
            omega_vec_M = np.array([[d_theta_M, 0, 0]]).T

            v_L = np.array([vels[self.ll_idx].translational()[0:3]]).T
            v_L + np.cross(omega_vec_L, p_LConL, axis=0)
            v_LN = get_N_proj(v_L)
            v_LT = get_T_proj(v_L)

            v_WConL = v_L + np.cross(omega_vec_L, p_LConL, axis=0)

            v_M = np.array([vels[self.finger_idx].translational()[0:3]]).T
            v_MN = get_N_proj(v_M)
            v_MT = get_T_proj(v_M)
            v_WConM = v_M + np.cross(omega_vec_M, p_MConM, axis=0)

            d_d_T = -d_theta_L*h_L/2-d_theta_L*r - v_LT + v_MT + d_theta_L*d_N
            d_d_N = -d_theta_L*w_L/2-v_LN+v_MN-d_theta_L*d_T

            v_S = np.matmul(T_hat.T, (v_WConM - v_WConL))[0, 0]

            # Targets
            dd_d_Nd = self.get_dd_d_Nd()
            dd_d_Td = self.get_dd_Td(d_T, d_d_T)
            a_LNd = self.get_a_LNd()
            dd_theta_Md = self.dd_theta_Md()

            # Forces
            stribeck_mu = stribeck(mu, mu, slip_speed/self.v_stiction)
            stribeck_sign_L = np.sign(v_S)
            mu_SM = -stribeck_mu * stribeck_sign_L
            mu_SL = stribeck_mu * stribeck_sign_L

            # Gravity
            F_G = np.array([[0, 0, -m_L*constants.g]]).T
            F_GT = get_T_proj(F_G)
            F_GN = get_N_proj(F_G)

            F_CN = self.get_F_CN(F_GN, I_L, dd_d_Nd, d_theta_L, d_d_T, mu_SL, mu_SM, theta_L,
                                 a_LNd, b_J, h_L, k_J, m_L, m_M, p_CN, p_CT, p_LN, p_LT, r, w_L, d_N, d_T)

            F_CT = self.get_F_CT(F_GN, I_L, dd_d_Td, d_theta_L, d_d_N, mu_SL, mu_SM, theta_L,
                                 a_LNd, b_J, h_L, k_J, m_L, m_M, p_CN, p_CT, p_LN, p_LT, r, w_L, d_N, d_T)

            tau_M = self.get_tau_M(p_LT, I_M, theta_L, p_CT, I_L, a_LNd, dd_theta_Md,
                                   F_GN, b_J, p_CN, d_theta_L, k_J, p_MN, mu_SM, w_L, mu_SL, p_LN, m_L)
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
            # self.debug['a_LNds'].append(a_LNd)
            self.debug['d_Ns'].append(d_N)
            self.debug['d_Ts'].append(d_T)
            self.debug['d_d_Ns'].append(d_d_N)
            self.debug['d_theta_Ls'].append(d_theta_L)
            # self.debug['dd_d_Tds'].append(dd_d_Td)
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

            self.debug['a_LNds'].append(a_LNd)
            self.debug['dd_d_Nds'].append(dd_d_Nd)
            self.debug['dd_d_Tds'].append(dd_d_Td)
            self.debug['dd_theta_Mds'].append(dd_theta_Md)

            self.debug['p_CNs'].append(p_CN)
            self.debug['p_CTs'].append(p_CT)
            self.debug['p_LNs'].append(p_LN)
            self.debug['p_LTs'].append(p_LT)
            self.debug['p_MNs'].append(p_MN)

        return F_M.flatten()[1], F_M.flatten()[2], tau_M

    def get_dd_d_Nd(self):
        return 0

    def get_dd_Td(self, d_T, d_d_T):
        Kp = 1000000
        Kd = 1000
        return Kp*(self.d_Td - d_T) - Kd*d_d_T

    def get_a_LNd(self):
        return 20

    def dd_theta_Md(self):
        return 0

    def init_math(self):
        self.alg = {}
        self.alg['inputs'] = {}

        # PROGRAMMING: Do this with Drake instead?
        # Physical and geometric quantities
        m_L = sp.symbols(r"m_L")
        self.alg['inputs']['m_L'] = m_L
        m_M = sp.symbols(r"m_M")
        self.alg['inputs']['m_M'] = m_M
        w_L = sp.symbols(r"w_L")
        self.alg['inputs']['w_L'] = w_L
        I_L = sp.symbols(r"I_L")
        self.alg['inputs']['I_L'] = I_L
        h_L = sp.symbols(r"h_L")
        self.alg['inputs']['h_L'] = h_L
        mu = sp.symbols(r"\mu")
        self.alg['inputs']['mu'] = mu
        r = sp.symbols(r"r")
        self.alg['inputs']['r'] = r
        g = sp.symbols(r"g")
        self.alg['inputs']['g'] = g
        I_M = sp.symbols(r"I_M")
        self.alg['inputs']['I_M'] = I_M

        # Friction coefficients
        mu_SL = sp.symbols(r"\mu_{SL}")
        self.alg['inputs']['mu_SL'] = mu_SL
        mu_SM = sp.symbols(r"\mu_{SM}")
        self.alg['inputs']['mu_SM'] = mu_SM

        # System gains
        b_J = sp.symbols(r"b_J")
        self.alg['inputs']['b_J'] = b_J
        k_J = sp.symbols(r"k_J")
        self.alg['inputs']['k_J'] = k_J

        # Positions
        p_CN = sp.symbols(r"p_{CN}")
        self.alg['inputs']['p_CN'] = p_CN
        p_CT = sp.symbols(r"p_{CT}")
        self.alg['inputs']['p_CT'] = p_CT
        p_MN = sp.symbols(r"p_{MN}")
        self.alg['inputs']['p_MN'] = p_MN
        p_MT = sp.symbols(r"p_{MT}")
        self.alg['inputs']['p_MT'] = p_MT
        p_LN = sp.symbols(r"p_{LN}")
        self.alg['inputs']['p_LN'] = p_LN
        p_LT = sp.symbols(r"p_{LT}")
        self.alg['inputs']['p_LT'] = p_LT
        theta_L = sp.symbols(r"\theta_L")
        self.alg['inputs']['theta_L'] = theta_L
        theta_M = sp.symbols(r"\theta_M")
        self.alg['inputs']['theta_M'] = theta_M
        d_T = sp.symbols(r"d_T")
        self.alg['inputs']['d_T'] = d_T
        d_N = sp.symbols(r"d_N")
        self.alg['inputs']['d_N'] = d_N

        # Velocities
        v_CN = sp.symbols(r"v_{CN}")
        self.alg['inputs']['v_CN'] = v_CN
        v_CT = sp.symbols(r"v_{CT}")
        self.alg['inputs']['v_CT'] = v_CT
        v_MN = sp.symbols(r"v_{MN}")
        self.alg['inputs']['v_MN'] = v_MN
        v_MT = sp.symbols(r"v_{MT}")
        self.alg['inputs']['v_MT'] = v_MT
        v_LN = sp.symbols(r"v_{LN}")
        self.alg['inputs']['v_LN'] = v_LN
        v_LT = sp.symbols(r"v_{LT}")
        self.alg['inputs']['v_LT'] = v_LT
        d_theta_L = sp.symbols(r"\dot\theta_L")
        self.alg['inputs']['d_theta_L'] = d_theta_L
        d_theta_M = sp.symbols(r"\dot\theta_M")
        self.alg['inputs']['d_theta_M'] = d_theta_M
        d_d_T = sp.symbols(r"\dot{d}_T")
        self.alg['inputs']['d_d_T'] = d_d_T
        d_d_N = sp.symbols(r"\dot{d}_N")
        self.alg['inputs']['d_d_N'] = d_d_N

        # Input forces
        F_GT = sp.symbols(r"F_{GT}")
        self.alg['inputs']['F_GT'] = F_GT
        F_GN = sp.symbols(r"F_{GN}")
        self.alg['inputs']['F_GN'] = F_GN
        a_LNd = sp.symbols(r"a_{LNd}")

        # Control inputs
        self.alg['inputs']['a_LNd'] = a_LNd
        dd_d_Nd = sp.symbols(r"\ddot{d}_{Nd}")
        self.alg['inputs']['dd_d_Nd'] = dd_d_Nd
        dd_d_Td = sp.symbols(r"\ddot{d}_{Td}")
        self.alg['inputs']['dd_d_Td'] = dd_d_Td
        dd_theta_Md = sp.symbols(r"\ddot\theta_{Md}")
        self.alg['inputs']['dd_theta_Md'] = dd_theta_Md

        # # Geometric quantities
        # m_L, m_M, w_L, I_L, h_L, mu, r, g = sp.symbols(
        #     "m_L m_M w_L I_L h_L \mu r g")
        # inputs = [m_L, m_M, w_L, I_L, h_L, mu, r, g]
        # I_M = sp.symbols("I_M")
        # inputs.append(I_M)
        # mu_SL, mu_SM = sp.symbols("\mu_{SL} \mu_{SM}")

        # # System gains
        # b_J, k_J = sp.symbols("b_J k_J")
        # inputs.append(b_J)
        # inputs.append(k_J)

        # # Positions
        # p_CN = sp.symbols(r"p_{CN}")
        # inputs.append(p_CN)
        # p_CT = sp.symbols(r"p_{CT}")
        # inputs.append(p_CT)
        # p_MN = sp.symbols(r"p_{MN}")
        # inputs.append(p_MN)
        # p_MT = sp.symbols(r"p_{MT}")
        # inputs.append(p_MT)
        # p_LN = sp.symbols(r"p_{LN}")
        # inputs.append(p_LN)
        # p_LT = sp.symbols(r"p_{LT}")
        # inputs.append(p_LT)
        # theta_L = sp.symbols(r"\theta_L")
        # inputs.append(theta_L)
        # theta_M = sp.symbols(r"\theta_M")
        # inputs.append(theta_M)
        # d_T = sp.symbols(r"{d}_T")
        # inputs.append(d_T)
        # d_N = sp.symbols(r"{d}_N")
        # inputs.append(d_N)

        # # Velocities
        # v_CN = sp.symbols(r"v_{CN}")
        # inputs.append(v_CN)
        # v_CT = sp.symbols(r"v_{CT}")
        # inputs.append(v_CT)
        # v_MN = sp.symbols(r"v_{MN}")
        # inputs.append(v_MN)
        # v_MT = sp.symbols(r"v_{MT}")
        # inputs.append(v_MT)
        # v_LN = sp.symbols(r"v_{LN}")
        # inputs.append(v_LN)
        # v_LT = sp.symbols(r"v_{LT}")
        # inputs.append(v_LT)
        # d_theta_L = sp.symbols(r"\dot\theta_L")
        # inputs.append(d_theta_L)
        # d_theta_M = sp.symbols(r"\dot\theta_M")
        # inputs.append(d_theta_M)
        # d_d_T = sp.symbols(r"\dot{d}_T")
        # inputs.append(d_d_T)
        # d_d_N = sp.symbols(r"\dot{d}_N")
        # inputs.append(d_d_N)

        # # Vectors which aren't determined by the force balance
        # F_GT, F_GN = sp.symbols("F_{GT} F_{GN}")
        # inputs.append(F_GT)
        # inputs.append(F_GN)

        # # Control targest
        # a_LNd, dd_d_Nd, dd_d_Td, dd_theta_Md = sp.symbols(
        #     r"a_{LNd} \ddot{d}_{Nd} \ddot{d}_{Td} \ddot{\theta}_{Md}")
        # inputs.append(a_LNd)
        # inputs.append(dd_d_Nd)
        # inputs.append(dd_d_Td)
        # inputs.append(dd_theta_Md)

        outputs = [
            a_LT, a_LN, a_MT, a_MN, dd_theta_L, dd_theta_M, F_OT, F_ON, tau_O, F_NM, F_FL, F_FM, dd_d_N, dd_d_T, F_NL, F_CN, F_CT, tau_M
        ] = sp.symbols(
            r"a_{LT} a_{LN} a_{MT} a_{MN} \ddot\theta_L \ddot\theta_M  F_{OT} F_{ON} \tau_O F_{NM} F_{FL} F_{FM} \ddot{d}_N \ddot{d}_T F_{NL} F_{CN}, F_{CT} \tau_M"
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
            # Joint constraint on link tangential acceleration
            [a_LT, -(w_L/2)*d_theta_L**2, ],
            # Joint constraint on link normal acceleration
            [a_LN, dd_theta_L*w_L/2, ],
            # Link moment balance
            [I_L*dd_theta_L, (-w_L/2)*F_ON - (p_CN-p_LN) * \
             F_FL + (p_CT-p_LT)*F_NL + tau_O, ],
            # Manipulator moment balance
            [I_M*dd_theta_M, tau_M-F_FM*(p_CN-p_MN), ],
            # Torque from object
            [tau_O, -b_J*d_theta_L-k_J*theta_L],
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

    def get_F_CN(self, F_GN, I_L, dd_d_Nd, d_theta_L, d_d_T, mu_SL, mu_SM, theta_L, a_LNd, b_J, h_L, k_J, m_L, m_M, p_CN, p_CT, p_LN, p_LT, r, w_L, d_N, d_T):
        return self.F_CN_exp.subs([
            (self.alg['inputs']['F_GN'], F_GN),
            (self.alg['inputs']['I_L'], I_L),
            (self.alg['inputs']['dd_d_Nd'], dd_d_Nd),
            (self.alg['inputs']['d_theta_L'], d_theta_L),
            (self.alg['inputs']['d_d_T'], d_d_T),
            (self.alg['inputs']['mu_SL'], mu_SL),
            (self.alg['inputs']['mu_SM'], mu_SM),
            (self.alg['inputs']['theta_L'], theta_L),
            (self.alg['inputs']['a_LNd'], a_LNd),
            (self.alg['inputs']['b_J'], b_J),
            (self.alg['inputs']['h_L'], h_L),
            (self.alg['inputs']['k_J'], k_J),
            (self.alg['inputs']['m_L'], m_L),
            (self.alg['inputs']['m_M'], m_M),
            (self.alg['inputs']['p_CN'], p_CN),
            (self.alg['inputs']['p_CT'], p_CT),
            (self.alg['inputs']['p_LN'], p_LN),
            (self.alg['inputs']['p_LT'], p_LT),
            (self.alg['inputs']['r'], r),
            (self.alg['inputs']['w_L'], w_L),
            (self.alg['inputs']['d_N'], d_N),
            (self.alg['inputs']['d_T'], d_T),
        ])

    def get_F_CT(self, F_GN, I_L, dd_d_Td, d_theta_L, d_d_N, mu_SL, mu_SM, theta_L, a_LNd, b_J, h_L, k_J, m_L, m_M, p_CN, p_CT, p_LN, p_LT, r, w_L, d_N, d_T):
        return self.F_CT_exp.subs([
            (self.alg['inputs']['F_GN'], F_GN),
            (self.alg['inputs']['I_L'], I_L),
            (self.alg['inputs']['dd_d_Td'], dd_d_Td),
            (self.alg['inputs']['d_theta_L'], d_theta_L),
            (self.alg['inputs']['d_d_N'], d_d_N),
            (self.alg['inputs']['mu_SL'], mu_SL),
            (self.alg['inputs']['mu_SM'], mu_SM),
            (self.alg['inputs']['theta_L'], theta_L),
            (self.alg['inputs']['a_LNd'], a_LNd),
            (self.alg['inputs']['b_J'], b_J),
            (self.alg['inputs']['h_L'], h_L),
            (self.alg['inputs']['k_J'], k_J),
            (self.alg['inputs']['m_L'], m_L),
            (self.alg['inputs']['m_M'], m_M),
            (self.alg['inputs']['p_CN'], p_CN),
            (self.alg['inputs']['p_CT'], p_CT),
            (self.alg['inputs']['p_LN'], p_LN),
            (self.alg['inputs']['p_LT'], p_LT),
            (self.alg['inputs']['r'], r),
            (self.alg['inputs']['w_L'], w_L),
            (self.alg['inputs']['d_N'], d_N),
            (self.alg['inputs']['d_T'], d_T),
        ])

    def get_tau_M(self, p_LT, I_M, theta_L, p_CT, I_L, a_LNd, dd_theta_Md, F_GN, b_J, p_CN, d_theta_L, k_J, p_MN, mu_SM, w_L, mu_SL, p_LN, m_L):
        return self.tau_M_exp.subs([
            (self.alg['inputs']['p_LT'], p_LT),
            (self.alg['inputs']['I_M'], I_M),
            (self.alg['inputs']['theta_L'], theta_L),
            (self.alg['inputs']['p_CT'], p_CT),
            (self.alg['inputs']['I_L'], I_L),
            (self.alg['inputs']['a_LNd'], a_LNd),
            (self.alg['inputs']['dd_theta_Md'], dd_theta_Md),
            (self.alg['inputs']['F_GN'], F_GN),
            (self.alg['inputs']['b_J'], b_J),
            (self.alg['inputs']['p_CN'], p_CN),
            (self.alg['inputs']['d_theta_L'], d_theta_L),
            (self.alg['inputs']['k_J'], k_J),
            (self.alg['inputs']['p_MN'], p_MN),
            (self.alg['inputs']['mu_SM'], mu_SM),
            (self.alg['inputs']['w_L'], w_L),
            (self.alg['inputs']['mu_SL'], mu_SL),
            (self.alg['inputs']['p_LN'], p_LN),
            (self.alg['inputs']['m_L'], m_L),
        ])
