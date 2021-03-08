# Standard imports
import finger
import paper
import constants
import numpy as np


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
        self.b_P = sys_params['b_P']
        self.g = sys_params['g']

        self.d_Td = -0.03  # -0.12

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
            b_P = self.b_P

            # Positions
            p_C = np.array([contact_point]).T
            p_CT = get_T_proj(p_C)
            p_CN = get_N_proj(p_C)

            p_L = np.array([poses[self.ll_idx].translation()[0:3]]).T
            p_LT = get_T_proj(p_L)
            p_LN = get_N_proj(p_L)

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
        return 0
        # return Kp*(self.d_Td - d_T) - Kd*d_d_T

    def get_a_LNd(self):
        return 20

    def dd_theta_Md(self):
        return 0
