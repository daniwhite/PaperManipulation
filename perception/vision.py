# Drake imports
import pydrake
from pydrake.all import RigidTransform, SpatialVelocity, RollPitchYaw, RotationMatrix
import numpy as np
import plant.paper as paper
import constants
import plant.manipulator as manipulator

class VisionSystem(pydrake.systems.framework.LeafSystem):
    """
    Simulates what we get out of our vision system:
    """

    # PROGRAMMING: Clean up passed around references
    def __init__(self, ll_idx, contact_body_idx):
        pydrake.systems.framework.LeafSystem.__init__(self)
        
        self.ll_idx = ll_idx
        self.contact_body_idx = contact_body_idx

        self.DeclareAbstractInputPort(
            "poses", pydrake.common.value.AbstractValue.Make([RigidTransform(), RigidTransform()]))
        self.DeclareAbstractInputPort(
            "vels", pydrake.common.value.AbstractValue.Make([SpatialVelocity(), SpatialVelocity()]))

        self.DeclareVectorOutputPort(
            "pose_L_translational", pydrake.systems.framework.BasicVector(3),
            self.output_pose_L_translational)
        self.DeclareVectorOutputPort(
            "pose_L_rotational", pydrake.systems.framework.BasicVector(3),
            self.output_pose_L_rotational)
        self.DeclareVectorOutputPort(
            "vel_L_translational", pydrake.systems.framework.BasicVector(3),
            self.output_vel_L_translational)
        self.DeclareVectorOutputPort(
            "vel_L_rotational", pydrake.systems.framework.BasicVector(3),
            self.output_vel_L_rotational)
        self.DeclareVectorOutputPort(
            "pose_M_translational", pydrake.systems.framework.BasicVector(3),
            self.output_pose_M_translational)
        self.DeclareVectorOutputPort(
            "pose_M_rotational", pydrake.systems.framework.BasicVector(3),
            self.output_pose_M_rotational)
        self.DeclareVectorOutputPort(
            "vel_M_translational", pydrake.systems.framework.BasicVector(3),
            self.output_vel_M_translational)
        self.DeclareVectorOutputPort(
            "vel_M_rotational", pydrake.systems.framework.BasicVector(3),
            self.output_vel_M_rotational)

    def output_pose_L_translational(self, context, output):
        poses = self.GetInputPort("poses").Eval(context)
        output.SetFromVector(list(poses[self.ll_idx].translation()))
    
    def output_pose_L_rotational(self, context, output):
        poses = self.GetInputPort("poses").Eval(context)
        output.SetFromVector(
            list(
                RollPitchYaw(
                    poses[self.ll_idx].rotation()
                    ).vector()
            )
        )

    def output_vel_L_translational(self, context, output):
        vels = self.GetInputPort("vels").Eval(context)
        output.SetFromVector(list(vels[self.ll_idx].translational()))
    
    def output_vel_L_rotational(self, context, output):
        vels = self.GetInputPort("vels").Eval(context)
        output.SetFromVector(list(vels[self.ll_idx].rotational()))

    def output_pose_M_translational(self, context, output):
        poses = self.GetInputPort("poses").Eval(context)
        output.SetFromVector(list(poses[self.contact_body_idx].translation()))
    
    def output_pose_M_rotational(self, context, output):
        poses = self.GetInputPort("poses").Eval(context)
        output.SetFromVector(
            list(
                RollPitchYaw(
                    poses[self.contact_body_idx].rotation()
                    ).vector()
            )
        )

    def output_vel_M_translational(self, context, output):
        vels = self.GetInputPort("vels").Eval(context)
        output.SetFromVector(list(vels[self.contact_body_idx].translational()))
    
    def output_vel_M_rotational(self, context, output):
        vels = self.GetInputPort("vels").Eval(context)
        output.SetFromVector(list(vels[self.contact_body_idx].rotational()))

class VisionProcessor(pydrake.systems.framework.LeafSystem):
    """
    Calculates geometry terms based on vision outputs.
    """

    def __init__(self, sys_consts):
        pydrake.systems.framework.LeafSystem.__init__(self)
        # Physical parameters
        self.sys_consts = sys_consts

        self.d_N_thresh = -5e-4
        self.t_contact_start = None

        self.DeclareVectorInputPort(
            "pose_L_translational", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "pose_L_rotational", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "vel_L_translational", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "vel_L_rotational", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "pose_M_translational", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "pose_M_rotational", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "vel_M_translational", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "vel_M_rotational", pydrake.systems.framework.BasicVector(3))

        self.DeclareVectorOutputPort(
            "T_hat", pydrake.systems.framework.BasicVector(3),
            self.output_T_hat)
        self.DeclareVectorOutputPort(
            "N_hat", pydrake.systems.framework.BasicVector(3),
            self.output_N_hat)
        self.DeclareVectorOutputPort(
            "theta_L", pydrake.systems.framework.BasicVector(1),
            self.output_theta_L)
        self.DeclareVectorOutputPort(
            "d_theta_L", pydrake.systems.framework.BasicVector(1),
            self.output_d_theta_L)
        self.DeclareVectorOutputPort(
            "p_LN", pydrake.systems.framework.BasicVector(1),
            self.output_p_LN)
        self.DeclareVectorOutputPort(
            "p_LT", pydrake.systems.framework.BasicVector(1),
            self.output_p_LT)
        self.DeclareVectorOutputPort(
            "v_LN", pydrake.systems.framework.BasicVector(1),
            self.output_v_LN)
        self.DeclareVectorOutputPort(
            "v_MN", pydrake.systems.framework.BasicVector(1),
            self.output_v_MN)
        self.DeclareVectorOutputPort(
            "theta_MX", pydrake.systems.framework.BasicVector(1),
            self.output_theta_MX)
        self.DeclareVectorOutputPort(
            "theta_MY", pydrake.systems.framework.BasicVector(1),
            self.output_theta_MY)
        self.DeclareVectorOutputPort(
            "theta_MZ", pydrake.systems.framework.BasicVector(1),
            self.output_theta_MZ)
        self.DeclareVectorOutputPort(
            "d_theta_MX", pydrake.systems.framework.BasicVector(1),
            self.output_d_theta_MX)
        self.DeclareVectorOutputPort(
            "d_theta_MY", pydrake.systems.framework.BasicVector(1),
            self.output_d_theta_MY)
        self.DeclareVectorOutputPort(
            "d_theta_MZ", pydrake.systems.framework.BasicVector(1),
            self.output_d_theta_MZ)
        self.DeclareVectorOutputPort(
            "F_GT", pydrake.systems.framework.BasicVector(1),
            self.output_F_GT)
        self.DeclareVectorOutputPort(
            "F_GN", pydrake.systems.framework.BasicVector(1),
            self.output_F_GN)
        self.DeclareVectorOutputPort(
            "d_X", pydrake.systems.framework.BasicVector(1),
            self.output_d_X)
        self.DeclareVectorOutputPort(
            "d_d_X", pydrake.systems.framework.BasicVector(1),
            self.output_d_d_X)
        self.DeclareVectorOutputPort(
            "d_N", pydrake.systems.framework.BasicVector(1),
            self.output_d_N)
        self.DeclareVectorOutputPort(
            "d_d_N", pydrake.systems.framework.BasicVector(1),
            self.output_d_d_N)
        self.DeclareVectorOutputPort(
            "d_T", pydrake.systems.framework.BasicVector(1),
            self.output_d_T)
        self.DeclareVectorOutputPort(
            "d_d_T", pydrake.systems.framework.BasicVector(1),
            self.output_d_d_T)
        self.DeclareVectorOutputPort(
            "p_CN", pydrake.systems.framework.BasicVector(1),
            self.output_p_CN)
        self.DeclareVectorOutputPort(
            "p_CT", pydrake.systems.framework.BasicVector(1),
            self.output_p_CT)
        self.DeclareVectorOutputPort(
            "p_MConM", pydrake.systems.framework.BasicVector(3),
            self.output_p_MConM)
        self.DeclareVectorOutputPort(
            "mu_S", pydrake.systems.framework.BasicVector(1),
            self.output_mu_S)
        self.DeclareVectorOutputPort(
            "hats_T", pydrake.systems.framework.BasicVector(1),
            self.output_hats_T)
        self.DeclareVectorOutputPort(
            "s_hat_X", pydrake.systems.framework.BasicVector(1),
            self.output_s_hat_X)
        self.DeclareVectorOutputPort(
            "in_contact", pydrake.systems.framework.BasicVector(1),
            self.output_in_contact)

    # =========================== GET FUNCTIONS ===========================
    def get_T_proj(self, context, vec):
        T_hat = self.calc_T_hat(context)
        return np.matmul(T_hat.T, vec).flatten()[0]
    
    def get_N_proj(self, context, vec):
        N_hat = self.calc_N_hat(context)
        return np.matmul(N_hat.T, vec).flatten()[0]

    # =========================== CALC FUNCTIONS ==========================
    def calc_p_L(self, context):
        p_L = np.array([self.GetInputPort(
            "pose_L_translational").Eval(context)[0:3]]).T
        return p_L
    
    def calc_p_M(self, context):
        p_M = np.array([self.GetInputPort(
            "pose_M_translational").Eval(context)[0:3]]).T
        return p_M
    
    def calc_rot_vec_L(self, context):
        rot_vec_L = np.array([self.GetInputPort(
            "pose_L_rotational").Eval(context)[0:3]]).T
        return rot_vec_L

    def calc_rot_vec_M(self, context):
        rot_vec_M = np.array([self.GetInputPort(
            "pose_M_rotational").Eval(context)[0:3]]).T
        return rot_vec_M
    
    def calc_v_L(self, context):
        v_L = np.array([self.GetInputPort(
            "vel_L_translational").Eval(context)[0:3]]).T
        return v_L
    
    def calc_v_M(self, context):
        v_M = np.array([self.GetInputPort(
            "vel_M_translational").Eval(context)[0:3]]).T
        return v_M
    
    def calc_omega_vec_L(self, context):
        omega_vec_L = np.array([self.GetInputPort(
            "vel_L_rotational").Eval(context)[0:3]]).T
        return omega_vec_L
    
    def calc_omega_vec_M(self, context):
        omega_vec_M = np.array([self.GetInputPort(
            "vel_M_rotational").Eval(context)[0:3]]).T
        return omega_vec_M
    
    def calc_T_hat(self, context):
        rot_vec_L = self.calc_rot_vec_L(context)
        R = RotationMatrix(RollPitchYaw(rot_vec_L)).matrix()
        y_hat = np.array([[0, 1, 0]]).T
        T_hat = R@y_hat
        return T_hat

    def calc_N_hat(self, context):
        rot_vec_L = self.calc_rot_vec_L(context)
        R = RotationMatrix(RollPitchYaw(rot_vec_L)).matrix()
        z_hat = np.array([[0, 0, 1]]).T
        N_hat = R@z_hat
        return N_hat

    def calc_p_C(self, context):
        p_M = self.calc_p_M(context)
        N_hat = self.calc_N_hat(context)
        p_C = p_M + N_hat*manipulator.RADIUS
        return p_C

    def calc_p_LE(self, context):
        N_hat = self.calc_N_hat(context)
        T_hat = self.calc_T_hat(context)
        p_LLE = N_hat * -self.sys_consts.h_L/2 + T_hat * self.sys_consts.w_L/2
        p_L = self.calc_p_L(context)
        p_LE = p_L + p_LLE
        return p_LE

    def calc_d(self, context):
        p_C = self.calc_p_C(context)
        p_LE = self.calc_p_LE(context)
        d = p_C - p_LE
        return d
    
    def calc_v_S(self, context):
        p_L = self.calc_p_L(context)
        p_M = self.calc_p_M(context)
        v_L = self.calc_v_L(context)
        v_M = self.calc_v_M(context)
        omega_vec_L = self.calc_omega_vec_L(context)
        omega_vec_M = self.calc_omega_vec_M(context)
        
        p_C = self.calc_p_C(context)

        p_MConM = p_C - p_M
        p_LConL = p_C - p_L

        v_WConL = v_L + np.cross(omega_vec_L, p_LConL, axis=0)
        v_WConM = v_M + np.cross(omega_vec_M, p_MConM, axis=0)
        v_S_raw = v_WConM - v_WConL
        v_S_N = self.get_N_proj(context, v_S_raw)
        v_S = v_S_raw - v_S_N
        return v_S

    # ========================== OUTPUT FUNCTIONS =========================
    def output_T_hat(self, context, output):
        T_hat = self.calc_T_hat(context)
        output.SetFromVector(T_hat.flatten())

    def output_N_hat(self, context, output):
        N_hat = self.calc_N_hat(context)
        output.SetFromVector(N_hat.flatten())

    def output_theta_L(self, context, output):
        theta_L = self.GetInputPort("pose_L_rotational").Eval(context)[0]
        output.SetFromVector([theta_L])

    def output_d_theta_L(self, context, output):
        d_theta_L = self.GetInputPort("vel_L_rotational").Eval(context)[0]
        output.SetFromVector([d_theta_L])

    def output_p_LN(self, context, output):
        p_L = self.calc_p_L(context)
        p_LN = self.get_N_proj(context, p_L)
        output.SetFromVector([p_LN])

    def output_p_LT(self, context, output):
        p_L = self.calc_p_L(context)
        p_LT = self.get_T_proj(context, p_L)
        output.SetFromVector([p_LT])

    def output_v_LN(self, context, output):
        v_L = self.calc_v_L(context)
        v_LN = self.get_N_proj(context, v_L)
        output.SetFromVector([v_LN])

    def output_v_MN(self, context, output):
        v_M = self.calc_v_M(context)
        v_MN = self.get_N_proj(context, v_M)
        output.SetFromVector([v_MN])

    def output_theta_MX(self, context, output):
        theta_MX = self.GetInputPort("pose_M_rotational").Eval(context)[0]
        output.SetFromVector([theta_MX])

    def output_theta_MY(self, context, output):
        theta_MY = self.GetInputPort("pose_M_rotational").Eval(context)[1]
        output.SetFromVector([theta_MY])

    def output_theta_MZ(self, context, output):
        theta_MZ = self.GetInputPort("pose_M_rotational").Eval(context)[2]
        output.SetFromVector([theta_MZ])

    def output_d_theta_MX(self, context, output):
        d_theta_MX = self.GetInputPort("vel_M_rotational").Eval(context)[0]
        output.SetFromVector([d_theta_MX])

    def output_d_theta_MY(self, context, output):
        d_theta_MY = self.GetInputPort("vel_M_rotational").Eval(context)[1]
        output.SetFromVector([d_theta_MY])

    def output_d_theta_MZ(self, context, output):
        d_theta_MZ = self.GetInputPort("vel_M_rotational").Eval(context)[2]
        output.SetFromVector([d_theta_MZ])

    def output_F_GT(self, context, output):
        F_G = np.array([[0, 0, -self.sys_consts.m_L*constants.g]]).T
        F_GT = self.get_T_proj(context, F_G)
        output.SetFromVector([F_GT])

    def output_F_GN(self, context, output):
        F_G = np.array([[0, 0, -self.sys_consts.m_L*constants.g]]).T
        F_GN = self.get_N_proj(context, F_G)
        output.SetFromVector([F_GN])

    def output_d_X(self, context, output):
        d_X = self.GetInputPort("pose_M_translational").Eval(context)[0]
        output.SetFromVector([d_X])

    def output_d_d_X(self, context, output):
        d_d_X = self.GetInputPort("vel_M_translational").Eval(context)[0]
        output.SetFromVector([d_d_X])

    def output_d_N(self, context, output):
        d = self.calc_d(context)
        d_N = self.get_N_proj(context, d)
        output.SetFromVector([d_N])

    def output_d_T(self, context, output):
        d = self.calc_d(context)
        d_T = self.get_T_proj(context, d)
        output.SetFromVector([d_T])

    def output_p_CN(self, context, output):
        p_C = self.calc_p_C(context)
        p_CN = self.get_N_proj(context, p_C)
        output.SetFromVector([p_CN])

    def output_p_CT(self, context, output):
        p_C = self.calc_p_C(context)
        p_CT = self.get_T_proj(context, p_C)
        output.SetFromVector([p_CT])

    def output_d_d_N(self, context, output):
        # Load scalars
        d_theta_L = self.GetInputPort("vel_L_rotational").Eval(context)[0]
        
        d = self.calc_d(context)
        v_M = self.calc_v_M(context)
        v_L = self.calc_v_L(context)

        v_LN = self.get_N_proj(context, v_L)
        d_T = self.get_T_proj(context, d)
        
        v_MN = self.get_N_proj(context, v_M)
        d_d_N = -d_theta_L*self.sys_consts.w_L/2-v_LN+v_MN-d_theta_L*d_T
        output.SetFromVector([d_d_N])
    
    def output_d_d_T(self, context, output):
        # Load scalars
        d_theta_L = self.GetInputPort("vel_L_rotational").Eval(context)[0]
        
        d = self.calc_d(context)
        v_M = self.calc_v_M(context)
        v_L = self.calc_v_L(context)

        v_LT = self.get_T_proj(context, v_L)
        v_MT = self.get_T_proj(context, v_M)
        d_N = self.get_N_proj(context, d)

        d_d_T = -d_theta_L*self.sys_consts.h_L/2 - d_theta_L*self.sys_consts.r - v_LT + v_MT \
            + d_theta_L*d_N
        output.SetFromVector([d_d_T])

    def output_p_MConM(self, context, output):
        p_C = self.calc_p_C(context)
        p_M = self.calc_p_M(context)
        p_MConM = p_C - p_M
        output.SetFromVector(p_MConM.flatten())

    def output_mu_S(self, context, output):
        v_S = self.calc_v_S(context)
        s_S = np.linalg.norm(v_S)
        mu_S = self.stribeck(1, 1, s_S/self.sys_consts.v_stiction)
        output.SetFromVector([mu_S])

    def output_hats_T(self, context, output):
        v_S = self.calc_v_S(context)
        s_hat = v_S/np.linalg.norm(v_S)
        hats_T = self.get_T_proj(context, s_hat)
        output.SetFromVector([hats_T])

    def output_s_hat_X(self, context, output):
        v_S = self.calc_v_S(context)
        s_hat = v_S/np.linalg.norm(v_S)
        s_hat_X = s_hat[0]
        output.SetFromVector([s_hat_X])

    def output_in_contact(self, context, output):
        d = self.calc_d(context)
        d_N = self.get_N_proj(context, d)
        raw_in_contact = d_N > self.d_N_thresh
        if raw_in_contact:
            if self.t_contact_start is None:
                self.t_contact_start = context.get_time()
        else:
            self.t_contact_start =  None
        in_contact = raw_in_contact and context.get_time() \
            - self.t_contact_start > 2
        output.SetFromVector([in_contact])

    # ========================== OTHER FUNCTIONS ==========================
    def step5(self, x):
        '''Python version of MultibodyPlant::StribeckModel::step5 method'''
        x3 = x * x * x
        return x3 * (10 + x * (6 * x - 15))

    def stribeck(self, us, uk, v):
        '''
        Python version of
        `MultibodyPlant::StribeckModel::ComputeFrictionCoefficient`

        From
        https://github.com/RobotLocomotion/drake/blob/b09e40db4b1c01232b22f7705fb98aa99ef91f87/multibody/plant/images/stiction.py
        '''
        u = uk
        if v < 1:
            u = us * self.step5(v)
        elif (v >= 1) and (v < 3):
            u = us - (us - uk) * self.step5((v - 1) / 2)
        return u
