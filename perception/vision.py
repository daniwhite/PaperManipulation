# Drake imports
import pydrake
from pydrake.all import RigidTransform, SpatialVelocity, RollPitchYaw, RotationMatrix
import numpy as np
import plant.paper as paper
import constants
import plant.manipulator as manipulator
from config import hinge_rotation_axis

from collections import defaultdict

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

    def __init__(self, sys_consts, X_LJ_L): # TODO: move to sys consts?
        pydrake.systems.framework.LeafSystem.__init__(self)
        # Physical parameters
        self.sys_consts = sys_consts

        self.debug = defaultdict(list)

        # RT from link CoM to joint. Come from 
        # `.frame_on_child().GetFixedPoseInBodyFrame()` and is used for
        # dd_d_theta_L dynamics
        self.X_LJ_L = X_LJ_L

        self.d_N_thresh = 5e-4
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
            "omega_MX", pydrake.systems.framework.BasicVector(1),
            self.output_omega_MX)
        self.DeclareVectorOutputPort(
            "omega_MY", pydrake.systems.framework.BasicVector(1),
            self.output_omega_MY)
        self.DeclareVectorOutputPort(
            "omega_MZ", pydrake.systems.framework.BasicVector(1),
            self.output_omega_MZ)
        self.DeclareVectorOutputPort(
            "F_GT", pydrake.systems.framework.BasicVector(1),
            self.output_F_GT)
        self.DeclareVectorOutputPort(
            "F_GN", pydrake.systems.framework.BasicVector(1),
            self.output_F_GN)
        self.DeclareVectorOutputPort(
            "d_H", pydrake.systems.framework.BasicVector(1),
            self.output_d_H)
        self.DeclareVectorOutputPort(
            "d_d_H", pydrake.systems.framework.BasicVector(1),
            self.output_d_d_H)
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
            "s_hat_H", pydrake.systems.framework.BasicVector(1),
            self.output_s_hat_H)
        self.DeclareVectorOutputPort(
            "in_contact", pydrake.systems.framework.BasicVector(1),
            self.output_in_contact)
        self.DeclareVectorOutputPort(
            "p_JL", pydrake.systems.framework.BasicVector(3),
            self.output_p_JL)
        self.DeclareVectorOutputPort(
            "p_JC", pydrake.systems.framework.BasicVector(3),
            self.output_p_JC)
        self.DeclareVectorOutputPort(
            "I_LJ", pydrake.systems.framework.BasicVector(1),
            self.output_I_LJ)
        self.DeclareVectorOutputPort(
            "gravity_torque_about_joint",
            pydrake.systems.framework.BasicVector(1),
            self.output_gravity_torque_about_joint
        )

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

        # TODO: Maybe move this to vision (rather than processor)
        self.debug['p_L_t'].append(context.get_time())
        self.debug['p_L'].append(p_L)
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

    def calc_theta_L(self, context):
        rot_vec_L = self.calc_rot_vec_L(context).flatten() # TODO: is this flatten needed?
        theta_L = rot_vec_L[hinge_rotation_axis]
        theta_Z = rot_vec_L[2]
        # Fix issue in RPY singularity
        if theta_Z > np.pi/2:
            theta_L = theta_L*-1 + np.pi
        return theta_L

    def calc_d_theta_L(self, context):
        omega_vec_L = self.calc_omega_vec_L(context).flatten() # TODO: is this flatten needed?
        d_theta_L = omega_vec_L[hinge_rotation_axis]
        return d_theta_L

    def calc_T_hat(self, context):
        rpy = [0,0,0]
        rpy[hinge_rotation_axis] = self.calc_theta_L(context) - np.pi/2
        R = RotationMatrix(RollPitchYaw(rpy)).matrix()
        z_hat = np.array([[0, 0, 1]]).T
        return np.matmul(R, z_hat)

    def calc_N_hat(self, context):
        rpy = [0,0,0]
        rpy[hinge_rotation_axis] = self.calc_theta_L(context)
        R = RotationMatrix(RollPitchYaw(rpy)).matrix()
        z_hat = np.array([[0, 0, 1]]).T
        return np.matmul(R, z_hat)

    def calc_p_C(self, context):
        p_M = self.calc_p_M(context)
        N_hat = self.calc_N_hat(context)
        p_C = p_M + N_hat*self.sys_consts.r

        # TODO: Should this be somewhere else?
        self.debug['p_C_t'].append(context.get_time())
        self.debug['p_C'].append(p_C)
        return p_C

    def calc_p_LE(self, context):
        N_hat = self.calc_N_hat(context)
        T_hat = self.calc_T_hat(context)
        p_L = self.calc_p_L(context)
        p_LE = p_L + (self.sys_consts.w_L/2)*T_hat-(self.sys_consts.h_L/2)*N_hat

        self.debug['p_LE_t'].append(context.get_time())
        self.debug['p_LE'].append(p_LE)
        return p_LE

    def calc_d(self, context):
        p_C = self.calc_p_C(context)
        p_LE = self.calc_p_LE(context)
        d = p_C - p_LE

        self.debug['d_vec_t'].append(context.get_time())
        self.debug['d_vec'].append(d)
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

    def calc_X_WJ(self, context):
        p_L = self.calc_p_L(context)
        rot_vec_L = self.calc_rot_vec_L(context)
        X_WL = RigidTransform(
            p=p_L,
            rpy=RollPitchYaw(rot_vec_L)
        )
        X_WJ = X_WL.multiply(self.X_LJ_L)
        return X_WJ
    
    def calc_p_JL(self, context):
        X_WJ = self.calc_X_WJ(context)
        p_L = self.calc_p_L(context)
        return p_L.flatten() - X_WJ.translation()
    
    def calc_p_JC(self, context):
        X_WJ = self.calc_X_WJ(context)
        p_C = self.calc_p_C(context)
        return p_C.flatten() - X_WJ.translation()

    # ========================== OUTPUT FUNCTIONS =========================
    def output_T_hat(self, context, output):
        T_hat = self.calc_T_hat(context)

        self.debug['T_hat'].append(T_hat)
        self.debug['T_hat_t'].append(context.get_time())
        output.SetFromVector(T_hat.flatten())

    def output_N_hat(self, context, output):
        N_hat = self.calc_N_hat(context)

        self.debug['N_hat'].append(N_hat)
        self.debug['N_hat_t'].append(context.get_time())
        output.SetFromVector(N_hat.flatten())

    def output_theta_L(self, context, output):
        theta_L = self.calc_theta_L(context)

        self.debug['theta_L'].append(theta_L)
        self.debug['theta_L_t'].append(context.get_time())
        output.SetFromVector([theta_L])

    def output_d_theta_L(self, context, output):
        d_theta_L = self.calc_d_theta_L(context)

        self.debug['d_theta_L'].append(d_theta_L)
        self.debug['d_theta_L_t'].append(context.get_time())
        output.SetFromVector([d_theta_L])

    def output_p_LN(self, context, output):
        p_L = self.calc_p_L(context)
        p_LN = self.get_N_proj(context, p_L)

        self.debug['p_LN'].append(p_LN)
        self.debug['p_LN_t'].append(context.get_time())
        output.SetFromVector([p_LN])

    def output_p_LT(self, context, output):
        p_L = self.calc_p_L(context)
        p_LT = self.get_T_proj(context, p_L)

        self.debug['p_LT'].append(p_LT)
        self.debug['p_LT_t'].append(context.get_time())
        output.SetFromVector([p_LT])

    def output_v_LN(self, context, output):
        v_L = self.calc_v_L(context)
        v_LN = self.get_N_proj(context, v_L)

        self.debug['v_LN'].append(v_LN)
        self.debug['v_LN_t'].append(context.get_time())
        output.SetFromVector([v_LN])

    def output_v_MN(self, context, output):
        v_M = self.calc_v_M(context)
        v_MN = self.get_N_proj(context, v_M)

        self.debug['v_MN'].append(v_MN)
        self.debug['v_MN_t'].append(context.get_time())
        output.SetFromVector([v_MN])

    def output_theta_MX(self, context, output):
        theta_MX = self.GetInputPort("pose_M_rotational").Eval(context)[0]

        self.debug['theta_MX'].append(theta_MX)
        self.debug['theta_MX_t'].append(context.get_time())
        output.SetFromVector([theta_MX])

    def output_theta_MY(self, context, output):
        theta_MY = self.GetInputPort("pose_M_rotational").Eval(context)[1]

        self.debug['theta_MY'].append(theta_MY)
        self.debug['theta_MY_t'].append(context.get_time())
        output.SetFromVector([theta_MY])

    def output_theta_MZ(self, context, output):
        theta_MZ = self.GetInputPort("pose_M_rotational").Eval(context)[2]

        self.debug['theta_MZ'].append(theta_MZ)
        self.debug['theta_MZ_t'].append(context.get_time())
        output.SetFromVector([theta_MZ])

    def output_omega_MX(self, context, output):
        omega_MX = self.GetInputPort("vel_M_rotational").Eval(context)[0]

        self.debug['omega_MX'].append(omega_MX)
        self.debug['omega_MX_t'].append(context.get_time())
        output.SetFromVector([omega_MX])

    def output_omega_MY(self, context, output):
        omega_MY = self.GetInputPort("vel_M_rotational").Eval(context)[1]

        self.debug['omega_MY'].append(omega_MY)
        self.debug['omega_MY_t'].append(context.get_time())
        output.SetFromVector([omega_MY])

    def output_omega_MZ(self, context, output):
        omega_MZ = self.GetInputPort("vel_M_rotational").Eval(context)[2]

        self.debug['omega_MZ'].append(omega_MZ)
        self.debug['omega_MZ_t'].append(context.get_time())
        output.SetFromVector([omega_MZ])

    def output_F_GT(self, context, output):
        F_G = np.array([[0, 0, -self.sys_consts.m_L*constants.g]]).T
        F_GT = self.get_T_proj(context, F_G)

        self.debug['F_GT'].append(F_GT)
        self.debug['F_GT_t'].append(context.get_time())
        output.SetFromVector([F_GT])

    def output_F_GN(self, context, output):
        F_G = np.array([[0, 0, -self.sys_consts.m_L*constants.g]]).T
        F_GN = self.get_N_proj(context, F_G)

        self.debug['F_GN'].append(F_GN)
        self.debug['F_GN_t'].append(context.get_time())
        output.SetFromVector([F_GN])

    def output_d_H(self, context, output):
        d = self.calc_d(context)
        d_H = d[hinge_rotation_axis]

        self.debug['d_H'].append(d_H)
        self.debug['d_H_t'].append(context.get_time())
        output.SetFromVector([d_H])

    def output_d_d_H(self, context, output):
        d_d_H = self.GetInputPort(
            "vel_M_translational").Eval(context)[hinge_rotation_axis]

        self.debug['d_d_H'].append(d_d_H)
        self.debug['d_d_H_t'].append(context.get_time())
        output.SetFromVector([d_d_H])

    def output_d_N(self, context, output):
        d = self.calc_d(context)
        d_N = self.get_N_proj(context, d)

        self.debug['d_N'].append(d_N)
        self.debug['d_N_t'].append(context.get_time())
        output.SetFromVector([d_N])

    def output_d_T(self, context, output):
        d = self.calc_d(context)
        d_T = self.get_T_proj(context, d)

        self.debug['d_T'].append(d_T)
        self.debug['d_T_t'].append(context.get_time())
        output.SetFromVector([d_T])

    def output_p_CN(self, context, output):
        p_C = self.calc_p_C(context)
        p_CN = self.get_N_proj(context, p_C)

        self.debug['p_CN'].append(p_CN)
        self.debug['p_CN_t'].append(context.get_time())
        output.SetFromVector([p_CN])

    def output_p_CT(self, context, output):
        p_C = self.calc_p_C(context)
        p_CT = self.get_T_proj(context, p_C)

        self.debug['p_CT'].append(p_CT)
        self.debug['p_CT_t'].append(context.get_time())
        output.SetFromVector([p_CT])

    def output_d_d_N(self, context, output):
        # Load scalars
        d_theta_L = self.calc_d_theta_L(context)
        
        d = self.calc_d(context)
        v_M = self.calc_v_M(context)
        v_L = self.calc_v_L(context)

        v_LN = self.get_N_proj(context, v_L)
        d_T = self.get_T_proj(context, d)
        
        v_MN = self.get_N_proj(context, v_M)
        d_d_N = -d_theta_L*self.sys_consts.w_L/2-v_LN+v_MN-d_theta_L*d_T

        self.debug['d_d_N'].append(d_d_N)
        self.debug['d_d_N_t'].append(context.get_time())
        output.SetFromVector([d_d_N])
    
    def output_d_d_T(self, context, output):
        # Load scalars
        d_theta_L = self.calc_d_theta_L(context)
        
        d = self.calc_d(context)
        v_M = self.calc_v_M(context)
        v_L = self.calc_v_L(context)

        v_LT = self.get_T_proj(context, v_L)
        v_MT = self.get_T_proj(context, v_M)
        d_N = self.get_N_proj(context, d)

        d_d_T = -d_theta_L*self.sys_consts.h_L/2 - d_theta_L*self.sys_consts.r - v_LT + v_MT \
            + d_theta_L*d_N

        self.debug['d_d_T'].append(d_d_T)
        self.debug['d_d_T_t'].append(context.get_time())
        output.SetFromVector([d_d_T])

    def output_p_MConM(self, context, output):
        p_C = self.calc_p_C(context)
        p_M = self.calc_p_M(context)
        p_MConM = p_C - p_M

        self.debug['p_MConM'].append(p_MConM)
        self.debug['p_MConM_t'].append(context.get_time())
        output.SetFromVector(p_MConM.flatten())

    def output_mu_S(self, context, output):
        v_S = self.calc_v_S(context)
        s_S = np.linalg.norm(v_S)
        mu_S = self.stribeck(1, 1, s_S/self.sys_consts.v_stiction)

        self.debug['mu_S'].append(mu_S)
        self.debug['mu_S_t'].append(context.get_time())
        output.SetFromVector([mu_S])

    def output_hats_T(self, context, output):
        v_S = self.calc_v_S(context)
        s_hat = v_S/np.linalg.norm(v_S)
        hats_T = self.get_T_proj(context, s_hat)

        self.debug['s_hat_T'].append(hats_T)
        self.debug['s_hat_T_t'].append(context.get_time())
        output.SetFromVector([hats_T])

    def output_s_hat_H(self, context, output):
        v_S = self.calc_v_S(context)
        s_hat = v_S/np.linalg.norm(v_S)
        s_hat_H = s_hat[hinge_rotation_axis]
    
        self.debug['s_hat_H'].append(s_hat_H)
        self.debug['s_hat_H_t'].append(context.get_time())
        output.SetFromVector([s_hat_H])

    def output_in_contact(self, context, output):
        d = self.calc_d(context)
        d_N = self.get_N_proj(context, d)
        raw_in_contact = np.abs(d_N) < self.d_N_thresh
        if raw_in_contact:
            if self.t_contact_start is None:
                self.t_contact_start = context.get_time()
        else:
            self.t_contact_start =  None
        in_contact = raw_in_contact and context.get_time() \
            - self.t_contact_start > 0.5

        self.debug['in_contact'].append(in_contact)
        self.debug['in_contact_t'].append(context.get_time())
        output.SetFromVector([in_contact])
    
    def output_p_JL(self, context, output):
        p_JL = self.calc_p_JL(context)

        self.debug['p_JL'].append(p_JL)
        self.debug['p_JL_t'].append(context.get_time())
        output.SetFromVector(p_JL)
    
    def output_p_JC(self, context, output):
        p_JC = self.calc_p_JC(context)

        self.debug['p_JC'].append(p_JC)
        self.debug['p_JC_t'].append(context.get_time())
        output.SetFromVector(p_JC)

    def output_I_LJ(self, context, output):
        p_JL = self.calc_p_JL(context)
        d = np.linalg.norm(p_JL)
        I_LJ = self.sys_consts.I_L + self.sys_consts.m_L*d**2

        self.debug['I_LJ'].append(I_LJ)
        self.debug['I_LJ_t'].append(context.get_time())
        output.SetFromVector([I_LJ])

    def output_gravity_torque_about_joint(self, context, output):
        F_G = np.array([[0, 0, -self.sys_consts.m_L*constants.g]]).T
        p_JL = self.calc_p_JL(context)
        gravity_torque_about_joint = np.cross(
            p_JL, F_G, axis=0).flatten()[hinge_rotation_axis]

        self.debug['gravity_torque_about_joint'].append(gravity_torque_about_joint)
        self.debug['gravity_torque_about_joint_t'].append(context.get_time())
        output.SetFromVector([gravity_torque_about_joint])

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
