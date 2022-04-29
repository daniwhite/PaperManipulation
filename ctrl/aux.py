"""
Systems that are not explicit control strategies but provide useful auxiliary
functions.
"""
# Drake imports
import pydrake  # pylint: disable=import-error
from pydrake.all import ContactResults, RigidTransform

import numpy as np
from numpy.random import default_rng

import plant.manipulator as manipulator
from plant.paper import settling_time
from config import hinge_rotation_axis
import sim_exceptions

import time

class JointCenteringCtrl(pydrake.systems.framework.LeafSystem):
    """
    PD controller projected into the nullspace of the Jacobian that keeps us
    close to the nominal configuration
    """

    def __init__(self, K=1, D=0.1):
        pydrake.systems.framework.LeafSystem.__init__(self)
        # System parameters
        self.nq = manipulator.data["nq"]

        # Gains
        self.K = K
        self.D = D

        # Intermediary terms
        self.init_q = None

        self.DeclareVectorInputPort(
            "J",
            pydrake.systems.framework.BasicVector(6*self.nq))
        self.DeclareVectorInputPort(
            "q", pydrake.systems.framework.BasicVector(self.nq))
        self.DeclareVectorInputPort(
            "v", pydrake.systems.framework.BasicVector(self.nq))

        self.DeclareVectorOutputPort(
            "tau_out", pydrake.systems.framework.BasicVector(self.nq),
            self.CalcOutput)

    def CalcOutput(self, context, output):
        # Load inputs
        J = self.GetInputPort("J").Eval(context).reshape(
            (6, self.nq))
        q = np.expand_dims(self.GetInputPort("q").Eval(context), 1)
        v = np.expand_dims(self.GetInputPort("v").Eval(context), 1)

        if self.init_q is None:
            self.init_q = q

        # Calculate torque
        J_plus = np.linalg.pinv(J)
        nullspace_basis = np.eye(self.nq) - np.matmul(J_plus, J)
        joint_centering_torque = np.matmul(
            nullspace_basis,
            self.K*(self.init_q - q) - self.D*v
        )
        
        # Set output
        output.SetFromVector(joint_centering_torque.flatten())


class PreContactCtrl(pydrake.systems.framework.LeafSystem):
    """
    Moves the controller towards the hinge.
    """

    def __init__(self, v_MZd=0.1, K=100):
        pydrake.systems.framework.LeafSystem.__init__(self)
        # System parameters
        self.nq = manipulator.data["nq"]

        # Gains
        self.v_MZd = v_MZd
        self.K = K

        # Intermediary terms
        self.init_q = None

        self.DeclareVectorInputPort(
            "J",
            pydrake.systems.framework.BasicVector(6*self.nq))
        self.DeclareVectorInputPort(
            "v_M", pydrake.systems.framework.BasicVector(3))

        self.DeclareVectorOutputPort(
            "tau_out", pydrake.systems.framework.BasicVector(self.nq),
            self.CalcOutput)

    def CalcOutput(self, context, output):
        # Load inputs
        J = self.GetInputPort("J").Eval(context).reshape(
            (6, self.nq))
        v_M = self.GetInputPort("v_M").Eval(context)

        # Defaults to int w/ no dtype
        v_Md = np.array([0, 0, 0], dtype=np.float64)

        if context.get_time() > settling_time:
            v_Md[-1] = self.v_MZd

        F_C = (v_Md - v_M)*self.K

        F_d = np.zeros((6,1))
        F_d[3:] += np.expand_dims(F_C, 1)

        tau_ctrl = J.T@F_d
        output.SetFromVector(tau_ctrl.flatten()) 


class CtrlSelector(pydrake.systems.framework.LeafSystem):
    def __init__(self):
        pydrake.systems.framework.LeafSystem.__init__(self)
        self.nq = manipulator.data["nq"]
        self.printed_yet = False

        self.DeclareVectorInputPort(
            "in_contact",
            pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "pre_contact_ctrl", pydrake.systems.framework.BasicVector(self.nq))
        self.DeclareVectorInputPort(
            "contact_ctrl", pydrake.systems.framework.BasicVector(self.nq))

        self.DeclareVectorOutputPort(
            "tau_out", pydrake.systems.framework.BasicVector(self.nq),
            self.CalcOutput)

    def CalcOutput(self, context, output):
        pre_contact_tau_ctrl = self.GetInputPort(
            "pre_contact_ctrl").Eval(context)
        contact_tau_ctrl = self.GetInputPort("contact_ctrl").Eval(context)
        in_contact = self.GetInputPort("in_contact").Eval(context)[0]
        use_contact_ctrl = False
        if in_contact:
            if context.get_time() > 1:
                if not self.printed_yet:
                    print("[CtrlSelector] Switching to contact ctrl at {:.2f}".format(context.get_time()))
                self.printed_yet = True
                use_contact_ctrl = True

        tau_ctrl = contact_tau_ctrl if use_contact_ctrl else pre_contact_tau_ctrl
        output.SetFromVector(tau_ctrl)


class HTNtoXYZ(pydrake.systems.framework.LeafSystem):
    """
    Convert a translation vector of expressed in the HTN coordinates to XYZ
    coordinates.
    """
    def __init__(self):
        pydrake.systems.framework.LeafSystem.__init__(self)
        self.nq = manipulator.data["nq"]

        self.DeclareVectorInputPort(
            "HTN",
            pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "T_hat", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "N_hat", pydrake.systems.framework.BasicVector(3))

        self.DeclareVectorOutputPort(
            "XYZ", pydrake.systems.framework.BasicVector(3),
            self.CalcOutput)

    def CalcOutput(self, context, output):
        [H, T, N] = self.GetInputPort("HTN").Eval(context)

        H_hat = np.array([[0, 0, 0]]).T
        H_hat[hinge_rotation_axis] = 1
        T_hat = np.expand_dims(self.GetInputPort("T_hat").Eval(context), 1)
        N_hat = np.expand_dims(self.GetInputPort("N_hat").Eval(context), 1)

        xyz = H*H_hat + T*T_hat + N*N_hat
        output.SetFromVector(xyz.flatten())

class NormalForceSelector(pydrake.systems.framework.LeafSystem):
    def __init__(self, ll_idx, contact_body_idx, paper):
        pydrake.systems.framework.LeafSystem.__init__(self)

        self.ll_idx = ll_idx
        self.contact_body_idx = contact_body_idx
        self.paper_idxs = set(paper.link_idxs)

        self.F_N = 0
        self.contact_force = np.array([0.0,0.0,0.0])

        self.DeclareAbstractInputPort(
            "contact_results",
            pydrake.common.value.AbstractValue.Make(ContactResults()))
        
        self.DeclareVectorOutputPort(
            "F_N", pydrake.systems.framework.BasicVector(1), self.get_F_N)

        self.DeclareVectorOutputPort(
            "contact_force",
            pydrake.systems.framework.BasicVector(3), self.get_contact_force)
    
    def update(self, context):
        contact_results = self.GetInputPort("contact_results").Eval(context)

        F_N = 0
        contact_force = np.array([0.0,0.0,0.0])
        for i in range(contact_results.num_point_pair_contacts()):
            point_pair_contact_info = \
                contact_results.point_pair_contact_info(i)
            body_A_is_contact_body = int(
                point_pair_contact_info.bodyA_index()) == \
                self.contact_body_idx
            body_B_is_contact_body = int(
                point_pair_contact_info.bodyB_index()) == \
                self.contact_body_idx
            body_A_is_last_link = int(
                point_pair_contact_info.bodyA_index()) in self.paper_idxs
            body_B_is_last_link = int(
                point_pair_contact_info.bodyB_index()) in self.paper_idxs
            if (body_A_is_contact_body and body_B_is_last_link) or \
                    (body_B_is_contact_body and body_A_is_last_link):
                contact_force_ = point_pair_contact_info.contact_force()
                nhat = point_pair_contact_info.point_pair().nhat_BA_W
                F_N_ = np.dot(contact_force_, nhat)
                if body_B_is_contact_body:
                    F_N_ *= -1
                    contact_force_ *= -1
                F_N += F_N_
                contact_force += contact_force_
        F_N = max(0,F_N)
        
        self.F_N = F_N
        self.contact_force = contact_force
    
    def get_contact_force(self, context, output):
        self.update(context)
        output.SetFromVector(self.contact_force)

    def get_F_N(self, context, output):
        self.update(context)
        output.SetFromVector([self.F_N])


class AnyContactsCalculator(pydrake.systems.framework.LeafSystem):
    def __init__(self, paper_idxs, contact_body_idx):
        pydrake.systems.framework.LeafSystem.__init__(self)
        self.paper_idxs = set(paper_idxs)
        self.contact_body_idx = contact_body_idx

        self.DeclareAbstractInputPort(
            "contact_results",
            pydrake.common.value.AbstractValue.Make(ContactResults()))
        self.DeclareVectorOutputPort(
            "out", pydrake.systems.framework.BasicVector(1),
            self.CalcOutput)

    def CalcOutput(self, context, output):
        contact_results = self.GetInputPort("contact_results").Eval(context)

        any_links_in_contact = False
        for i in range(contact_results.num_point_pair_contacts()):
            point_pair_contact_info = \
                contact_results.point_pair_contact_info(i)
            # See if we're in contact with any link
            if int(point_pair_contact_info.bodyA_index()) == self.contact_body_idx:
                if int(point_pair_contact_info.bodyB_index()) in self.paper_idxs:
                    any_links_in_contact = True
            if int(point_pair_contact_info.bodyB_index()) == self.contact_body_idx:
                if int(point_pair_contact_info.bodyA_index()) in self.paper_idxs:
                    any_links_in_contact = True
        output.SetFromVector([any_links_in_contact])

class ExitSystem(pydrake.systems.framework.LeafSystem):
    def __init__(self, timeout, exit_when_folded, ll_idx, paper,
            z_thresh_offset, z_lockdown_thresh_offset):
        pydrake.systems.framework.LeafSystem.__init__(self)
        
        # General exit params
        self.exit_when_folded = exit_when_folded
        self.timeout = timeout
        self.start_time = None
        self.last_contact_time = None

        self.start_t__d_theta_L_below_thresh = None
        self.prev_overall_theta = None
        self.start_t_success = None
        self.d_theta_L_thresh = 0.005

        # System params
        self.ll_idx = ll_idx
        self.paper = paper
        self.z_thresh_offset = z_thresh_offset
        self.z_lockdown_thresh_offset = z_lockdown_thresh_offset
        self.overall_thetas = []
        self.overall_theta_times = []
        self.overall_thetas_xs = []
        self.overall_thetas_ys = []

        self.DeclareAbstractInputPort(
            "poses", pydrake.common.value.AbstractValue.Make([RigidTransform(), RigidTransform()]))
        self.DeclareVectorInputPort(
            "theta_L",
            pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "d_theta_L",
            pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "in_contact", pydrake.systems.framework.BasicVector(1))

        self.DeclareVectorOutputPort(
            "alive_signal", pydrake.systems.framework.BasicVector(1),
            self.calc_lockdown_signal)

    def calc_lockdown_signal(self, context, output):
        # Check for exit criteria
        poses = self.GetInputPort("poses").Eval(context)
        theta_L = self.GetInputPort("theta_L").Eval(context)[0]
        d_theta_L = self.GetInputPort("d_theta_L").Eval(context)[0]
        in_contact = self.GetInputPort("in_contact").Eval(context)[0]

        p_LL = poses[self.ll_idx].translation()
        p_FL = poses[self.paper.link_idxs[0]].translation()
        # print(p_FL.shape)

        X_FL_FJ_L = \
            self.paper.joints[0].frame_on_parent().GetFixedPoseInBodyFrame()
        offset = np.array([0.0,0.0,0.0])
        offset[1-hinge_rotation_axis] = -self.paper.w_L/2
        offset[2] = self.paper.h_L*1.5
        p_W_FJ = p_FL.flatten() + offset

        p_FJ_LL = p_LL.flatten() - p_W_FJ.flatten()
        atan_x = -p_FJ_LL[1-hinge_rotation_axis]
        atan_y = p_FJ_LL[2]
        self.overall_thetas_xs.append(atan_x)
        self.overall_thetas_ys.append(atan_y)
        overall_theta = np.arctan2(atan_y,atan_x)
        # print(overall_theta)
        if not(self.prev_overall_theta is None):
            if np.abs(self.prev_overall_theta - overall_theta) > 0.99*2*np.pi:
                overall_theta -= 2*np.pi*np.sign(overall_theta)
        self.prev_overall_theta = overall_theta

        # TODO: is sign modular?
        horizontal_thresh = p_FL[1-hinge_rotation_axis]
        z_thresh = p_FL[-1] + self.z_thresh_offset
        z_lockdown_thresh = z_thresh + self.z_lockdown_thresh_offset

        if in_contact:
            self.last_contact_time = context.get_time()

        outval = 0
        hor_thresh_reached = p_LL[1-hinge_rotation_axis] > horizontal_thresh
        # Calculate lock down
        if in_contact and hor_thresh_reached and \
                (p_LL[-1] < z_lockdown_thresh) and \
                (theta_L > np.pi*0.9):
            outval = 0

        # Calculate exit terms
        if self.exit_when_folded:
            self.overall_thetas.append(overall_theta)
            self.overall_theta_times.append(context.get_time())
            if overall_theta > np.pi*0.99:
                if self.start_t_success is None:
                    self.start_t_success = context.get_time()
                if (context.get_time() - self.start_t_success > 0.5):
                    raise sim_exceptions.SimTaskComplete
            else:
                self.start_t_success = None

            # Stalling criteria
            if abs(d_theta_L) < self.d_theta_L_thresh:
                if self.start_t__d_theta_L_below_thresh is None:
                    self.start_t__d_theta_L_below_thresh = context.get_time()
                if (context.get_time() - \
                        self.start_t__d_theta_L_below_thresh > 1):
                    raise sim_exceptions.SimStalled
            else:
                self.start_t__d_theta_L_below_thresh = None

        # Timeout criteria
        if self.timeout is not None:
            if (time.time() - self.start_time) > self.timeout:
                raise sim_exceptions.SimTimedOut

        output.SetFromVector([outval])


    def set_start_time(self):
        self.start_time = time.time()


class NoiseGenerator(pydrake.systems.framework.LeafSystem):
    def __init__(self, size, scale):
        pydrake.systems.framework.LeafSystem.__init__(self)
        self.size = size
        self.scale = scale
        self.rng = default_rng()

        self.DeclareVectorOutputPort(
            "out",
            pydrake.systems.framework.BasicVector(size),
            self.CalcOutput)

    def CalcOutput(self, context, output):
        output.SetFromVector(self.rng.standard_normal(self.size)*self.scale)
