"""Functions for creating and controlling the finger manipulator."""

# Standard imports
import constants
import pedestal
from pydrake.multibody.tree import SpatialInertia, UnitInertia, JacobianWrtVariable
from pydrake.all import BasicVector, MultibodyPlant, ContactResults
import numpy as np
from collections import defaultdict
from common import get_contact_point_from_results

# Drake imports
import pydrake
from pydrake.all import FindResourceOrThrow, RigidTransform, RotationMatrix

FINGER_NAME = "panda_leftfinger"


def AddArm(plant, scene_graph=None):
    """
    Creates the panda arm.
    """
    parser = pydrake.multibody.parsing.Parser(plant, scene_graph)
    arm_instance = parser.AddModelFromFile("panda_arm_hand.urdf")

    # Weld pedestal to world
    jnt = plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("panda_link0", arm_instance),
        RigidTransform(RotationMatrix(), [0, 0.85, 0])
    )
    # Weld fingers (offset matches original urdf)
    plant.WeldFrames(
        plant.GetFrameByName("panda_hand", arm_instance),
        plant.GetFrameByName("panda_leftfinger", arm_instance),
        RigidTransform(RotationMatrix(), [0, 0, 0.0584])
    )
    plant.WeldFrames(
        plant.GetFrameByName("panda_hand", arm_instance),
        plant.GetFrameByName("panda_rightfinger", arm_instance),
        RigidTransform(RotationMatrix(), [0, 0, 0.0584])
    )

    return arm_instance


class ArmForceController(pydrake.systems.framework.LeafSystem):
    """Base class for implementing a controller at the finger."""

    def __init__(self, arm_acc_log, ll_idx, finger_idx):
        pydrake.systems.framework.LeafSystem.__init__(self)
        self.ll_idx = ll_idx
        self.finger_idx = finger_idx

        self.arm_acc_log = arm_acc_log
        self.arm_plant = MultibodyPlant(constants.DT)
        AddArm(self.arm_plant)
        self.arm_plant.Finalize()
        self.arm_plant_context = self.arm_plant.CreateDefaultContext()

        self.nq_arm = self.arm_plant.get_actuation_input_port().size()

        self.last_tau_ctrl = np.zeros(self.nq_arm)
        self.last_tau_error = 0
        self.tau_error_int = 0

        self.t_contact_start =  None
        self.max_tau_fb = np.ones(7)
        self.max_tau_fb[0] = 0.1
        self.max_tau_fb[3] = 10

        # Input ports
        self.DeclareVectorInputPort("q", BasicVector(self.nq_arm*2))
        self.DeclareAbstractInputPort(
            "contact_results",
            pydrake.common.value.AbstractValue.Make(ContactResults()))
        self.DeclareVectorInputPort("F", BasicVector(3))

        # Output ports
        self.DeclareVectorOutputPort(
            "arm_actuation", pydrake.systems.framework.BasicVector(
                self.nq_arm),
            self.CalcOutput)

        self.debug = defaultdict(list)

    def get_tau_measured(self, q, v, tau_g, J):
        """
        Get the torque that corresponds to me exerting the desired force
        """
        # TODO: can I remove some of these arguments?

        # Manipulator equations:
        # M(q) v_dot + C(q, v) v = tau_g + tau_ctrl - J^T F_measured
        # Moving terms around:
        # J^T F_measured = tau_g + tau_ctrl - M(q) v_dot - C(q, v) v
        # J^T F_measured = tau_measured
        
        tau_ctrl = self.last_tau_ctrl

        v_dot = np.array(self.arm_acc_log[-1])

        M = self.arm_plant.CalcMassMatrixViaInverseDynamics(self.arm_plant_context)
        C = self.arm_plant.CalcBiasTerm(self.arm_plant_context)
        self.debug["M"].append(M)
        self.debug["C"].append(C)
        self.debug["tau_g"].append(tau_g)

        tau_measured = tau_g + tau_ctrl- C@v  - M@v_dot 
        return tau_measured

    def CalcOutput(self, context, output):
        self.debug['times'].append(context.get_time())
        contact_results = self.get_input_port(1).Eval(context)
        contact_point = get_contact_point_from_results(contact_results, self.ll_idx, self.finger_idx)
        raw_in_contact = not (contact_point is None)
        if raw_in_contact:
            if self.t_contact_start is None:
                self.t_contact_start = self.debug['times'][-1]
        else:
            self.t_contact_start =  None
        
        in_contact = raw_in_contact and self.debug['times'][-1] - self.t_contact_start > 0.002

        
        # This input put is already restricted to the arm, but it includes both q and v
        state = self.get_input_port(0).Eval(context)
        q = state[:self.nq_arm]
        v = state[self.nq_arm:]
        self.arm_plant.SetPositions(self.arm_plant_context, q)
        self.arm_plant.SetVelocities(self.arm_plant_context, v)
        real_q_dot = self.arm_plant.MapVelocityToQDot(self.arm_plant_context, v)

        # Get gravity 
        grav = self.arm_plant.CalcGravityGeneralizedForces(
            self.arm_plant_context)

        # Get desired forces
        F_d = self.get_input_port(2).Eval(context)
        if not in_contact:
            F_d = np.array([[0, 0, 0.1]]).T


        # Convert forces to joint torques
        finger_body = self.arm_plant.GetBodyByName(FINGER_NAME)
        J_full = self.arm_plant.CalcJacobianSpatialVelocity(
            self.arm_plant_context,
            JacobianWrtVariable.kQDot,
            finger_body.body_frame(),
            [0, 0, 0],
            self.arm_plant.world_frame(),
            self.arm_plant.world_frame())
        J = J_full[3:,:]

        tau_d = J.T@F_d
        tau_d = tau_d.flatten()

        tau_g = grav
        tau_measured = self.get_tau_measured(q, v, tau_g, J)
        assert tau_measured.size == self.nq_arm, "Size is {} instead of {}".format(
                tau_measured.size, self.nq_arm)
        kernel_size = 50
        filtered_tau_measured = np.zeros(7)
        
        tau_measured_log = self.debug['tau_measured']
        if len(tau_measured_log) > 0:
            for i in range(7):
                if len(tau_measured_log) < kernel_size:
                    tau_slice = np.array(tau_measured_log)[:,i]
                else:
                    tau_slice = np.array(tau_measured_log[-kernel_size:])[:,i]
                filtered_tau_measured[i] = np.mean(tau_slice)
        tau_error = tau_d - filtered_tau_measured#tau_measured

        dt = 0
        if len(self.debug['times']) > 2:
            dt = self.debug['times'][-1] - self.debug['times'][-2]

        if dt > 0:
            d_tau_error = (tau_error - self.last_tau_error)/dt
        else:
            d_tau_error = 0
        self.last_tau_error = tau_error

        self.tau_error_int += dt * tau_error

        # K_Ps = np.array([5, 1, 0.99, 0.99, 0.99, 0.99, 0.99]) #0.99])
        # Gains with 20 filter: K_Ps = np.array([20, 20, 5, 10, 5, 5, 5])
        K_Ps = np.array([5, 0.1, 5, 0.1, 5, 0.1, 0.1])
        K_Ds = np.array([0, 0, 0, 0, 0, 0, 0])
        K_Is = np.array([0, 0, 0, 0, 0, 0, 0])
        tau_fb = K_Ps*tau_error + K_Ds*d_tau_error + K_Is*self.tau_error_int
        tau_fb =  tau_fb.flatten()
        if not in_contact:
            tau_fb *= 0
            self.last_tau_error = 0
            self.tau_error_int = 0
        
        # tau_fb_clipped = np.min([tau_fb, self.max_tau_fb], axis=0)
        # tau_fb_clipped = np.max([tau_fb, -self.max_tau_fb], axis=0)
        tau_fb_clipped = tau_fb
        
        # tau_ctrl = tau_fb_clipped + tau_d - grav
        tau_ctrl = tau_d - grav
        output.SetFromVector(tau_ctrl)
        self.last_tau_ctrl = tau_ctrl

        # Debug
        self.debug['F_d'].append(F_d)
        self.debug['tau_d'].append(tau_d)
        self.debug['tau_ctrl'].append(tau_ctrl)
        self.debug['tau_measured'].append(tau_measured)
        self.debug['filtered_tau_measured'].append(filtered_tau_measured)
        self.debug['tau_fb'].append(tau_fb)
        self.debug['tau_fb_clipped'].append(tau_fb_clipped)
        self.debug['J'].append(J_full)
        self.debug['real_q_dot'].append(real_q_dot)
        
        self.debug['raw_in_contact'].append(raw_in_contact)
        self.debug['in_contact'].append(in_contact)
