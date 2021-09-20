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

        self.K_centering = 1
        self.D_centering = 0.1

        self.t_contact_start =  None

        self.init_q = None

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

        if self.init_q is None:
            self.init_q = q

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

        J_plus = np.linalg.pinv(J)
        nullspace_basis = np.eye(self.nq_arm) - np.matmul(J_plus, J)

        # Add a PD controller projected into the nullspace of the Jacobian that keeps us close to the nominal configuration
        joint_centering_torque = np.matmul(nullspace_basis, self.K_centering*(self.init_q - q) + self.D_centering*(-v))

        tau_d = (J.T@F_d).flatten()

        tau_ctrl = tau_d - grav + joint_centering_torque
        output.SetFromVector(tau_ctrl)

        # Debug
        M = self.arm_plant.CalcMassMatrixViaInverseDynamics(self.arm_plant_context)
        C = self.arm_plant.CalcBiasTerm(self.arm_plant_context)
        self.debug["M"].append(M)
        self.debug["C"].append(C)
        self.debug["tau_g"].append(grav)

        self.debug['F_d'].append(F_d)
        self.debug['tau_d'].append(tau_d)
        self.debug['tau_ctrl'].append(tau_ctrl)
        self.debug['J'].append(J_full)
        self.debug['real_q_dot'].append(real_q_dot)
        
        self.debug['raw_in_contact'].append(raw_in_contact)
        self.debug['in_contact'].append(in_contact)
