"""Functions for creating and controlling the finger manipulator."""

# Standard imports
import constants
import pedestal
from pydrake.multibody.tree import SpatialInertia, UnitInertia, JacobianWrtVariable
from pydrake.all import BasicVector, MultibodyPlant
import numpy as np
from collections import defaultdict

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
        RigidTransform(RotationMatrix(), [0, 0.8, 0])
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

    def __init__(self, arm_acc_log):
        pydrake.systems.framework.LeafSystem.__init__(self)

        self.arm_acc_log = arm_acc_log
        self.arm_plant = MultibodyPlant(constants.DT)
        AddArm(self.arm_plant)
        self.arm_plant.Finalize()
        self.arm_plant_context = self.arm_plant.CreateDefaultContext()

        self.nq_arm = self.arm_plant.get_actuation_input_port().size()

        self.DeclareVectorInputPort("q", BasicVector(self.nq_arm*2))
        # self.DeclareAbstractInputPort(
        #     "poses", pydrake.common.value.AbstractValue.Make([RigidTransform(), RigidTransform()]))
        # self.DeclareAbstractInputPort(
        #     "vels", pydrake.common.value.AbstractValue.Make([SpatialVelocity(), SpatialVelocity()]))
        # self.DeclareAbstractInputPort(
        #     "contact_results",
        #     pydrake.common.value.AbstractValue.Make(ContactResults()))
        self.DeclareVectorOutputPort(
            "arm_actuation", pydrake.systems.framework.BasicVector(
                self.nq_arm),
            self.CalcOutput)

        self.debug = defaultdict(list)

    # , poses, vels, contact_point, slip_speed, pen_depth, N_hat):
    def GetForces(self):
        """
        Should be overloaded to return [Fx, Fy, Fz] to move manipulator (not including gravity
        compensation.)
        """
        # raise NotImplementedError()
        return np.array([[0, 0, 0.1]]).T

    def get_force_exerted(self, q, v, tau_g, J):
        # TODO: can I remove some of these arguments?

        # Manipulator equations:
        # M(q) v_dot + C(q, v) v = tau_g + tau_ctrl - tau_exerted
        # Moving terms around:
        # tau_exerted = tau_g + tau_ctrl - M(q) v_dot - C(q, v) v
        
        tau_ctrl = self.last_tau_ctrl

        v_dot = np.array(self.arm_acc_log[-1])

        M = self.arm_plant.CalcMassMatrixViaInverseDynamics(self.arm_plant_context)
        C = self.arm_plant.CalcBiasTerm(self.arm_plant_context)

        tau_exerted = tau_g + tau_ctrl - M@v_dot - C@v
        force_exerted = np.expand_dims(np.linalg.lstsq(J.T, tau_exerted)[0], 1)
        return force_exerted

    def CalcOutput(self, context, output):
        self.debug['times'].append(context.get_time())
        # This input put is already restricted to the arm, but it includes both q and v
        q = self.get_input_port().Eval(context)[:self.nq_arm]
        v = self.get_input_port().Eval(context)[self.nq_arm:]
        self.arm_plant.SetPositions(self.arm_plant_context, q)
        self.arm_plant.SetVelocities(self.arm_plant_context, v)

        # Get gravity 
        grav = self.arm_plant.CalcGravityGeneralizedForces(
            self.arm_plant_context)

        # Get desired forces
        forces = self.GetForces()

        # Convert forces to joint torques
        finger_body = self.arm_plant.GetBodyByName(FINGER_NAME)
        J = self.arm_plant.CalcJacobianTranslationalVelocity(
            self.arm_plant_context,
            JacobianWrtVariable.kQDot,
            finger_body.body_frame(),
            [0, 0, 0],
            self.arm_plant.world_frame(),
            self.arm_plant.world_frame())

        tau_ff = J.T@forces
        tau_ff = tau_ff.flatten()

        tau_g = grav
        force_exerted = self.get_force_exerted(q, v, tau_g, J)
        f_error = forces - force_exerted
        Kp = 10
        tau_fb = (J.T@(Kp*f_error))
        tau_fb = tau_fb.flatten()
        
        tau_ctrl = tau_fb + tau_ff - grav
        output.SetFromVector(tau_ctrl)
        self.last_tau_ctrl = tau_ctrl
