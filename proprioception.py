import constants
import manipulator

import pydrake
from pydrake.all import (
    MultibodyPlant, JacobianWrtVariable, BasicVector
)

class ProprioceptionSystem(pydrake.systems.framework.LeafSystem):
    """
    Calculates J, M, Cv, etc. based on arm state.
    """

    def __init__(self):
        pydrake.systems.framework.LeafSystem.__init__(self)
        # Physical parameters
        self.manipulator_plant = MultibodyPlant(constants.DT)
        manipulator.data["add_plant_function"](self.manipulator_plant)
        self.manipulator_plant.Finalize()
        self.manipulator_plant_context = \
            self.manipulator_plant.CreateDefaultContext()
        self.nq_manipulator = \
            self.manipulator_plant.get_actuation_input_port().size()

        self.DeclareVectorInputPort("state",
            BasicVector(self.nq_manipulator*2))

        self.DeclareVectorOutputPort(
            "q", pydrake.systems.framework.BasicVector(self.nq_manipulator),
            self.output_q)
        self.DeclareVectorOutputPort(
            "v", pydrake.systems.framework.BasicVector(self.nq_manipulator),
            self.output_v)
        self.DeclareVectorOutputPort(
            "tau_g",
            pydrake.systems.framework.BasicVector(self.nq_manipulator),
            self.output_tau_g)
        self.DeclareVectorOutputPort(
            "M",
            pydrake.systems.framework.BasicVector(
                self.nq_manipulator*self.nq_manipulator),
            self.output_M)
        self.DeclareVectorOutputPort(
            "Cv",
            pydrake.systems.framework.BasicVector(
                self.nq_manipulator),
            self.output_Cv)
        self.DeclareVectorOutputPort(
            "J",
            pydrake.systems.framework.BasicVector(
                6*self.nq_manipulator),
            self.output_J)
        self.DeclareVectorOutputPort(
            "J_translational",
            pydrake.systems.framework.BasicVector(
                3*self.nq_manipulator),
            self.output_J_translational)
        self.DeclareVectorOutputPort(
            "J_rotational",
            pydrake.systems.framework.BasicVector(
                3*self.nq_manipulator),
            self.output_J_rotational)
        self.DeclareVectorOutputPort(
            "Jdot_qdot",
            pydrake.systems.framework.BasicVector(6),
            self.output_Jdot_qdot)

    # =========================== CALC FUNCTIONS ==========================
    def calc_q(self, context):
        state = self.GetInputPort("state").Eval(context)
        q = state[:self.nq_manipulator]
        return q

    def calc_v(self, context):
        state = self.GetInputPort("state").Eval(context)
        v = state[self.nq_manipulator:]
        return v

    def calc_J(self, context):
        self.update_plant(context)
        contact_body = self.manipulator_plant.GetBodyByName(
            manipulator.data["contact_body_name"])
        J = self.manipulator_plant.CalcJacobianSpatialVelocity(
            self.manipulator_plant_context,
            JacobianWrtVariable.kV,
            contact_body.body_frame(),
            [0, 0, 0],
            self.manipulator_plant.world_frame(),
            self.manipulator_plant.world_frame())
        return J

    # ========================== OUTPUT FUNCTIONS =========================
    def output_q(self, context, output):
        q = self.calc_q(context)
        output.SetFromVector(q.flatten())

    def output_v(self, context, output):
        v = self.calc_v(context)
        output.SetFromVector(v.flatten())

    def output_tau_g(self, context, output):
        self.update_plant(context)
        tau_g = self.manipulator_plant.CalcGravityGeneralizedForces(
            self.manipulator_plant_context)
        output.SetFromVector(tau_g.flatten())

    def output_M(self, context, output):
        self.update_plant(context)
        M = self.manipulator_plant.CalcMassMatrixViaInverseDynamics(self.manipulator_plant_context)
        output.SetFromVector(M.flatten())

    def output_Cv(self, context, output):
        self.update_plant(context)
        Cv = self.manipulator_plant.CalcBiasTerm(self.manipulator_plant_context)
        output.SetFromVector(Cv.flatten())

    def output_J(self, context, output):
        J = self.calc_J(context)
        output.SetFromVector(J.flatten())

    def output_J_translational(self, context, output):
        J = self.calc_J(context)
        J_translational = J[3:,:]
        output.SetFromVector(J_translational.flatten())

    def output_J_rotational(self, context, output):
        J = self.calc_J(context)
        J_rotational = J[:3,:]
        output.SetFromVector(J_rotational.flatten())

    def output_Jdot_qdot(self, context, output):
        self.update_plant(context)
        contact_body = self.manipulator_plant.GetBodyByName(
            manipulator.data["contact_body_name"])
        Jdot_qdot_raw = self.manipulator_plant.CalcBiasSpatialAcceleration(
            self.manipulator_plant_context,
            JacobianWrtVariable.kV,
            contact_body.body_frame(),
            [0, 0, 0],
            self.manipulator_plant.world_frame(),
            self.manipulator_plant.world_frame())

        Jdot_qdot = list(Jdot_qdot_raw.rotational()) + \
            list(Jdot_qdot_raw.translational())

        output.SetFromVector(Jdot_qdot)

    # ========================== OTHER FUNCTIONS ==========================
    def update_plant(self, context):
        self.manipulator_plant.SetPositions(
            self.manipulator_plant_context, self.calc_q(context))
        self.manipulator_plant.SetVelocities(
            self.manipulator_plant_context, self.calc_v(context))
