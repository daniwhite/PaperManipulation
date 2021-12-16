"""
Systems that are not explicit control strategies but provide useful auxiliarly
functions.
"""
# Drake imports
import pydrake
import numpy as np
import plant.manipulator as manipulator
from plant.paper import settling_time

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

    def __init__(self, v_MNd=0.1, K=10):
        pydrake.systems.framework.LeafSystem.__init__(self)
        # System parameters
        self.nq = manipulator.data["nq"]

        # Gains
        self.v_MNd = v_MNd
        self.K = K

        # Intermediary terms
        self.init_q = None

        self.DeclareVectorInputPort(
            "J",
            pydrake.systems.framework.BasicVector(6*self.nq))
        self.DeclareVectorInputPort(
            "v_MN", pydrake.systems.framework.BasicVector(1))
        self.DeclareVectorInputPort(
            "N_hat", pydrake.systems.framework.BasicVector(3))

        self.DeclareVectorOutputPort(
            "tau_out", pydrake.systems.framework.BasicVector(self.nq),
            self.CalcOutput)

    def CalcOutput(self, context, output):
        # Load inputs
        J = self.GetInputPort("J").Eval(context).reshape(
            (6, self.nq))
        v_MN = self.GetInputPort("v_MN").Eval(context)
        N_hat = np.expand_dims(self.GetInputPort("N_hat").Eval(context), 1)

        F_CN = (self.v_MNd - v_MN)*self.K

        F_d = np.zeros((6,1))
        if context.get_time() > settling_time:
            F_d[3:] += N_hat * F_CN

        tau_ctrl = J.T@F_d
        output.SetFromVector(tau_ctrl.flatten()) 


