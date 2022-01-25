import numpy as np

# Drake imports
import pydrake
from pydrake.all import Meshcat, RigidTransform, RollPitchYaw, RotationMatrix

import plant.manipulator

class LinkFeedbackSetpointGenerator(pydrake.systems.framework.LeafSystem):
    """
    Generates an impedance trajectory relative to the edge of the link.
    joint.
    """
    def __init__(self):
        pydrake.systems.framework.LeafSystem.__init__(self)

        # =========================== DECLARE INPUTS ==========================
        self.DeclareVectorInputPort(
            "pose_L_rotational", pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "pose_L_translational", pydrake.systems.framework.BasicVector(3))

        # ========================== DECLARE OUTPUTS ==========================
        self.DeclareVectorOutputPort(
            "x0",
            pydrake.systems.framework.BasicVector(6),
            self.calc_x0
        )
        self.DeclareVectorOutputPort(
            "dx0",
            pydrake.systems.framework.BasicVector(6),
            self.calc_dx0
        )


    def calc_x0(self, context, output):
        x0 = np.zeros(6)

        # Evaluate inputs
        pose_L_rotational = self.GetInputPort(
            "pose_L_rotational").Eval(context)
        pose_L_translational = self.GetInputPort(
            "pose_L_translational").Eval(context)
        theta_L = pose_L_rotational[0]

        # Calc X_L_SP, the transform from the link to the setpoint
        offset_Z_rot = -theta_L
        offset_Z_rot = 0 if offset_Z_rot > 0 else offset_Z_rot
        offset_Z_rot = -np.pi/2 if offset_Z_rot < -np.pi/2 else offset_Z_rot
        X_L_SP = RigidTransform(
            R=RotationMatrix.MakeZRotation(offset_Z_rot),
            p=[0, 0, 0]
        )

        # Calc X_W_L
        X_W_L = RigidTransform(
            p=pose_L_translational,
            R=RotationMatrix(RollPitchYaw(pose_L_rotational))
        )

        # Calc X_W_SP
        X_W_SP = X_W_L.multiply(X_L_SP)

        translation = X_W_SP.translation()
        rotation = RollPitchYaw(X_W_SP.rotation()).vector()
        rotation[0] += plant.manipulator.RotX_L_Md

        x0[:3] = rotation
        x0[3:] = translation
        output.SetFromVector(x0)


    def calc_dx0(self, context, output):
        output.SetFromVector(np.zeros(6))
