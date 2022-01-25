import numpy as np

import pydrake
from pydrake.geometry import Cylinder, Rgba
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.all import (
    RigidTransform, RollPitchYaw, RotationMatrix, Meshcat
)

# This function taken from:
# https://github.com/RussTedrake/manipulation/blob/008cec6343dd39063705287e6664a3fee71a43b8/manipulation/meshcat_cpp_utils.py
def AddMeshcatTriad(meshcat,
                    path,
                    length=.25,
                    radius=0.01,
                    opacity=1.,
                    X_PT=RigidTransform()):
    meshcat.SetTransform(path, X_PT)
    # x-axis
    X_TG = RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2),
                          [length / 2., 0, 0])
    meshcat.SetTransform(path + "/x-axis", X_TG)
    meshcat.SetObject(path + "/x-axis", Cylinder(radius, length),
                      Rgba(1, 0, 0, opacity))

    # y-axis
    X_TG = RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2),
                          [0, length / 2., 0])
    meshcat.SetTransform(path + "/y-axis", X_TG)
    meshcat.SetObject(path + "/y-axis", Cylinder(radius, length),
                      Rgba(0, 1, 0, opacity))

    # z-axis
    X_TG = RigidTransform([0, 0, length / 2.])
    meshcat.SetTransform(path + "/z-axis", X_TG)
    meshcat.SetObject(path + "/z-axis", Cylinder(radius, length),
                      Rgba(0, 0, 1, opacity))

class FrameVisualizer(pydrake.systems.framework.LeafSystem):
    """
    Take in a rotation and position and visualize the corresponding set of axes
    in meshcat.
    """
    def __init__(self, meshcat: Meshcat, name: str):
        pydrake.systems.framework.LeafSystem.__init__(self)
        self.meshcat = meshcat
        self.name = name

        # =========================== DECLARE INPUTS ==========================
        self.DeclareVectorInputPort(
            "pos",
            pydrake.systems.framework.BasicVector(3))
        self.DeclareVectorInputPort(
            "rot",
            pydrake.systems.framework.BasicVector(3))

        # ========================== DECLARE UPDATES ==========================
        self.DeclarePerStepEvent(
            pydrake.systems.framework.PublishEvent(
                self.update_frame))

    def update_frame(self, context, event):
        pos = np.expand_dims(
            np.array(self.GetInputPort("pos").Eval(context)), 1)
        rot = np.expand_dims(
            np.array(self.GetInputPort("rot").Eval(context)), 1)

        X = RigidTransform(p=pos, R=RotationMatrix(RollPitchYaw(rot)))

        AddMeshcatTriad(self.meshcat, self.name, X_PT=X)
