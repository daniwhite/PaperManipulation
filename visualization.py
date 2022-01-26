import numpy as np

import pydrake
from pydrake.geometry import Cylinder, Rgba, MeshcatAnimation
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.all import (
    RigidTransform, RollPitchYaw, RotationMatrix, Meshcat
)

# This function mostly taken from:
# https://github.com/RussTedrake/manipulation/blob/008cec6343dd39063705287e6664a3fee71a43b8/manipulation/meshcat_cpp_utils.py
# ^This didn't have support for editing the animation
def AddMeshcatTriad(meshcat,
                    path,
                    length=.25,
                    radius=0.01,
                    opacity=1.,
                    X_PT=RigidTransform(),
                    frame_time=None,
                    ani=None):

    meshcat.SetTransform(path, X_PT)

    # x-axis
    X_TGx = RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2),
                          [length / 2., 0, 0])
    meshcat.SetTransform(path + "/x-axis", X_TGx)
    meshcat.SetObject(path + "/x-axis", Cylinder(radius, length),
                        Rgba(1, 0, 0, opacity))

    # y-axis
    X_TGy = RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2),
                          [0, length / 2., 0])
    meshcat.SetTransform(path + "/y-axis", X_TGy)
    meshcat.SetObject(path + "/y-axis", Cylinder(radius, length),
                    Rgba(0, 1, 0, opacity))

    # z-axis
    X_TGz = RigidTransform([0, 0, length / 2.])
    meshcat.SetTransform(path + "/z-axis", X_TGz)
    meshcat.SetObject(path + "/z-axis", Cylinder(radius, length),
                    Rgba(0, 0, 1, opacity))

    if ani is not None:
        frame = ani.frame(frame_time)
        ani.SetTransform(frame, path, X_PT)
        ani.SetTransform(frame, path + "/x-axis", X_TGx)
        ani.SetTransform(frame, path + "/y-axis", X_TGy)
        ani.SetTransform(frame, path + "/z-axis", X_TGz)

class FrameVisualizer(pydrake.systems.framework.LeafSystem):
    """
    Take in a rotation and position and visualize the corresponding set of axes
    in meshcat.
    """
    def __init__(self, meshcat: Meshcat, name: str, opacity=1):
        pydrake.systems.framework.LeafSystem.__init__(self)
        self.meshcat = meshcat
        self.name = name
        self.ani = None
        self.opacity = opacity

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

        AddMeshcatTriad(self.meshcat, self.name, X_PT=X, 
            frame_time=context.get_time(), ani=self.ani, opacity=self.opacity)
    
    def set_animation(self, ani: MeshcatAnimation):
        self.ani = ani
