# Drake imports
import pydrake
from pydrake.all import RigidTransform, SpatialVelocity, RollPitchYaw

class VisionSystem(pydrake.systems.framework.LeafSystem):
    """
    Simulates what we get out of our vision system:
    """

    # PROGRAMMING: Clean up passed around references
    def __init__(self, ll_idx, contact_body_idx):
        pydrake.systems.framework.LeafSystem.__init__(self)

        self.DeclareAbstractInputPort(
            "poses", pydrake.common.value.AbstractValue.Make([RigidTransform(), RigidTransform()]))
        self.DeclareAbstractInputPort(
            "vels", pydrake.common.value.AbstractValue.Make([SpatialVelocity(), SpatialVelocity()]))

        self.DeclareVectorOutputPort(
            "pose_L_translational", pydrake.systems.framework.BasicVector(3),
            self.calc_pose_L_translational)
        self.DeclareVectorOutputPort(
            "pose_L_rotational", pydrake.systems.framework.BasicVector(3),
            self.calc_pose_L_rotational)
        self.DeclareVectorOutputPort(
            "vel_L_translational", pydrake.systems.framework.BasicVector(3),
            self.calc_vel_L_translational)
        self.DeclareVectorOutputPort(
            "vel_L_rotational", pydrake.systems.framework.BasicVector(3),
            self.calc_vel_L_rotational)
        self.DeclareVectorOutputPort(
            "pose_M_translational", pydrake.systems.framework.BasicVector(3),
            self.calc_pose_M_translational)
        self.DeclareVectorOutputPort(
            "pose_M_rotational", pydrake.systems.framework.BasicVector(3),
            self.calc_pose_M_rotational)
        self.DeclareVectorOutputPort(
            "vel_M_translational", pydrake.systems.framework.BasicVector(3),
            self.calc_vel_M_translational)
        self.DeclareVectorOutputPort(
            "vel_M_rotational", pydrake.systems.framework.BasicVector(3),
            self.calc_vel_M_rotational)
        
        self.ll_idx = ll_idx
        self.contact_body_idx = contact_body_idx

        # self._size = 6*24

    def calc_pose_L_translational(self, context, output):
        poses = self.GetInputPort("poses").Eval(context)
        output.SetFromVector(list(poses[self.ll_idx].translation()))
        # output.SetFromVector([0, 0, 0])
    
    def calc_pose_L_rotational(self, context, output):
        poses = self.GetInputPort("poses").Eval(context)
        output.SetFromVector(
            list(
                RollPitchYaw(
                    poses[self.ll_idx].rotation()
                    ).vector()
            )
        )
        # output.SetFromVector([0, 0, 0])

    def calc_vel_L_translational(self, context, output):
        vels = self.GetInputPort("vels").Eval(context)
        output.SetFromVector(list(vels[self.ll_idx].translational()))
        # output.SetFromVector([0, 0, 0])
    
    def calc_vel_L_rotational(self, context, output):
        vels = self.GetInputPort("vels").Eval(context)
        output.SetFromVector(list(vels[self.ll_idx].rotational()))
        # output.SetFromVector([0, 0, 0])

    def calc_pose_M_translational(self, context, output):
        poses = self.GetInputPort("poses").Eval(context)
        output.SetFromVector(list(poses[self.contact_body_idx].translation()))
        # output.SetFromVector([0, 0, 0])
    
    def calc_pose_M_rotational(self, context, output):
        poses = self.GetInputPort("poses").Eval(context)
        output.SetFromVector(
            list(
                RollPitchYaw(
                    poses[self.contact_body_idx].rotation()
                    ).vector()
            )
        )

    def calc_vel_M_translational(self, context, output):
        vels = self.GetInputPort("vels").Eval(context)
        output.SetFromVector(list(vels[self.contact_body_idx].translational()))
        # output.SetFromVector([0, 0, 0])
    
    def calc_vel_M_rotational(self, context, output):
        vels = self.GetInputPort("vels").Eval(context)
        output.SetFromVector(list(vels[self.contact_body_idx].rotational()))
        # output.SetFromVector([0, 0, 0])
