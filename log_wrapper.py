"""Defines wrapper class to pack inputs for a logger."""
# Drake imports
import pydrake
from pydrake.all import RigidTransform, RollPitchYaw, SpatialVelocity, SpatialAcceleration, ContactResults, SpatialForce
import numpy as np


class LogWrapper(pydrake.systems.framework.LeafSystem):
    """
    Wrapper system that converts RigidTransform and SpatialVelocity inputs into vector output so it
    can be used easily with a logger.
    """

    # PROGRAMMING: Clean up passed around references
    def __init__(self, num_bodies, finger_idx, paper, jnt_frc_log):
        pydrake.systems.framework.LeafSystem.__init__(self)
        self.entries_per_body = 3*6
        self.contact_entries = 18
        self.joint_entries = len(paper.joints)*6
        self.ctrl_forces_entries = 3

        self.contact_entry_start_idx = num_bodies*self.entries_per_body
        self.joint_entry_start_idx = num_bodies * \
            self.entries_per_body + self.contact_entries
        self.ctrl_forces_start_idx = self.joint_entry_start_idx + self.joint_entries

        self._size = num_bodies*self.entries_per_body + \
            self.contact_entries + self.joint_entries + self.ctrl_forces_entries
        self.paper = paper
        self.jnt_frc_log = jnt_frc_log

        self.DeclareAbstractInputPort(
            "poses", pydrake.common.value.AbstractValue.Make([RigidTransform(), RigidTransform()]))
        self.DeclareAbstractInputPort(
            "vels", pydrake.common.value.AbstractValue.Make([SpatialVelocity(), SpatialVelocity()]))
        self.DeclareAbstractInputPort(
            "accs", pydrake.common.value.AbstractValue.Make([SpatialAcceleration(), SpatialAcceleration()]))
        self.DeclareAbstractInputPort(
            "contact_results",
            pydrake.common.value.AbstractValue.Make(ContactResults()))
        self.DeclareAbstractInputPort(
            "joint_forces", pydrake.common.value.AbstractValue.Make([SpatialForce(), SpatialForce()]))
        self.DeclareVectorInputPort(
            "finger_actuation", pydrake.systems.framework.BasicVector(self.ctrl_forces_entries))
        self.DeclareVectorOutputPort(
            "out", pydrake.systems.framework.BasicVector(
                self._size),
            self.CalcOutput)

        self.finger_idx = finger_idx
        self.ll_idx = paper.link_idxs[-1]

    def CalcOutput(self, context, output):
        out = []
        poses = self.get_input_port(0).Eval(context)
        vels = self.get_input_port(1).Eval(context)
        accs = self.get_input_port(2).Eval(context)
        contact_results = self.get_input_port(3).Eval(context)
        joint_forces = self.get_input_port(4).Eval(context)
        ctrl_forces = self.get_input_port(5).Eval(context)
        # PROGRAMMING: Better interface fro this
        self.jnt_frc_log.append(
            joint_forces[int(self.paper.joints[-1].index())])
        for pose, vel, acc in zip(poses, vels, accs):
            out += list(pose.translation())
            rot_vec = RollPitchYaw(pose.rotation()).vector()

            # AngleAxis is more convenient because of where it wraps around
            rot_vec[1] = pose.rotation().ToAngleAxis().angle()
            if sum(pose.rotation().ToAngleAxis().axis()) < 0:
                rot_vec[1] *= -1

            out += list(rot_vec)
            out += list(vel.translational())
            out += list(vel.rotational())
            out += list(acc.translational())
            out += list(acc.rotational())
        force_found = False
        for i in range(contact_results.num_point_pair_contacts()):
            point_pair_contact_info = \
                contact_results.point_pair_contact_info(i)

            use_this_contact = False
            # Always take contact forces on manipulator
            if int(point_pair_contact_info.bodyA_index()) == self.finger_idx \
                    and int(point_pair_contact_info.bodyB_index()) == self.ll_idx:
                out += list(-point_pair_contact_info.contact_force())
                # PROGRAMMING: Move sparation speed back into this section with correct signs
                use_this_contact = True
                force_found = True
                out += list(point_pair_contact_info.point_pair().p_WCa)
                out += list(point_pair_contact_info.point_pair().p_WCb)
            elif int(point_pair_contact_info.bodyA_index()) == self.ll_idx \
                    and int(point_pair_contact_info.bodyB_index()) == self.finger_idx:
                out += list(point_pair_contact_info.contact_force())
                use_this_contact = True
                force_found = True
                out += list(point_pair_contact_info.point_pair().p_WCb)
                out += list(point_pair_contact_info.point_pair().p_WCa)
            if use_this_contact:
                out += [point_pair_contact_info.separation_speed(),
                        point_pair_contact_info.slip_speed()]
                out += list(point_pair_contact_info.contact_point())
                pen_point_pair = point_pair_contact_info.point_pair()
                out += list(pen_point_pair.nhat_BA_W)
                out += [pen_point_pair.depth]
        if not force_found:
            out += [np.nan]*self.contact_entries
        for j in self.paper.joints:
            out += list(joint_forces[int(j.index())].translational())
            out += list(joint_forces[int(j.index())].rotational())
        out += list(ctrl_forces)
        output.SetFromVector(out)
