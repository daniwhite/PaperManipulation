"""Defines wrapper class to pack inputs for a logger."""
# Drake imports
import pydrake
from pydrake.all import RigidTransform, RollPitchYaw, SpatialVelocity, SpatialAcceleration, ContactResults, SpatialForce, BasicVector
import numpy as np
import plant.manipulator as manipulator

class LogWrapper(pydrake.systems.framework.LeafSystem):
    """
    Wrapper system that converts RigidTransform and SpatialVelocity inputs into vector output so it
    can be used easily with a logger.
    """

    def __init__(self, num_bodies, contact_body_idx, paper):
        pydrake.systems.framework.LeafSystem.__init__(self)
        # Offsets
        self.type_strs_to_offsets = {
            "pos": 0,
            "vel": 6,
            "acc": 12,
        }
        self.direction_to_offset = {
            "trn": 0,
            "rot": 3
        }
        self.translational_offset = 0
        self.rotational_offset = 3


        # General constants/members
        self.max_contacts = 10
        self.nq_manipulator = manipulator.data['nq']
        self.paper = paper


        # Numbers of entries
        self.entries_per_body = 3*6
        self.entries_per_contact = 18
        self.contact_entries = self.entries_per_contact*self.max_contacts
        self.joint_entries = len(paper.joints)*6
        self.ctrl_forces_entries = 3
        self.gen_accs_entries = self.nq_manipulator
        self.state_entries = self.nq_manipulator*2 + 1
        self.tau_g_entries = self.nq_manipulator
        self.M_entries = self.nq_manipulator*self.nq_manipulator


        # Start indices
        # PROGRAMMING: Make this into a for loop?
        self._size = num_bodies*self.entries_per_body

        self.contact_entry_start_idx = self._size
        self._size += self.contact_entries

        self.joint_entry_start_idx = self._size
        self._size += self.joint_entries

        self.gen_accs_start_idx = self._size
        self._size += self.gen_accs_entries

        self.state_start_idx = self._size
        self._size += self.state_entries

        self.tau_g_start_idx = self._size
        self._size +=  self.tau_g_entries

        self.M_start_idx = self._size
        self._size += self.M_entries


        # Poses, velocities, accelerations, etc
        self.DeclareAbstractInputPort(
            "poses", pydrake.common.value.AbstractValue.Make([RigidTransform(), RigidTransform()]))
        self.DeclareAbstractInputPort(
            "vels", pydrake.common.value.AbstractValue.Make([SpatialVelocity(), SpatialVelocity()]))
        self.DeclareAbstractInputPort(
            "accs", pydrake.common.value.AbstractValue.Make([SpatialAcceleration(), SpatialAcceleration()]))
        
        # Other forces/accelerations
        self.DeclareAbstractInputPort(
            "contact_results",
            pydrake.common.value.AbstractValue.Make(ContactResults()))
        self.DeclareAbstractInputPort(
            "joint_forces", pydrake.common.value.AbstractValue.Make([SpatialForce(), SpatialForce()]))
        self.DeclareVectorInputPort(
            "manipulator_accs", pydrake.systems.framework.BasicVector(self.nq_manipulator))

        # q and v
        self.DeclareVectorInputPort("state", BasicVector(self.nq_manipulator*2))

        # Manipulator equation terms
        self.DeclareVectorInputPort(
            "tau_g",
            pydrake.systems.framework.BasicVector(self.nq_manipulator))
        self.DeclareVectorInputPort(
            "M",
            pydrake.systems.framework.BasicVector(
                self.nq_manipulator*self.nq_manipulator))
        self.DeclareVectorOutputPort(
            "out", pydrake.systems.framework.BasicVector(
                self._size),
            self.CalcOutput)

        self.contact_body_idx = contact_body_idx
        self.ll_idx = paper.link_idxs[-1]

    def CalcOutput(self, context, output):
        out = []
        poses = self.GetInputPort("poses").Eval(context)
        vels = self.GetInputPort("vels").Eval(context)
        accs = self.GetInputPort("accs").Eval(context)
        contact_results = self.GetInputPort("contact_results").Eval(context)
        joint_forces = self.GetInputPort("joint_forces").Eval(context)
        manipulator_accs = self.GetInputPort("manipulator_accs").Eval(context)
        state = self.GetInputPort("state").Eval(context)
        tau_g = self.GetInputPort("tau_g").Eval(context)
        M = self.GetInputPort("M").Eval(context)

        # Add body poses, velocities, accelerations, etc.
        for i, (pose, vel, acc) in enumerate(zip(poses, vels, accs)):
            assert len(out) == self.get_idx("pos", "trn", i)
            out += list(pose.translation())

            rot_vec = RollPitchYaw(pose.rotation()).vector()

            assert len(out) == self.get_idx("pos", "rot", i)
            out += list(rot_vec)

            assert len(out) == self.get_idx("vel", "trn", i)
            out += list(vel.translational())
            assert len(out) == self.get_idx("vel", "rot", i)
            out += list(vel.rotational())
            assert len(out) == self.get_idx("acc", "trn", i)
            out += list(acc.translational())
            assert len(out) == self.get_idx("acc", "rot", i)
            out += list(acc.rotational())

        # Add contact results
        forces_found = 0
        for i in range(contact_results.num_point_pair_contacts()):
            point_pair_contact_info = \
                contact_results.point_pair_contact_info(i)

            use_this_contact = False
            # Always take contact forces on manipulator
            if int(point_pair_contact_info.bodyA_index()) == self.contact_body_idx \
                    and int(point_pair_contact_info.bodyB_index()) == self.ll_idx:
                # PROGRAMMING: Move sparation speed back into this section with correct signs
                out += list(-point_pair_contact_info.contact_force())
                out += list(point_pair_contact_info.point_pair().p_WCa)
                out += list(point_pair_contact_info.point_pair().p_WCb)
                use_this_contact = True
            elif int(point_pair_contact_info.bodyA_index()) == self.ll_idx \
                    and int(point_pair_contact_info.bodyB_index()) == self.contact_body_idx:
                out += list(point_pair_contact_info.contact_force())
                out += list(point_pair_contact_info.point_pair().p_WCb)
                out += list(point_pair_contact_info.point_pair().p_WCa)
                use_this_contact = True
            if use_this_contact:
                forces_found += 1
                assert  self.max_contacts >= forces_found
                out += [point_pair_contact_info.separation_speed(),
                        point_pair_contact_info.slip_speed()]
                out += list(point_pair_contact_info.contact_point())
                pen_point_pair = point_pair_contact_info.point_pair()
                out += list(pen_point_pair.nhat_BA_W)
                out += [pen_point_pair.depth]
        forces_found_idx = forces_found
        while forces_found_idx < self.max_contacts:
            out += [np.nan]*self.entries_per_contact
            forces_found_idx += 1

        # Add joint forces
        for i, j in enumerate(self.paper.joints):
            if i == len(self.paper.joints) - 1:
                assert self.get_last_jnt_idx("trn") == len(out) 
            out += list(joint_forces[int(j.index())].translational())
            if i == len(self.paper.joints) - 1:
                assert self.get_last_jnt_idx("rot") == len(out) 
            out += list(joint_forces[int(j.index())].rotational())
        # Other generic terms
        assert len(out) == self.gen_accs_start_idx
        out += list(manipulator_accs)
        assert len(out) == self.state_start_idx
        out += list(state)
        out += [forces_found]
        assert(len(out) == self.tau_g_start_idx)
        out += list(tau_g)
        assert(len(out) == self.M_start_idx)
        out += list(M)

        output.SetFromVector(out)

    def get_idx(self, type_str, direction, body_idx):
        return body_idx*self.entries_per_body + self.type_strs_to_offsets[type_str] + \
            self.direction_to_offset[direction]
    
    def get_last_jnt_idx(self, direction):
        idx = (len(self.paper.joints)-1)*6 + self.joint_entry_start_idx
        if direction == "trn":
            idx += 0
        elif direction == "rot":
            idx += 3
        else:
            raise ValueError
        return idx
