# Standard python libraries
from multiprocessing import Pool
import time
import copy

import sys

# Numpy/scipy/matplotlib/etc.
import numpy as np

# Project files
import plant.simulation
import constants
from config import hinge_rotation_axis


def get_max_overall_theta(sim, log):
    idx = sim.log_wrapper.get_idx("pos", "trn", sim.ll_idx)
    length = 3
    p_L = np.expand_dims(log.data()[idx:idx+length].T, 2)

    idx = sim.log_wrapper.get_idx("pos", "trn", sim.paper.link_idxs[0])
    length = 3
    p_FL = np.expand_dims(log.data()[idx:idx+length].T, 2)

    X_FL_FJ_L = sim.paper.joints[0].frame_on_parent().GetFixedPoseInBodyFrame()
    p_W_FJ = p_FL[0] + np.expand_dims(X_FL_FJ_L.translation(), 1)

    overall_thetas = np.arctan2((p_L - p_W_FJ)[:,2], -(p_L - p_W_FJ)[:,1-hinge_rotation_axis]).flatten()

    # Deal with rollover
    overall_thetas_ = np.zeros_like(overall_thetas)
    overall_thetas_[0] = overall_thetas[0]
    for i in range(len(overall_thetas[1:])):
        overall_thetas_[i+1] = overall_thetas[i+1]
        if np.abs(overall_thetas_[i+1] - overall_thetas_[i]) > 2*np.pi*0.99:
            overall_thetas_[i+1] += 2*np.pi

    return np.max(overall_thetas_)


def get_max_F_ON_trace(sim, log):
    num_jnts = sim.paper.num_links.value - 1
    F_ONs = []
    for i in range(num_jnts):
        idx = sim.log_wrapper.joint_entry_start_idx + 6*i
        joint_force_in_compliance_frame = log.data()[idx:idx+3]
        F_ONs.append(joint_force_in_compliance_frame[2,:])
    return np.max(np.abs(F_ONs), axis=0)


# Sweep runner class
class SweepRunner:
    default_sim_args = {
        "exit_when_folded": True,
        "timeout": 600,
        "DT": 0,
        "TSPAN": 35,
        "noise": copy.deepcopy(plant.simulation.default_port_noise_map)
    }

    def __init__(self, other_sim_args, sweep_args, sweep_vars,
            sweep_var_names=None, other_data_attrs={}):
        self.other_data_attrs = other_data_attrs
        # Set sweep var name base on invoked file
        if sweep_var_names is None:
            file_name_found = False
            for arg in sys.argv:
                if arg.endswith('.py'):
                    if file_name_found:
                        raise ValueError(
                            "Multiple python files in args, so can't " + \
                                "automatically detect sweep_var_name!")
                    else:
                        base_name = arg.split('/')[-1][:-3]
                        self.sweep_var_names = base_name.split("__")
                        file_name_found = True
        else:
            self.sweep_var_names = sweep_var_names
        self.sweep_args = np.array(sweep_args).flatten()
        self.sweep_vars = np.array(sweep_vars)
        if len(self.sweep_vars.shape) < 2:
            self.sweep_vars = np.expand_dims(self.sweep_vars, 1)
        print(sweep_vars)
        print(sweep_args)
        if np.all(self.sweep_args != "impedance_stiffness"):
            assert self.sweep_vars.shape[1] == self.sweep_args.shape[0]

        self.sim_args = copy.deepcopy(other_sim_args)
        # Fill in default values if they're note specified
        for k, v in self.default_sim_args.items():
            if k not in self.sim_args.keys() and not (k in self.sweep_args):
                self.sim_args[k] = v
        if "sim_params" not in self.sweep_args:
            # Default depends on num_links
            self.sim_args["sim_params"] = copy.deepcopy(
                constants.nominal_sys_consts(self.sim_args["num_links"]))

    def proc_func(self, sim, log):
        out = {}

        non_vertical_axis = 1 - hinge_rotation_axis

        # Get paper link data
        horizontal_paper_traces = []
        vertical_paper_traces = []
        angle_paper_traces = []
        for b in sim.paper.link_idxs:
            horizontal_paper_traces.append(log.data()[
                sim.log_wrapper.get_idx("pos", "trn", b) + non_vertical_axis])
            vertical_paper_traces.append(log.data()[
                sim.log_wrapper.get_idx("pos", "trn", b) + 2])

            ang_trace = np.array(log.data()[
                sim.log_wrapper.get_idx("pos", "rot", b) + hinge_rotation_axis])
            # Handle RPY stuff
            if hinge_rotation_axis == 1:
                z_angs = log.data()[sim.log_wrapper.get_idx("pos", "rot", b) + 2]
                ang_trace[z_angs > np.pi/2] = ang_trace[z_angs > np.pi/2]*-1 + np.pi
                ang_trace = ang_trace *-1
            angle_paper_traces.append(ang_trace)

        out["horizontal_paper_traces"] = np.array(horizontal_paper_traces)
        out["vertical_paper_traces"] = np.array(vertical_paper_traces)
        out["angle_paper_traces"] = np.array(angle_paper_traces)

        # Get times
        out["ctrl_first_contact_idx"] = np.argmax(
            log.data()[sim.log_wrapper.calc_in_contact_start_idx])
        out["any_links_in_contact_idx"] = np.argmax(
            log.data()[sim.log_wrapper.any_links_in_contact_idx])
        out["times"] = log.sample_times()

        # Get manipulator traces
        out["horizontal_manipulator_trace"] = log.data()[sim.log_wrapper.get_idx(
                "pos", "trn", sim.contact_body_idx) + non_vertical_axis,:]
        out["vertical_manipulator_trace"] = log.data()[sim.log_wrapper.get_idx(
                "pos", "trn", sim.contact_body_idx) + 2,:]

        # Get figures of merit/other useful traces
        out["overall_theta_times"] = sim.exit_system.overall_thetas
        out["overall_thetas"] = sim.exit_system.overall_thetas
        out["max_overall_theta"] = np.max(sim.exit_system.overall_thetas)
        out["max_F_ONs"] = get_max_F_ON_trace(sim, log)
        return out

    def sweep_func(self, vals):
        vals = vals.flatten()
        sim_args = copy.deepcopy(self.sim_args)
        printing_label = "[ "
        for val, arg in zip(vals, self.sweep_args):
            if arg in vars(sim_args["sim_params"]).keys(): # TODO: fix mismatch?
                setattr(sim_args["sim_params"], arg, val)
            elif arg.startswith("noise__"):
                key_name = arg[len("noise__"):]
                sim_args["noise"][key_name] = val
            else:
                sim_args[arg] = val
            try:
                sweep_label = "{:5.2f}".format(val)
            except (TypeError, ValueError):
                sweep_label = str(val)
            printing_label += "{} = {} ".format(arg, sweep_label)
        printing_label += " ]"
        sim = plant.simulation.Simulation(**sim_args)

        print(printing_label + " Starting")

        # Run sim
        t_start_ = time.time()
        log = sim.run_sim()
        print(printing_label + " Total time: {:.2f}".format(
            time.time() - t_start_))

        # Grab output var
        output_var = self.proc_func(sim, log)

        return (output_var, sim.success, sim.exit_message, sim_args)

    def run_sweep(self):
        t_start__ = time.time()
        with Pool(8) as p:
            sweep_result = p.map(self.sweep_func, self.sweep_vars)
            out_data = [val[0] for val in sweep_result]
            successes = [val[1] for val in sweep_result]
            exit_messages = [val[2] for val in sweep_result]
            sim_args = [val[3] for val in sweep_result]
            base_name = "__".join(self.sweep_var_names)
            np.savez("sweeps/" + base_name + ".npz",
                sweep_vars=self.sweep_vars,
                out_data=out_data,
                successes=successes,
                exit_messages=exit_messages,
                sim_args=sim_args,
                **self.other_data_attrs
            )
        print("OVERALL RUNTIME:", time.time() - t_start__)
