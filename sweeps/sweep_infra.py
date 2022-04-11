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

# General helper functions
def get_max_theta_L(sim, log):
    theta_L = log.data()[sim.log_wrapper.get_idx(
        "pos", "rot", sim.ll_idx) + hinge_rotation_axis].copy()
    theta_LZ = log.data()[sim.log_wrapper.get_idx("pos", "rot", sim.ll_idx)+2]
    # Fix issue in RPY singularity
    theta_L[theta_LZ > np.pi/2] = theta_L[theta_LZ > np.pi/2]*-1 + np.pi

    return np.max(theta_L)

def get_max_overall_theta(sim, log):
    idx = sim.log_wrapper.get_idx("pos", "trn", sim.ll_idx)
    length = 3
    p_L = np.expand_dims(log.data()[idx:idx+length].T, 2)

    idx = sim.log_wrapper.get_idx("pos", "trn", sim.paper.link_idxs[0])
    length = 3
    p_FL = np.expand_dims(log.data()[idx:idx+length].T, 2)

    X_FL_FJ_L = sim.paper.joints[0].frame_on_parent().GetFixedPoseInBodyFrame()
    p_W_FJ = p_FL[0] + np.expand_dims(X_FL_FJ_L.translation(), 1)

    overall_thetas = np.arctan2((p_L - p_W_FJ)[:,2], -(p_L - p_W_FJ)[:,1-hinge_rotation_axis])
    return np.max(overall_thetas)


# Sweep runner class
class SweepRunner:
    default_sim_args = {
        "exit_when_folded": True,
        "timeout": 600,
        "DT": 0,
        "TSPAN": 35
    }

    def __init__(self, proc_func, other_sim_args, sweep_args, sweep_vars,
            sweep_var_names=None):

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

        self.proc_func = proc_func
        self.sweep_args = np.array(sweep_args).flatten()
        self.sweep_vars = np.array(sweep_vars)
        if len(self.sweep_vars.shape) < 2:
            self.sweep_vars = np.expand_dims(self.sweep_vars, 1)
        assert self.sweep_vars.shape[1] == self.sweep_args.shape[0]

        self.sim_args = copy.deepcopy(other_sim_args)
        for k, v in self.default_sim_args.items():
            if k not in self.sim_args.keys() and not (k in self.sweep_args):
                self.sim_args[k] = v
        # Default depend son num_links
        if "sim_params" not in self.sweep_args:
            self.sim_args["sim_params"] = copy.deepcopy(
                constants.nominal_sys_consts(self.sim_args["num_links"]))

    def sweep_func(self, vals):
        vals = vals.flatten()
        sim_args = copy.deepcopy(self.sim_args)
        printing_label = "[ "
        for val, arg in zip(vals, self.sweep_args):
            if arg in vars(sim_args["sim_params"]).keys(): # TODO: fix mismatch?
                setattr(sim_args["sim_params"], arg, val)
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
            x_axis = self.sweep_vars
            sweep_result = p.map(self.sweep_func, x_axis)
            y_axis = [val[0] for val in sweep_result]
            successes = [val[1] for val in sweep_result]
            exit_messages = [val[2] for val in sweep_result]
            sim_args = [val[3] for val in sweep_result]
            base_name = "__".join(self.sweep_var_names)
            np.savez("sweeps/" + base_name + ".npz",
                x_axis=x_axis,
                y_axis=y_axis,
                successes=successes,
                exit_messages=exit_messages,
                sim_args=sim_args
            )
            print("Result:", y_axis)
        print("OVERALL RUNTIME:", time.time() - t_start__)
