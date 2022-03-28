# Standard python libraries
from multiprocessing import Pool
import time
import copy

import sys

# Numpy/scipy/matplotlib/etc.
import numpy as np

# Project files
import plant.simulation

class SweepRunner:
    default_sim_args = {
        "exit_when_folded": True,
        "timeout": 180
    }

    def __init__(self, proc_func, other_sim_args, sweep_arg, sweep_vars,
            sweep_var_name=None):
        
        # Set sweep var name base on invoked file
        if sweep_var_name is None:
            file_name_found = False
            for arg in sys.argv:
                if arg.endswith('.py'):
                    if file_name_found:
                        raise ValueError(
                            "Multiple python files in args, so can't " + \
                                "automatically detect sweep_var_name!")
                    else:
                        self.sweep_var_name = arg.split('/')[-1][:-3]
                        file_name_found = True
        else:
            self.sweep_var_name = sweep_var_name

        self.sim_args = copy.deepcopy(other_sim_args)
        for k, v in self.default_sim_args.items():
            if k not in self.sim_args.keys() and k != sweep_arg:
                self.sim_args[k] = v

        self.proc_func = proc_func
        self.sweep_arg = sweep_arg
        self.sweep_vars = sweep_vars

    def sweep_func(self, val):
        sim_args = copy.deepcopy(self.sim_args)
        sim_args[self.sweep_arg] = val
        sim = plant.simulation.Simulation(**sim_args)

        print("[ {} = {:5.2f} ] Starting".format(self.sweep_arg, val))

        # Run sim
        t_start_ = time.time()
        log = sim.run_sim()
        print("[ ff_Fn = {:5.2f} ] Total time: {:.2f}".format(
            val, time.time() - t_start_))

        # Grab output var
        output_var = self.proc_func(sim, log)

        return (output_var, sim.success, sim.exit_message)

    def run_sweep(self):
        t_start__ = time.time()
        with Pool(8) as p:
            x_axis = self.sweep_vars
            sweep_result = p.map(self.sweep_func, x_axis)
            y_axis = [val[0] for val in sweep_result]
            successes = [val[1] for val in sweep_result]
            exit_messages = [val[2] for val in sweep_result]
            np.savez("sweep_scripts/" + self.sweep_var_name + ".npz",
                x_axis=x_axis,
                y_axis=y_axis,
                successes=successes,
                exit_messages=exit_messages
            )
            print("Result:", y_axis)
        print("OVERALL RUNTIME:", time.time() - t_start__)
