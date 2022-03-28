# Standard python libraries
from multiprocessing import Pool
import time

# Numpy/scipy/matplotlib/etc.
import numpy as np

# Project files
import plant.simulation
from config import hinge_rotation_axis

def sweep_ff_Fn(N_constant_ff_F):
    # Initialize parameters
    ctrl_paradigm = plant.simulation.CtrlParadigm.IMPEDANCE
    impedance_type = plant.simulation.ImpedanceType.LINK_FB
    n_hat_force_compensation_source = \
        plant.simulation.NHatForceCompensationSource.CONSTANT

    # Create sim object
    sim = plant.simulation.Simulation(
        ctrl_paradigm=ctrl_paradigm,
        impedance_type=impedance_type,
        n_hat_force_compensation_source=n_hat_force_compensation_source,
        exit_when_folded=True,
        N_constant_ff_F=N_constant_ff_F,
        timeout=180
    )
    print("[ ff_Fn = {:5.2f} ] Starting".format(sim.N_constant_ff_F))

    # Run sim
    t_start_ = time.time()
    log = sim.run_sim()
    print("[ ff_Fn = {:5.2f} ] Total time: {:.2f}".format(
        sim.N_constant_ff_F, time.time() - t_start_))
    
    # Grab output var
    theta_L = log.data()[sim.log_wrapper.get_idx(
        "pos", "rot", sim.ll_idx) + hinge_rotation_axis].copy()
    theta_LZ = log.data()[sim.log_wrapper.get_idx("pos", "rot", sim.ll_idx)+2]
    # Fix issue in RPY singularity
    theta_L[theta_LZ > np.pi/2] = theta_L[theta_LZ > np.pi/2]*-1 + np.pi
    
    return np.max(theta_L)

if __name__ == '__main__':
    t_start__ = time.time()
    with Pool(8) as p:
        x_axis = np.linspace(0, 10, 8)
        y_axis = p.map(sweep_ff_Fn, x_axis)
        np.savez("sweep_scripts/ff_Fn.npz", x_axis=x_axis, y_axis=y_axis)
        print("Result:", y_axis)
    print("OVERALL RUNTIME:", time.time() - t_start__)
