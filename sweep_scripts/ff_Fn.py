import sweep_scripts.sweep_runner
import numpy as np
from config import hinge_rotation_axis
import plant.simulation

def proc_func(sim, log):
    theta_L = log.data()[sim.log_wrapper.get_idx(
        "pos", "rot", sim.ll_idx) + hinge_rotation_axis].copy()
    theta_LZ = log.data()[sim.log_wrapper.get_idx("pos", "rot", sim.ll_idx)+2]
    # Fix issue in RPY singularity
    theta_L[theta_LZ > np.pi/2] = theta_L[theta_LZ > np.pi/2]*-1 + np.pi

    return np.max(theta_L)

print(__file__.split('/')[-1][:-3])

if __name__ == "__main__":
    other_sim_args = {
        "ctrl_paradigm": plant.simulation.CtrlParadigm.IMPEDANCE,
        "impedance_type": plant.simulation.ImpedanceType.LINK_FB,
        "n_hat_force_compensation_source": 
            plant.simulation.NHatForceCompensationSource.CONSTANT
    }
    sweep_runner = sweep_scripts.sweep_runner.SweepRunner(
        proc_func=proc_func,
        other_sim_args=other_sim_args,
        sweep_arg="N_constant_ff_F",
        sweep_vars=np.linspace(0, 10, 16),
    )
    sweep_runner.run_sweep()
