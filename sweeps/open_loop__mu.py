import sweeps.sweep_infra
import numpy as np
import plant.simulation
import config

if __name__ == "__main__":
    other_sim_args = {
        "ctrl_paradigm": plant.simulation.CtrlParadigm.IMPEDANCE,
        "impedance_type": plant.simulation.ImpedanceType.OFFLINE_TRAJ,
        "n_hat_force_compensation_source": 
            plant.simulation.NHatForceCompensationSource.CONSTANT,
        "impedance_stiffness": [40,40,40,400,400,400],
        "num_links": config.NumLinks.FOUR,
        "timeout": 600,
        "DT": 0.0001,
        "const_ff_Fn": 0
    }
    mus = np.concatenate((np.linspace(0,0.4,12),np.linspace(0.5,1,4)))
    sweep_runner = sweeps.sweep_infra.SweepRunner(
        other_sim_args=other_sim_args,
        sweep_args="mu",
        sweep_vars=mus,
    )
    sweep_runner.run_sweep()
