import sweeps.sweep_infra
import numpy as np
import plant.simulation
import config

if __name__ == "__main__":
    other_sim_args = {
        "ctrl_paradigm": plant.simulation.CtrlParadigm.IMPEDANCE,
        "impedance_type": plant.simulation.ImpedanceType.LINK_FB,
        "n_hat_force_compensation_source": 
            plant.simulation.NHatForceCompensationSource.PURE_FN,
        "impedance_stiffness": [4,4,4,40,40,40],
        "num_links": config.NumLinks.FOUR,
        "timeout": 1500,
        "DT": 0,
        "const_ff_Fn": 5
    }
    sweep_runner = sweeps.sweep_infra.SweepRunner(
        proc_func=sweeps.sweep_infra.get_max_overall_theta,
        other_sim_args=other_sim_args,
        sweep_args="b_J",
        sweep_vars=10**np.linspace(-2, 0.5, 16),
    )
    sweep_runner.run_sweep()
