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
        "timeout": 600,
        "DT": 0.0001,
        "const_ff_Fn": 5
    }

    mus = np.concatenate((np.linspace(0,0.4,12),np.linspace(0.5,1,4)))
    sweep_runner = sweeps.sweep_infra.SweepRunner(
        other_sim_args=other_sim_args,
        sweep_args="mu",
        sweep_vars=mus,
    )
    sweep_runner.run_sweep()
