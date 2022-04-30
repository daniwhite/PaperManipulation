import sweeps.sweep_infra
import numpy as np
import plant.simulation
import config

if __name__ == "__main__":
    other_sim_args = {
        "ctrl_paradigm": plant.simulation.CtrlParadigm.IMPEDANCE,
        "impedance_type": plant.simulation.ImpedanceType.FORCE_FB,
        "n_hat_force_compensation_source": 
            plant.simulation.NHatForceCompensationSource.CONTACT_FORCE,
        "impedance_stiffness": [4,4,4,40,40,40],
        "num_links": config.NumLinks.FOUR,
        "timeout": 600,
        "DT": 0.0001,
        "const_ff_Fn": 5,
    }
    sweep_runner = sweeps.sweep_infra.SweepRunner(
        other_sim_args=other_sim_args,
        sweep_args="impedance_scale",
        sweep_vars=10**np.linspace(-1.5, 1.5, 16)
    )
    sweep_runner.run_sweep()
