import sweeps.sweep_infra
import numpy as np
import plant.simulation
from config import NumLinks

if __name__ == "__main__":
    other_sim_args = {
        "ctrl_paradigm": plant.simulation.CtrlParadigm.IMPEDANCE,
        "impedance_type": plant.simulation.ImpedanceType.LINK_FB,
        "n_hat_force_compensation_source": 
            plant.simulation.NHatForceCompensationSource.MEASURED,
        "impedance_stiffness": [4,4,4,40,40,40],
        "num_links": NumLinks.FOUR,
        "timeout": 600,
        "DT": 0.0001,
        "const_ff_Fn": 2,
    }

    k_Js = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 2, 3])
    sweep_runner = sweeps.sweep_infra.SweepRunner(
        other_sim_args=other_sim_args,
        sweep_args=["k_J", "b_J"],
        sweep_vars=np.array([
            k_Js,
            k_Js/10
        ]).T,
    )
    sweep_runner.run_sweep()
