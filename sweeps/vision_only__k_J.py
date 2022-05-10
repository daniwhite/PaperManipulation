import sweeps.sweep_infra
import numpy as np
import plant.simulation
from config import NumLinks

if __name__ == "__main__":
    other_sim_args = {
        "ctrl_paradigm": plant.simulation.CtrlParadigm.IMPEDANCE,
        "impedance_type": plant.simulation.ImpedanceType.LINK_FB,
        "n_hat_force_compensation_source": 
            plant.simulation.NHatForceCompensationSource.CONSTANT,
        "impedance_stiffness": [4,4,4,40,40,40],
        "num_links": NumLinks.FOUR,
        "timeout": 600,
        "DT": 0.0001,
        "const_ff_Fn": 15,
    }

    k_Js = np.concatenate((np.arange(0.01, 0.1, 0.01), np.arange(0.1, 1, 0.1), [1, 1.2, 1.4, 1.6, 2, 3]))
    sweep_runner = sweeps.sweep_infra.SweepRunner(
        other_sim_args=other_sim_args,
        sweep_args=["k_J", "b_J"],
        sweep_vars=np.array([
            k_Js,
            k_Js/10
        ]).T,
    )
    sweep_runner.run_sweep()
