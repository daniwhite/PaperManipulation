import sweeps.sweep_infra
import numpy as np
import plant.simulation
import config


if __name__ == "__main__":
    other_sim_args = {
        "ctrl_paradigm": plant.simulation.CtrlParadigm.IMPEDANCE,
        "impedance_type": plant.simulation.ImpedanceType.LINK_FB,
        "n_hat_force_compensation_source": 
            plant.simulation.NHatForceCompensationSource.CONSTANT,
        "num_links": config.NumLinks.FOUR,
        "timeout": 1200,
        "DT": 0.0001
    }

    sweep_runner = sweeps.sweep_infra.SweepRunner(
        other_sim_args=other_sim_args,
        sweep_args=[
            "impedance_type",
            "n_hat_force_compensation_source",
            "impedance_stiffness",
            "const_ff_Fn"
        ],
        sweep_vars=np.array([
            [
                plant.simulation.ImpedanceType.OFFLINE_TRAJ,
                plant.simulation.ImpedanceType.LINK_FB,
                plant.simulation.ImpedanceType.LINK_FB,
            ],
            [
                plant.simulation.NHatForceCompensationSource.NONE,
                plant.simulation.NHatForceCompensationSource.CONSTANT,
                plant.simulation.NHatForceCompensationSource.PURE_FN,
            ],
            [
                [40,40,40,400,400,400],
                [4,4,4,40,40,40],
                [4,4,4,40,40,40]
            ],
            [0, 50, 5],
        ]).T
    )
    sweep_runner.run_sweep()
