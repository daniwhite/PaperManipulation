import sweeps.sweep_infra
import numpy as np
import plant.simulation
import config


if __name__ == "__main__":
    other_sim_args = {
        "ctrl_paradigm": plant.simulation.CtrlParadigm.IMPEDANCE,
        "impedance_stiffness": [4,4,4,40,40,40],
        "num_links": config.NumLinks.FOUR,
        "timeout": 600,
        "DT": 0.0001
    }

    sweep_runner = sweeps.sweep_infra.SweepRunner(
        other_sim_args=other_sim_args,
        sweep_args=[
            "impedance_type",
            "n_hat_force_compensation_source",
            "const_ff_Fn"
        ],
        sweep_vars=np.array([
            [
                plant.simulation.ImpedanceType.OFFLINE_TRAJ,
                plant.simulation.ImpedanceType.LINK_FB,
                plant.simulation.ImpedanceType.FORCE_FB,
                plant.simulation.ImpedanceType.LINK_FB,
            ],
            [
                plant.simulation.NHatForceCompensationSource.NONE,
                plant.simulation.NHatForceCompensationSource.CONSTANT,
                plant.simulation.NHatForceCompensationSource.CONTACT_FORCE,
                plant.simulation.NHatForceCompensationSource.PURE_FN,
            ],
            [0, 15, 5, 5],
        ]).T
    )
    sweep_runner.run_sweep()
