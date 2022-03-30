import sweeps.sweep_infra
import numpy as np
import plant.simulation

if __name__ == "__main__":
    imps = [
        [ 4, 40,  4,  40,  40,   40],
        [ 4,  4, 40,  40,  40,   40],
        [40,  4,  4,  40,  40,   40],
        [40,  4,  40,  40,  40,  40],
        [ 4, 40,  4, 400, 400,  400],
        [ 4,  4, 40, 400, 400,  400],
        [40,  4,  4, 400, 400,  400],
        [40,  4,  40, 400, 400, 400],
    ]

    other_sim_args = {
        "ctrl_paradigm": plant.simulation.CtrlParadigm.IMPEDANCE,
        "impedance_type": plant.simulation.ImpedanceType.LINK_FB,
        "n_hat_force_compensation_source": 
            plant.simulation.NHatForceCompensationSource.MEASURED,
    }
    sweep_runner = sweeps.sweep_infra.SweepRunner(
        proc_func=sweeps.sweep_infra.get_max_theta_L,
        other_sim_args=other_sim_args,
        sweep_arg="impedance_stiffness",
        sweep_vars=imps,
    )
    sweep_runner.run_sweep()
