import sweeps.sweep_infra
import numpy as np
import plant.simulation
import config

if __name__ == "__main__":
    other_sim_args = {
        "ctrl_paradigm": plant.simulation.CtrlParadigm.IMPEDANCE,
        "impedance_type": plant.simulation.ImpedanceType.LINK_FB,
        "n_hat_force_compensation_source": 
            plant.simulation.NHatForceCompensationSource.MEASURED,
        "impedance_stiffness": [4,4,4,40,40,40],
        "num_links": config.NumLinks.FOUR,
        "timeout": 600,
        "DT": 0.0001,
        "const_ff_Fn": 5
    }
    num_samples = 30
    noise = np.repeat(np.linspace(0,100,16), num_samples)
    sweep_runner = sweeps.sweep_infra.SweepRunner(
        other_sim_args=other_sim_args,
        sweep_args="noise__Fn",
        sweep_vars=noise,
        other_data_attrs={"num_samples": num_samples}
    )
    sweep_runner.run_sweep()
