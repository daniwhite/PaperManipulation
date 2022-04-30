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
        "impedance_stiffness": [4,4,4,40,40,40],
        "num_links": config.NumLinks.FOUR,
        "timeout": 600,
        "DT": 0.0001,
        "const_ff_Fn": 40
    }
    num_samples = 30
    pose_L_translational_noise = np.repeat(list(10**np.linspace(-3, np.log10(3), 16)), num_samples)
    pose_L_rotational_noise    = pose_L_translational_noise/(1e-2)
    sweep_runner = sweeps.sweep_infra.SweepRunner(
        other_sim_args=other_sim_args,
        sweep_args=[
            "noise__pose_L_rotational",
            "noise__pose_L_translational",
            "noise__vel_L_rotational",
            "noise__vel_L_translational"],
        sweep_vars=np.array([
            pose_L_translational_noise,
            pose_L_rotational_noise,
            pose_L_translational_noise,
            pose_L_rotational_noise,
        ]).T,
        other_data_attrs={"num_samples": num_samples}
    )
    sweep_runner.run_sweep()
