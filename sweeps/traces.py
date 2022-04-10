import sweeps.sweep_infra
import numpy as np
import plant.simulation
import config

def proc_func(sim, log):
    
    out = {}

    non_vertical_axis = 1 - config.hinge_rotation_axis

    # Get paper link data
    horizontal_paper_traces = []
    vertical_paper_traces = []
    angle_paper_traces = []
    for b in sim.paper.link_idxs:
        horizontal_paper_traces.append(log.data()[
            sim.log_wrapper.get_idx("pos", "trn", b) + non_vertical_axis])
        vertical_paper_traces.append(log.data()[
            sim.log_wrapper.get_idx("pos", "trn", b) + 2])

        ang_trace = np.array(log.data()[
            sim.log_wrapper.get_idx("pos", "rot", b) + \
                config.hinge_rotation_axis])
        # Handle RPY stuff
        if config.hinge_rotation_axis == 1:
            z_angs = log.data()[sim.log_wrapper.get_idx("pos", "rot", b) + 2]
            ang_trace[z_angs > np.pi/2] = ang_trace[z_angs > np.pi/2]*-1 + np.pi
            ang_trace = ang_trace *-1
        angle_paper_traces.append(ang_trace)

    out["horizontal_paper_traces"] = np.array(horizontal_paper_traces)
    out["vertical_paper_traces"] = np.array(vertical_paper_traces)
    out["angle_paper_traces"] = np.array(angle_paper_traces)

    # Get times
    out["first_contact_idx"] = np.argmax(
        log.data()[sim.log_wrapper.calc_in_contact_start_idx])
    out["times"] = log.sample_times()[:-1]

    # Get manipulator traces
    out["horizontal_manipulator_trace"] = log.data()[sim.log_wrapper.get_idx(
            "pos", "trn", sim.contact_body_idx) + non_vertical_axis,:]
    out["vertical_manipulator_trace"] = log.data()[sim.log_wrapper.get_idx(
            "pos", "trn", sim.contact_body_idx) + 2,:]
    return out


if __name__ == "__main__":
    other_sim_args = {
        "ctrl_paradigm": plant.simulation.CtrlParadigm.IMPEDANCE,
        "impedance_type": plant.simulation.ImpedanceType.LINK_FB,
        "n_hat_force_compensation_source": 
            plant.simulation.NHatForceCompensationSource.CONSTANT,
        "impedance_stiffness": [4,4,4,40,40,40],
        "num_links": config.NumLinks.FOUR
    }

    sweep_runner = sweeps.sweep_infra.SweepRunner(
        proc_func=proc_func,
        other_sim_args=other_sim_args,
        sweep_args=["impedance_type", "n_hat_force_compensation_source"],
        sweep_vars=np.array([
            [
                plant.simulation.ImpedanceType.OFFLINE_TRAJ,
                plant.simulation.ImpedanceType.LINK_FB,
                plant.simulation.ImpedanceType.LINK_FB,
            ],
            [
                plant.simulation.NHatForceCompensationSource.NONE,
                plant.simulation.NHatForceCompensationSource.CONSTANT,
                plant.simulation.NHatForceCompensationSource.MEASURED,
            ],
        ]).T
    )
    sweep_runner.run_sweep()
