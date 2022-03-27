from multiprocessing import Pool

import plant.simulation
import time
import numpy as np
from config import hinge_rotation_axis

# Numpy, Scipy, Matplotlib
import scipy.interpolate
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import numpy as np

# Drake imports
import pydrake
from pydrake.all import (
    DirectCollocation, DirectTranscription, MathematicalProgram,
    InputPortSelection, LogVectorOutput
)
from pydrake.all import FindResourceOrThrow
from pydrake.all import (MultibodyPlant, Parser, DiagramBuilder, Simulator, RigidTransform,
                         PlanarSceneGraphVisualizer, SceneGraph, TrajectorySource,
                         SnoptSolver, MultibodyPositionToGeometryPose, PiecewisePolynomial,
                         MathematicalProgram, JacobianWrtVariable, eq, RollPitchYaw, AutoDiffXd, BodyIndex,
                        RotationMatrix, Meshcat,MeshcatVisualizerParams, MeshcatVisualizerCpp, MeshcatVisualizer,
                        Adder, Gain, ConstantVectorSource, Demultiplexer, Multiplexer, AngleAxis)

# Other imports
import importlib
import re
import enum
from IPython.display import display, SVG, Image
import pydot

# Imports of other project files
import constants
import config
from config import hinge_rotation_axis

import plant.simulation
import plant.manipulator as manipulator


import ctrl.aux
import time

def pool_func_1(N_constant_ff_F):
    ctrl_paradigm = plant.simulation.CtrlParadigm.IMPEDANCE
    impedance_type = plant.simulation.ImpedanceType.LINK_FB
    n_hat_force_compensation_source = plant.simulation.NHatForceCompensationSource.CONSTANT
    sim = plant.simulation.Simulation(
        ctrl_paradigm=ctrl_paradigm,
        impedance_type=impedance_type,
        n_hat_force_compensation_source=n_hat_force_compensation_source,
        exit_when_folded=True,
        N_constant_ff_F=N_constant_ff_F,
        timeout=180
    )
    print(sim.N_constant_ff_F)

    # Run sim
    t_start_ = time.time()
    log = sim.run_sim()
    print(time.time() - t_start_)
    
    # Grab output var
    theta_L = log.data()[sim.log_wrapper.get_idx("pos", "rot", sim.ll_idx) + hinge_rotation_axis].copy()
    theta_LZ = log.data()[sim.log_wrapper.get_idx("pos", "rot", sim.ll_idx)+2]
    # Fix issue in RPY singularity
    theta_L[theta_LZ > np.pi/2] = theta_L[theta_LZ > np.pi/2]*-1 + np.pi
    
    return np.max(theta_L)

if __name__ == '__main__':
    t_start__ = time.time()
    with Pool(8) as p:
        print(p.map(pool_func_1, np.linspace(0, 10, 8)))
    print(time.time() - t_start__)
