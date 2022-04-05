import numpy as np
import scipy.interpolate

import constants
import plant.manipulator
import config
import plant.pedestal

# Drake imports
import pydrake
from pydrake.all import Meshcat, RigidTransform, RollPitchYaw, RotationMatrix

class OfflineTrajLoader(pydrake.systems.framework.LeafSystem):
    """
    Load a Cartesian endpoint trajectory and interpolate it through time.
    """
    def __init__(self,  num_links: config.NumLinks, speed_factor=1):
        pydrake.systems.framework.LeafSystem.__init__(self)
        # System constants/parameters

        # traj_npz = np.load(config.base_path + \
        #     "x0s_sim_{}_links.npz".format(num_links.value))
        traj_npz = np.load(config.base_path + "x0s_sim.npz")
        self.t = traj_npz['ts']
        self.x0s = traj_npz['poses']
        self._calc_x0 = scipy.interpolate.interp1d(self.t, self.x0s)

        # We want to generate the trajectory for the Panda in the real world,
        # but we want to run it much more quickly often in sim.
        self.speed_factor = speed_factor

        # =========================== DECLARE INPUTS ==========================
        # No inputs

        # ========================== DECLARE OUTPUTS ==========================
        # Impedance setpoint
        self.DeclareVectorOutputPort(
            "x0",
            pydrake.systems.framework.BasicVector(6),
            self.calc_x0
        )
        self.DeclareVectorOutputPort(
            "dx0",
            pydrake.systems.framework.BasicVector(6),
            self.calc_dx0
        )


    def calc_x0(self, context, output):
        x0 = self._calc_x0(context.get_time()*self.speed_factor)
        output.SetFromVector(x0)


    def calc_dx0(self, context, output):
        output.SetFromVector(np.zeros(6))
