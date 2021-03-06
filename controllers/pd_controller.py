# Standard imports
import constants
import finger
import numpy as np


class PDFinger(finger.FingerController):
    """Set up PD controller for finger."""

    def __init__(self, plant, finger_idx, pts, tspan_per_segment=5, ky=5, kz=5, dy=0.01, dz=0.01):
        super().__init__(plant, finger_idx)

        # Generate trajectory
        self.ys = []
        self.zs = []
        self.ydots = [0]
        self.zdots = [0]

        for segment_i in range(len(pts)-1):
            start_pt = pts[segment_i]
            end_pt = pts[segment_i+1]
            for prog_frac in np.arange(0, tspan_per_segment, constants.DT)/tspan_per_segment:
                new_y = start_pt[0] + (end_pt[0] - start_pt[0])*prog_frac
                self.ys.append(new_y)
                new_z = start_pt[1] + (end_pt[1] - start_pt[1])*prog_frac
                self.zs.append(new_z)

        for i in range(len(self.ys)-1):
            self.ydots.append((self.ys[i+1]-self.ys[i])/constants.DT)
            self.zdots.append((self.zs[i+1]-self.zs[i])/constants.DT)

        # Set gains
        self.ky = ky
        self.kz = kz
        self.dy = dy
        self.dz = dz

        # For keeping track of place in trajectory
        self.idx = 0

    def GetForces(self, poses, vels, contact_point):
        # Unpack values
        y = poses[self.finger_idx].translation()[1]
        z = poses[self.finger_idx].translation()[2]
        ydot = vels[self.finger_idx].translational()[1]
        zdot = vels[self.finger_idx].translational()[2]

        if self.idx < len(self.ys):
            fy = self.ky*(self.ys[self.idx] - y) + \
                self.dy*(self.ydots[self.idx] - ydot)
            fz = self.kz*(self.zs[self.idx] - z) + \
                self.dz*(self.zdots[self.idx] - zdot)
        else:
            fy = self.ky*(self.ys[-1] - y) + self.dy*(-ydot)
            fz = self.kz*(self.zs[-1] - z) + self.dz*(- zdot)
        self.idx += 1

        return fy, fz
