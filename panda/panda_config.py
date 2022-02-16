from plant.manipulator import X_W_panda
from pydrake.all import RigidTransform

# Transform from SP in simulation to setpoint in hardware
# Basically, need to move from world frame to Panda frame
def X_panda_SP_hw(X_W_SP_sw):
    X_panda_SP_sw = X_W_panda.inverse().multiply(X_W_SP_sw)
    hardware_offset = RigidTransform()
    X_panda_SP_hw = X_panda_SP_sw.multiply(hardware_offset)
    return X_panda_SP_hw

