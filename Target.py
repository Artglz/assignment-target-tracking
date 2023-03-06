import numpy as np
from Sensor import pos_estimate

class Target:  # TODO: Double integration motion model -- Brent
    def __init__(self, sz, id, perb):
        sig = 2
        self.perb = perb
        self.ground_truth = np.round(
            sz
            * np.random.rand(
                2,
            ),
            1,
        )
        self.id = id
        self.estimated_location = pos_estimate(self.perb, self.ground_truth)
        self.sigma = np.eye(2) * sig
        self.sensor_measurement = None
        self.sensor_assigned = []
        self.r_set = []

    def get_action(self, u, dt, x_adj, y_adj):
        v = u[0]
        curr_pos = self.ground_truth
        new_x = curr_pos[0] + v * dt * x_adj
        new_y = curr_pos[1] + v * dt * y_adj
        new_pos = np.array([new_x, new_y])
        return new_pos

    def update(self, pos_new, sig_new):
        self.ground_truth = pos_new
        self.estimated_location = pos_estimate(self.perb, self.ground_truth)
        self.sigma = sig_new
        return None

    def __str__(self):
        return f"Target {self.id} has ground truth location {self.ground_truth}, and estimated location {self.estimated_location}"
