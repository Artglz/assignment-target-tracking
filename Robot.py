import numpy as np

class Robot:
    def __init__(self, sz, id, scene):
        """
        Input:
            sz  - environment size
            id  - ID of the robot
            type- type of the robot, 1 - range and bearing, 2 - range, 3 - bearing
        """
        xy_loc = np.round(
            sz
            * np.random.rand(
                2,
            ),
            1,
        )
        heading = np.round(np.random.rand(1) * 2 * np.pi - np.pi, 4)
        self.location = np.append(xy_loc, heading)
        self.id = id
        self.step_xy = 0.5
        self.step_th = 0.3
        self.cov_list = None
        self.action_num = None
        if scene != 4:
            self.type = scene
        else:
            self.type = np.random.randint(2, 4, 1)[0]
        self.pos_hist = None

    def set_steps(self, num, Nt):
        self.action_num = num
        self.cov_list = np.zeros((self.action_num, Nt))
        return None

    def update_cov(self, new_cov, index):
        self.cov_list[index] = new_cov
        return None

    def update_pos(self, pos_new):
        self.location = pos_new
        return None

    def get_cov(self):
        return self.cov_list

    def get_action(self, u, dt):
        """
        use unicycle model for the robot to perform a
        :param u: linear velocity, and angular
        :param dt: time step
        :return: new position
        """
        v = u[0]
        w = u[1]
        curr_pos = self.location
        # new_x = curr_pos[0] + v * dt * np.cos(curr_pos[2])
        # new_y = curr_pos[1] + v * dt * np.sin(curr_pos[2])
        # new_th = curr_pos[2] + w * dt

        new_th = curr_pos[2] + w * dt
        new_x = curr_pos[0] + v * dt * np.cos(new_th)
        new_y = curr_pos[1] + v * dt * np.sin(new_th)
        return np.array([new_x, new_y, new_th])

    def __str__(self):
        return f"robot {self.id} has location {self.location}"
