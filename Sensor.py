import random

from munkres import Munkres
import numpy as np
from itertools import permutations, product, combinations
import time
import matplotlib.pyplot as plt
import matplotlib
import csv
from Robot import Robot

matplotlib.rcParams['pdf.fonttype'] = 42


def find_error(d_true, d_est):
    """
    This function finds the Euclidean distance between the estimated position and the true position
    Parameters
    ----------
    d_true: Ground truth position, 2D array
    d_est:  Estimated position, 2D array

    Returns
    -------
    d_error: The Euclidean distance, scalar.
    """
    return np.linalg.norm(d_true - d_est)


def wrapToPi(angleIn):
    if angleIn > np.pi or angleIn < -np.pi:
        angleOut = angleIn - 2 * np.pi * (angleIn % 2 * np.pi)
    else:
        angleOut = angleIn
    return angleOut


def euler_angle(pose):
    """
    This function transform a given position to euler angle relative to the origin
    :param pose: position [x, y], can be an N x 2 array for multiple inputs
    :return:
    """
    if pose.shape[0] == 1:
        angle = np.arctan2(pose[1], pose[0])
    else:
        angle = np.arctan2(pose[:, 1], pose[:, 0])
    return angle


def pos_estimate(perb, ground_truth):
    return ground_truth + (2 * perb * np.random.rand(1) - perb)


class SensorAssignment:
    def __init__(self, Nt, Nr, sz, scene):
        """
        Initialize the sensor assignment code, randomly place the sensor and targets based on the input number.
        The placement is within the size input.
        :param Nt: Number of target
        :param Nr: Number of robot (for bearing and ranging, Nr = Nt)
        :param sz: Size of the environment
        :return: None
        """
        self.Nt = Nt
        self.scene = scene
        if scene != 1:
            self.adjust = 2
        else:
            self.adjust = 1
        self.Nr = self.adjust * Nt
        self.dt = 0.5  # set the time step
        perb = 0.3
        self.r_sen = (
                sz * 100
        )  # assume the sensing range is far larger than the size of the world
        from Target import Target
        self.targets = [Target(sz, i, perb) for i in range(self.Nt)]
        self.targets_greedy = self.targets.copy()

        self.robots = [Robot(sz, i, scene) for i in range(self.Nr)]
        self.robots_comb = None
        if self.scene == 1:
            self.robots_comb = self.robots
            self.robots_list = np.array([i for i in range(self.Nr)])
        else:
            # total different robots combination (group)
            self.robots_list = np.array(
                tuple(combinations([ii for ii in range(self.Nr)], 2))
            )
            # transfer to robot objects
            self.robots_comb = [
                [self.robots[ii[0]], self.robots[ii[1]]] for ii in self.robots_list
            ]

        self.target_pos_sig = None
        self.target_true_pos = None
        self.target_pos_hat = None

        self.targetVals()
        self.robotVals()

        self.noise = 0.2 * np.random.rand(200, 1)

        self.robot_pos = None
        self.v = None  # linear velocity list
        self.w = None  # angular velocity list
        self.actions = None  # action combinations for single robot
        self.action_num = None
        self.record = None
        self.action_for_each_robot = None
        self.action_combs = None  # action combination across all the robot
        self.sensor_target = None  # sensor and target pair
        self.cov_matrix = None
        self.sig_matrix = None
        self.target_est = None

    #this method is calculating and storing 
    #important information about a set of targets, including their true positions, estimated positions, and sigma values.
    def targetVals(self):
        self.target_true_pos = np.array(
            [self.targets[i].ground_truth for i in range(self.Nt)]
        )
        self.target_pos_hat = np.array(
            [self.targets[i].estimated_location for i in range(self.Nt)]
        )
        self.target_pos_sig = np.array(
            [[self.targets[i].sigma] for i in range(self.Nt)]
        ).reshape((2 * self.Nt, 2))

    # get the estimated position of all the robots and stores it in a numpy array
    def robotVals(self):
        self.robot_pos = np.array([self.robots[i].location for i in range(self.Nr)])

    def set_actions(self, v, w):
        """
        Set the linear velocity and angular velocity, and then get the combinations for the actions
        """

        self.v = v
        self.w = w
        self.actions = np.array([[ii, jj] for ii in self.v for jj in self.w]) # get all the combinations of actions
        self.action_num = len(self.v) * len(self.w) # get the number of possible actions
        for i in range(self.Nr):
            self.robots[i].set_steps(self.action_num, self.Nt) # ihitialize a cov_list for each robot with zeros

    def all_pairs(self, lst):# only used in scene 2
        """
        Credit to shang from StackOverflow
        https://stackoverflow.com/questions/5360220/how-to-split-a-list-into-pairs-in-all-possible-ways
        :param lst:
        :return:
        """
        if len(lst) < 2:
            yield []
            return
        if len(lst) % 2 == 1:
            # Handle odd length list
            for i in range(len(lst)):
                for result in self.all_pairs(lst[:i] + lst[i + 1:]):
                    yield result
        else:
            a = lst[0]
            for i in range(1, len(lst)):
                pair = [a, lst[i]]
                # print(lst[1:i], lst[i + 1 :])
                for rest in self.all_pairs(lst[1:i] + lst[i + 1:]):
                    yield [pair] + rest

    def step(self):
        """
        Perform a step operation for all the robots, the robot will perform all the actions and estimate all the targets
        using EKF for each action.
        """
        self.targetVals()
        self.robotVals()
        if self.scene != 1:
            action_comb = [[u1, u2] for u1 in self.actions for u2 in self.actions]
        else:
            action_comb = self.actions
        self.action_for_each_robot = action_comb
        # storing the covariance matrix, sigma matrix, and target estimates, respectively, for each robot-action combination.
        self.cov_matrix = np.zeros((len(self.robots_comb), len(action_comb), self.Nt))
        self.sig_matrix = np.zeros(
            (len(self.robots_comb), len(action_comb), 2 * self.Nt, 2)
        )
        self.target_est = np.zeros(
            (len(self.robots_comb), len(action_comb), self.Nt, 2)
        )
        # obtained from targetVals()
        tPos_hat = self.target_pos_hat # estimated target position
        tSigma_hat = self.target_pos_sig # estimated target covariance
        tPos_true = self.target_true_pos # true target position
        
        # loop through all the robots
        for i in range(len(self.robots)):
            # loop through all the actions combinations for each robot
            for j in range(len(action_comb)):
                u = action_comb[j]
                # print(u)
                if self.scene == 1:
                    rPos = self.robots[i].get_action(u, self.dt).reshape((1, 3))
                    robot_curr = self.robots[i]
                else:
                    robot_curr = []
                    rPos = np.zeros((len(self.robots_comb[i]), 3))
                    for k in range(len(self.robots_comb[0])):
                        rPos[k] = (
                            self.robots_comb[i][k]
                                .get_action(u[k], self.dt)
                                .reshape((1, 3))
                        )
                        robot_curr.append(self.robots_comb[i][k])
                # print(len(rPos))
                
                # call the EKF function for this action-robot combination
                # if there are 3 robots and 3 actions, there will be 27 (9 action-combinations * 3 robots ) EKF calls
                (
                    quality_ts,
                    trace_ts,
                    sqerr_ts,
                    tPos_hat_kp1, # estimated target position at time k+1
                    tSigma_hat_kp1, # estimated target covariance at time k+1
                    trace_t,
                    sqerr_t,
                    trace_sig_diff, # trace of the difference between the covariance matrix at time k+1 and time k
                ) = self.EKF(
                    tPos_hat, tSigma_hat, tPos_true, 1, rPos, robot_curr
                )  # r_set = 1
                # print(f'robot {self.robots[i].id} with action {j} has {trace_t}')

                # print(j)
                # print(f'robot {[self.robots_comb[i][0].type, self.robots_comb[i][1].type]} and action {u}, pos {rPos} with trace {trace_sig_diff}')
                # print(f"action {u}, pos {rPos} with trace {trace_sig_diff}")
                # print(trace_sig_diff)

                # updates the covariance matrix, sigma matrix, and target estimate for the current robot-action combination.
                # i is the current robot, j is the current action
                if self.scene == 1:
                    self.robots[i].update_cov(trace_sig_diff, j) # update the covariance matrix for the current robot-action combination at index j
                self.cov_matrix[i][j] = trace_sig_diff # covariance matrix
                self.sig_matrix[i][j] = tSigma_hat_kp1 # sigma matrix
                self.target_est[i][j] = tPos_hat_kp1 # target estimate

        # print(self.action_num)
        # print(len(action_comb))
        # print(f"cov_matrix has shape of {self.cov_matrix.shape}")

    def OPT(self):
        # combination of all the actions
        if self.scene == 1:
            action_num = self.action_num
            # all the possible action combinations(idk why its different from the above one)
            self.action_combs = np.array(
                tuple(product(range(action_num), repeat=self.Nt))
            )
            # print(self.action_num)
            # combination of all the robot(sensor) and target pairs
            self.sensor_target = np.array(
                tuple(permutations(range(len(self.robots_comb)), r=self.Nt))
            )

        else:
            action_num = self.action_num ** 2
            all_pair = np.array(tuple(self.all_pairs(self.robots)))
            # all_pair = np.array(tuple(self.all_pairs([i for i in range(self.Nr)])))
            all_sensor_target_pair = np.array(
                [tuple(permutations(pair, r=self.Nt)) for pair in all_pair]
            )
            all_st_pair_shape = all_sensor_target_pair.shape
            all_st_pair_reshape = all_sensor_target_pair.reshape(
                (
                    all_st_pair_shape[0] * all_st_pair_shape[1],
                    all_st_pair_shape[2],
                    all_st_pair_shape[3],
                )
            )
            all_st_pair_id = [
                [[kk.id for kk in jj] for jj in ii] for ii in all_st_pair_reshape
            ]
            self.sensor_target = all_st_pair_id
            self.action_combs = np.array(
                tuple(product(range(action_num), repeat=self.Nt))
            )
            # print(action_num)
            # print(f"robots_comb has length = {len(self.robots_comb)}")
            # print("a")
            # print(
            #     f"robots pairs = {len(self.robots_comb)}, robot-target pairs = {len(self.sensor_target)}"
            # )
            # initialize the tracker of sum of trace
            # print(self.action_combs)
            # print(len(self.action_combs))
            # print(all_pair.shape)
            # print(all_st_pair_reshape.shape)

        # initialize the tracker of sum of trace
        # basically make a matrix of zeros with the size of the action combinations and sensor-target pairs
        self.record = np.zeros((len(self.action_combs), len(self.sensor_target)))
        # print(all_pair)
        # print(all_pair)

        # [v1w1, v1w2, v1w3, v2w1, v2w2, v2w3, v3w1, v3w2, v3w3]
        for ii in range(len(self.action_combs)):
            action_comb = self.action_combs[ii]
            # [robo1_action1, robo2_action1, robo3_action1]...
            for jj in range(len(self.sensor_target)):
                pair = self.sensor_target[jj]
                # [robo1_target, robo2_target, robo3_target]...
                for rr in range(self.Nt):
                    robots_ids = self.robots_list[rr]
                    if self.scene == 1:
                        # calling the get_cov method on the robot combination at index rr, and indexing into the result with action_comb[rr] and pair[rr]
                        self.record[ii][jj] += self.robots_comb[rr].get_cov()[action_comb[rr]][pair[rr]]
                    else:
                        robot_loc = np.where(
                            np.all(self.robots_list == pair[rr], axis=1)
                        )
                        action_loc = action_comb[rr]
                        # print(self.cov_matrix)
                        # print('---')
                        # print(self.cov_matrix[robot_loc][0])
                        # print('---')
                        # print(self.cov_matrix[robot_loc][0][action_loc])
                        # print('---')
                        # print(self.cov_matrix[robot_loc][0][action_loc][rr])
                        # print(self.record[ii][jj])
                        self.record[ii][jj] += self.cov_matrix[robot_loc][0][
                            action_loc
                        ][rr]
                        # for kk in range(len())
                        # print(f'Action: {self.action_combs[ii]}, ST pair: {self.sensor_target[jj]}, robot: {self.robots_comb[rr]}')
        
        max_val = np.amax(self.record) # get the maximum value in the record matrix
        max_loc = np.where(self.record == max_val) # get the location of the maximum value
        action_ind = max_loc[0][0]
        pair_ind = max_loc[1][0]
        # print(f"max value: {max_val:0.3f}")
        # print(f"action index is {action_ind}, which is {self.action_combs[action_ind]}")
        # print(f"pair index is {pair_ind}, which is {self.sensor_target[pair_ind]}")
        # print(
        #     f"total iteration is {len(self.action_combs) * len(self.sensor_target) * self.Nt}"
        # )

        return max_val  # , max_loc

    def Greedy(self):
        if self.scene == 1:
            # copy all the list of target objects idk why, maybe to not deal with ownership issues
            targets_greedy = self.targets.copy()
            target_poped = []
            # copy all the list of robot objects idk why, maybe to not deal with ownership issues
            robots_greedy = self.robots.copy()
            # this is the matrix that will store the robot, action, and target, bascically the most important part
            robot_action_target = np.zeros((len(robots_greedy), 2), dtype=int) 
            robot = []
            while bool(robots_greedy): # while there are still robots
                if bool(targets_greedy): # if there are still targets
                    record_list = np.zeros(
                        (len(robots_greedy), self.action_num, len(targets_greedy))
                    ) # making a matrix of zeros with the size of the robots, actions, and targets to store the trace/covariance
                    # i.e if i have 3 robots, 3 actions, and 3 targets, I will have 3 matrices of 9 rows and 3 columns (9x3)

                    # assemble all the trace for all the robot, action, for all targets
                    for ii in range(len(robots_greedy)):
                        cov_matrix = robots_greedy[ii].get_cov()
                        if bool(target_poped):
                            cov_matrix = np.delete(cov_matrix, target_poped, 1)
                        record_list[ii] = cov_matrix

                    # find the max one and the location
                    max_val = np.amax(record_list)
                    max_loc = np.where(record_list == max_val)

                    # keep record of the action and target assigned to the robot
                    robot_pop = robots_greedy[max_loc[0][0]].id # get the robot id that has the max value
                    max_action = max_loc[1][0] # get the action that has the max value
                    target_pop = targets_greedy[max_loc[2][0]].id # get the target id that has the max value
                    target_poped.append(target_pop) # append the target id to the target_poped list
                    robot_action_target[robot_pop][0] = max_action # assign the action to the robot
                    robot_action_target[robot_pop][1] = target_pop # assign the target to the robot
                    robot.append(robot_pop) # append the robot id to the robot list
                    # pop the robot and target selected in this round
                    # so they cant be chosen again
                    targets_greedy.pop(max_loc[2][0])
                    robots_greedy.pop(max_loc[0][0])
                    # print(f'robot {robot_pop} with action {max_action} assigned to target {target_pop}')

            actions = robot_action_target[:, 0] # get all the actions, first column
            pairs = robot_action_target[:, 1] # get all the targets, second column
            max_cov = sum(
                [
                    self.robots[ii].get_cov()[robot_action_target[ii][0]][
                        robot_action_target[ii][1]
                    ]
                    for ii in range(self.Nr)
                ]
            ) # retrieves the covariance value from the corresponding 
            # robot's covariance matrix using the action and target indices stored in robot_action_target
            # the goal is to reduce the covariance, so the lower the better

        else:
            targets_greedy = self.targets.copy()
            target_popped = []
            if self.scene == 1:
                robots_greedy = self.robots.copy()
            else:
                robots_greedy = self.robots_list.copy()
                robots_greedy = robots_greedy.tolist()
            robot_action_target = np.zeros((self.Nt, 3), dtype=int)
            covariance_matrix = self.cov_matrix.copy()
            # robot_action_target = np.zeros((self.Nt, 2), dtype=int)
            iter = 0
            max_cov = 0
            while bool(robots_greedy):
                if bool(targets_greedy):

                    record_list = np.zeros(
                        (
                            len(robots_greedy),
                            len(self.action_for_each_robot),
                            len(targets_greedy),
                        )
                    )

                    # assemble all the trace for all the robot, action, for all targets
                    for ii in range(len(robots_greedy)):
                        cov_matrix = covariance_matrix[ii]
                        # if bool(target_popped):
                        #     cov_matrix = np.delete(cov_matrix, target_popped, 1)
                        record_list[ii] = cov_matrix

                    # find the max one and the location
                    max_val = np.amax(record_list)
                    max_loc = np.array(np.where(record_list == max_val))
                    max_cov += max_val

                    # keep record of the action and target assigned to the robot
                    # robots_max_greedy = robots
                    robot_pop = np.array(
                        np.where(
                            np.all(
                                self.robots_list == robots_greedy[max_loc[0][0]], axis=1
                            )
                        )
                    ).reshape((1,))
                    # robot_pop = robots_greedy[max_loc[0][0]].id
                    max_action = max_loc[1][0]
                    target_pop = targets_greedy[max_loc[2][0]].id
                    target_popped.append(target_pop)
                    robot_action_target[iter][0] = max_action
                    robot_action_target[iter][1] = target_pop
                    robot_action_target[iter][2] = robot_pop

                    # pop the robot and target selected in this round
                    targets_greedy.pop(max_loc[2][0])
                    robots_greedy = np.array(robots_greedy)
                    # slice = np.where(np.any(robots_greedy == self.robots_list[robot_pop], axis=1))
                    # something = self.robots_list[robot_pop]
                    place_to_delete = np.where(
                        np.any(
                            robots_greedy
                            == np.array(self.robots_list[robot_pop][0][0]),
                            axis=1,
                        )
                    ) + np.where(
                        np.any(
                            robots_greedy
                            == np.array(self.robots_list[robot_pop][0][1]),
                            axis=1,
                        )
                    )
                    robots_greedy = np.delete(
                        robots_greedy,
                        place_to_delete,
                        axis=0,
                    )
                    covariance_matrix = np.delete(
                        covariance_matrix, place_to_delete, axis=0
                    )
                    covariance_matrix = np.delete(
                        covariance_matrix, max_loc[2][0], axis=2
                    )
                    robots_greedy = robots_greedy.tolist()
                    iter += 1

                    # print(f'robot {robot_pop} with action {max_action} assigned to target {target_pop}')
            # print('no problem fk')
            actions = robot_action_target[:, 0]
            pairs = robot_action_target[:, 1]
            robot = robot_action_target[:, 2]

        return (max_cov, robot, actions, pairs)

    def Hungarian(self):
        """
        Using Hungarian method to get the perfect choice
        """
        # initialize the hungarian method solver
        m = Munkres()

        # reshape the covariance matrix, combine the robots pairs and actions
        # the reshaped matrix would be a 2D matrix with row being robots pairs at each actions
        # col being the targets
        cov_matrix_hun = np.copy(self.cov_matrix)
        cov_hun_shape = cov_matrix_hun.shape
        cov_hun_row = cov_hun_shape[0] * cov_hun_shape[1]

        cov_matrix_hun = cov_matrix_hun.reshape((cov_hun_row, cov_hun_shape[2]))
        # print(cov_matrix_hun.shape)
        # print(cov_matrix_hun)

        # padding with 0 to get a square matrix for Hungarian Method
        profit_mat = np.zeros((cov_hun_row, cov_hun_row))
        profit_mat[:, 0: cov_hun_shape[2]] = cov_matrix_hun
        profit_mat_trans = profit_mat.T
        max_element = np.max(profit_mat_trans)
        cost_mat = max_element - profit_mat_trans
        indexes = m.compute(cost_mat)
        # get the first Nt indexes
        index_first_Nt = indexes[0: self.Nt]
        max_cov = 0
        comb_action_lst = []
        for row, col in index_first_Nt:
            max_cov += profit_mat_trans[row][col]
            comb = row // len(self.action_for_each_robot)
            action = row % len(self.action_for_each_robot)
            comb_action_lst += [comb, action]

        # print(f"max value: {max_cov: 0.3f}")  # {max_cov: 0.3f}")
        return max_cov
        # print(f"actions are {}")

    def EKF(self, tPos_hat_k, tSigma_hat_k, tPos_true_k, r_set, rPos_kp1, robot_curr):
        """

        Extended Kalman Filter for range and bearing sensors (1 to 1 assignment)
        :param tPos_hat_k:      estimated target position
        :param tSigma_hat_k:    estimated target covariance
        :param tPos_true_k:     true target position
        :param r_set:           number of robot
        :param rPos_kp1:        position of robot
        :return:
        """
        # kp1 --> k+1
        # initialize tPos and tSigma for the k+1 step
        tPos_hat_kp1 = np.zeros([self.Nt, 2])
        tSigma_hat_kp1 = np.zeros([self.Nt * 2, 2])

        # Simulate measurements for each robot, We assume distance measurements
        # z = true distance between robot and target + noise
        # noise is zero-mean Gaussian with variance sigma_z^2
        sigma_z = 0.1
        # sigma_z = 0.2*(1-1/())

        # Q: 2x2 state noise covariance
        sigma_q = 0.1
        Q = sigma_q * np.identity(2)

        # check for each target, which robot track it
        r_set_t = [np.zeros((1,)) for _ in range(self.Nt)]

        # store the trace of covariance for each target
        trace_t = np.zeros([1, self.Nt])
        trace_t_1 = np.zeros([1, self.Nt])

        # store the squared error
        sqerr_t = np.zeros([1, self.Nt])

        # store the tracking quality of each target
        quality_t = np.zeros([1, self.Nt])

        # if no robot
        if np.isnan(r_set):
            for j in range(self.Nt):
                tPos_hat_kp1[j, :] = tPos_hat_k[j, :]
                tSigma_hat_kp1[2 * j: 2 * j + 1, :] = (
                        tSigma_hat_k[2 * j: 2 * j + 1, :] + Q
                )

                # trace
                trace_t[j] = np.trace(tSigma_hat_kp1[2 * j: 2 * j + 1, :])

                # squared error
                sqerr_t[j] = (tPos_hat_kp1[j, :] - tPos_true_k[j, :]) * (
                        tPos_hat_kp1[j, :] - tPos_true_k[j, :]
                ).T

                # no measurements, tracking quality
                quality_t[j] = 0
                raise Exception("No Robot Detected")

        else:
            # loop through all the robot
            for j in range(self.Nt):
                # r_set_t_j = len(r_set_t[j])  # number of all the
                r_set_t_j = 1  ############## hard coded ##############
                # KF prediction step for target j
                jPos_hat_kp1_1 = tPos_hat_k[j, :].reshape((1, 2))
                # predict the pos of the target
                jSigma_hat_kp1_1 = tSigma_hat_k[2 * j: 2 * j + 2, :] + Q
                r_set_index = np.array([r_set - 1])
                for i in range(r_set_index.shape[0]):
                    # check if the target j is within the sensing range of robot i
                    if (
                            np.linalg.norm(rPos_kp1[i, 0:2] - tPos_true_k[j, :])
                            <= self.r_sen
                    ):
                        r_set_t[j] = np.append(r_set_t[j], [i])

                # if it is not tracked by any robot
                if len(r_set_t) == 0:
                    # copy the prediction step -- no update
                    tPos_hat_kp1 = jPos_hat_kp1_1
                    tSigma_hat_kp1[2 * j: 2 * j + 1, :] = jSigma_hat_kp1_1
                    raise Exception("Target not tracked by any robots")

                else:
                    # if there are some robots r_set_t[0,j] track target j
                    # KF update step
                    Zj = np.zeros([2 * r_set_t_j, 1])  # measurement
                    Zj_hat = 0 * Zj  # estimated measurement

                    # using range and bearing sensors
                    Hj = np.zeros([2 * r_set_t_j, 2])
                    Rj = np.zeros([2 * r_set_t_j, 2 * r_set_t_j])

                    # residual
                    res = np.zeros([2 * r_set_t_j, 1])
                    if self.scene == 1:
                        robots = 1
                    else:
                        robots = 2
                    for i in range(robots):
                        # distance based measurements and variance
                        dij = np.linalg.norm(rPos_kp1[i, 0:2] - tPos_true_k[j, :])
                        # real angle based measurements and variance
                        head_pi = wrapToPi(rPos_kp1[i, 2])  # TODO: verify the wrapToPi

                        aij_delta = (
                                np.arctan2(
                                    tPos_true_k[j, 1] - rPos_kp1[i, 1],
                                    tPos_true_k[j, 0] - rPos_kp1[i, 0],
                                )
                                - head_pi
                        )
                        aij = aij_delta - 2 * np.pi * np.round(aij_delta / (2 * np.pi))

                        noisei_d = sigma_z * dij * random.choice(self.noise) + 0.001
                        noisei_a = (
                                sigma_z * np.abs(aij) * random.choice(self.noise) + 0.001
                        )

                        distLin = self.distLinearization(
                            jPos_hat_kp1_1, rPos_kp1[i, 0:2]
                        )
                        angleLin = self.angleLinearization(
                            jPos_hat_kp1_1, rPos_kp1[i, 0:2], head_pi
                        )

                        # Apply Kalman Filter update
                        if self.scene == 1:
                            # real measurements
                            Zj[0:2, :] = np.array(
                                [[dij + noisei_d[0]], [wrapToPi(aij + noisei_a[0])]]
                            )  # measurement is noisy
                            Zj_hat[0:2, :] = np.array([[distLin], [angleLin]])
                            Hj[0, :] = self.distJac(jPos_hat_kp1_1, rPos_kp1[i, 0:2])
                            Hj[1, :] = self.angleJac(
                                jPos_hat_kp1_1, rPos_kp1[i, 0:2], head_pi
                            )

                            Rj[0, 0] = sigma_z * dij + 0.0001
                            Rj[1, 1] = sigma_z * np.abs(aij) + 0.0001

                            res[0, 0] = Zj[0, 0] - Zj_hat[0, 0]
                            res[1, 0] = (
                                                Zj[1, 0] - Zj_hat[1, 0]
                                        ) - 2 * np.pi * np.round(
                                (Zj[1, 0] - Zj_hat[1, 0]) / (2 * np.pi)
                            )
                        else:
                            if robot_curr[i].type == 2:
                                Zj[i, 0] = dij + noisei_d[0]
                                Zj_hat[i, :] = distLin
                                Hj[i, :] = self.distJac(
                                    jPos_hat_kp1_1, rPos_kp1[i, 0:2]
                                )
                                Rj[i, i] = sigma_z * dij + 0.0001
                                res[i, 0] = Zj[i, 0] - Zj_hat[i, 0]

                            elif robot_curr[i].type == 3:
                                Zj[i, 0] = aij + noisei_a[0]
                                Zj_hat[i, :] = angleLin
                                Hj[i, :] = self.angleJac(
                                    jPos_hat_kp1_1, rPos_kp1[i, 0:2], head_pi
                                )
                                Rj[i, i] = sigma_z * np.abs(aij) + 0.0001
                                res[i, 0] = (
                                                    Zj[i, 0] - Zj_hat[i, 0]
                                            ) - 2 * np.pi * np.round(
                                    (Zj[i, 0] - Zj_hat[i, 0]) / (2 * np.pi)
                                )

                    # residual covariance
                    Sj = Hj @ jSigma_hat_kp1_1 @ Hj.T + Rj
                    # Kalman Gain
                    Kj = jSigma_hat_kp1_1 @ Hj.T @ np.linalg.inv(Sj)
                    # state update
                    tPos_hat_kp1[j, :] = jPos_hat_kp1_1 + (Kj @ res).T
                    # covariance update
                    tSigma_hat_kp1[2 * j: 2 * j + 2, :] = (
                                                                  np.identity(2) - Kj @ Hj
                                                          ) @ jSigma_hat_kp1_1
                    # @ (
                    #             np.identity(2) - Kj @ Hj).T + Kj @ Rj @ Kj.T

                # trace
                trace_t[0][j] = np.trace(tSigma_hat_kp1[2 * j: 2 * j + 2, :])
                trace_t_1[0][j] = np.trace(jSigma_hat_kp1_1)
                # squared error
                sqerr_t[0][j] = (tPos_hat_kp1[j, :] - tPos_true_k[j, :]) @ (
                        tPos_hat_kp1[j, :] - tPos_true_k[j, :]
                ).T
                # tracking quality
                quality_t[0][j] = np.trace(jSigma_hat_kp1_1) - trace_t[0][j]
        trace_sig_diff = trace_t_1 - trace_t
        # print(f'updated trace is {trace_t}, prediction trace is {trace_t_1}, diff is {trace_sig_diff}')
        trace_ts = np.sum(trace_t)
        sqerr_ts = np.sum(sqerr_t)
        quality_ts = np.sum(quality_t)
        return (
            quality_ts,
            trace_ts,
            sqerr_ts,
            tPos_hat_kp1,
            tSigma_hat_kp1,
            trace_t,
            sqerr_t,
            trace_sig_diff,
        )

    def distLinearization(self, targetPose, robotPose):
        return np.linalg.norm(targetPose - robotPose)

    def angleLinearization(self, targetPose, robotPose, heading):
        Zj_hat_br_delta = (
                np.arctan2(
                    targetPose[0, 1] - robotPose[1],
                    targetPose[0, 0] - robotPose[0],
                )
                - heading
        )
        Zj_hat_br = Zj_hat_br_delta - 2 * np.pi * np.round(
            Zj_hat_br_delta / (2 * np.pi)
        )
        return Zj_hat_br

    def distJac(self, targetPose, robotPose):
        Zj_hat = self.distLinearization(targetPose, robotPose)
        jac = 1 / Zj_hat * (targetPose - robotPose)
        return jac

    def angleJac(self, targetPose, robotPose, heading):
        Zj_hat_br_delta = (
                np.arctan2(
                    targetPose[0, 1] - robotPose[1],
                    targetPose[0, 0] - robotPose[0],
                )
                - heading
        )
        Zj_hat = self.distLinearization(targetPose, robotPose)

        jac = (
                1
                / Zj_hat
                * np.array(
            [
                -np.sin(heading + Zj_hat_br_delta),
                np.cos(heading + Zj_hat_br_delta),
            ]
        )
        )
        return jac


def ten_runs(Nt, Nr, sz, scene, v, w, rounds, if_OPT, if_plot):
    if if_OPT:
        box_line = [
            dict(color="blue", linewidth=1.4),
            dict(color="red", linewidth=1.4),
            dict(color="orange", linewidth=1.4),
        ]
        names = ["Greedy", "Hungarian", "OPT", "Lower Baseline"]
    else:
        box_line = [
            dict(color="red", linewidth=1.4),
            dict(color="orange", linewidth=1.4),
        ]
        names = ["Greedy", "Hungarian", "Lower Baseline"]

    bound = 1 / 2 if scene == 1 else 1 / 3
    Nt_lst = [i for i in range(1, rounds + 1)]
    Nt_lst_str = [str(i) for i in Nt_lst]
    pos = np.zeros((len(Nt_lst), 1))
    fig, ax = plt.subplots(2)
    # ax = plt.axes()
    mean_val = np.zeros((len(Nt_lst), (len(names) - 1)))
    x_pos = np.zeros((len(Nt_lst), (len(names) - 1)))
    test_rlt = np.zeros((len(Nt_lst), 10, (len(names) - 1)))

    # Nt = 2

    for t in Nt_lst:
        Nt = t
        ind = t - 1
        for i in range(10):
            test = SensorAssignment(Nt, Nr, sz, scene)
            test.set_actions(v, w)
            test.step()

            if if_OPT:
                test_rlt[ind][i][0] = test.Greedy()[0]
                test_rlt[ind][i][1] = test.Hungarian()
                test_rlt[ind][i][2] = test.OPT()
            else:
                test_rlt[ind][i][0] = test.Greedy()[0]
                test_rlt[ind][i][1] = test.Hungarian()

            print(f"target num: {t} out of {rounds}, run {i}")

        # print(test_rlt)
        # print(np.mean(test_rlt, axis=0))
        mean_val[ind] = np.mean(test_rlt[ind], axis=0)

        # pos_val = 2 * (2*t-3)
        pos_val = 2 * t - 3
        pos[ind] = pos_val
        x_pos[ind] = (
            [pos_val - 1 / 3, pos_val, pos_val + 1 / 3]
            if if_OPT
            else [pos_val - 1 / 3, pos_val]
        )
        if if_OPT:
            bp = ax[0].boxplot(
                test_rlt[ind], positions=x_pos[ind]
            )  # , boxprops=box_line[t-2])
        else:
            bp = ax[0].boxplot(
                test_rlt[ind], positions=x_pos[ind]
            )  # , boxprops=box_line)
        write_csv(test_rlt, "ten_run_result" + str(Nt_lst[-1]) + "_scene_" + str(scene))

    # save the result
    # write_csv(test_rlt, "ten_run_result" + str(Nt_lst[-1]) + "_scene_" + str(scene))

    mean_val = mean_val.T
    x_pos = x_pos.T

    if if_plot:
        for i in range(len(x_pos)):
            # ax[1].plot(x_pos[i], mean_val[i])
            ax[1].plot(Nt_lst, mean_val[i], "-s", label=names[i])

        if if_OPT:
            ax[1].plot(Nt_lst, mean_val[0] * bound, "-s", label=names[-1])
        else:
            ax[1].plot(Nt_lst, mean_val[-1] * bound, "-s", label=names[-1])
        # ax[1].legend(['OPT', 'Greedy', 'Hungarian', '1/2 OPT'])
        ax[0].set_xticks(
            list(
                pos.reshape(
                    len(pos),
                )
            )
        )
        ax[0].set_xticklabels(Nt_lst_str)

        ax[1].set_xticks(Nt_lst)
        ax[1].set_xticklabels(Nt_lst_str)
        ax[1].legend()
        # ax[0].xlim(0, 4*len(Nt_lst))

        plt.grid()
        plt.show()


def confidence_ellipse(cov, c, color, k=2.296):
    # find the eigenvalue and eigenvector of the given covariance matrix
    w, v = np.linalg.eig(cov)
    largest_eig_val = max(w)
    largest_vec_ind = np.where(w == largest_eig_val)[0]
    largest_eig_vec = v[:, largest_vec_ind]
    smallest_eig_val = min(w)

    # calculate the angle between the largest eigenvector and the x-axis
    angle = np.arctan2(largest_eig_vec[1], largest_eig_vec[0])
    if angle < 0:
        angle = angle + 2 * np.pi

    # initialize the values for ellipse calculation
    chi_square_val = 2.4477
    phi = angle[0]
    q = np.linspace(0, 2 * np.pi, 30)

    # find the a and b for the ellipse
    a = chi_square_val * np.sqrt(largest_eig_val)
    b = chi_square_val * np.sqrt(smallest_eig_val)

    ellipse_x_r = a * np.cos(q)
    ellipse_y_r = b * np.sin(q)

    # rotate the ellipse to get the correct confidence ellipse
    R = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
    ellipse_r = np.array([ellipse_x_r, ellipse_y_r]).T
    r_ellipse = ellipse_r @ R

    plt.plot(r_ellipse[:, 0] + c[0], r_ellipse[:, 1] + c[1], color)
    plt.axis("equal")

    return None


def write_csv(data: np.ndarray, name: str):
    """
    Function save the given numpy array data to a csv file with specified name
    Parameters
    ----------
    data: numpy array to be saved
    name: name of the file to be saved, without .csv

    Returns
    -------
    None

    """
    data = data.tolist()
    with open(name + ".csv", "w", newline="") as csvfile:
        my_writer = csv.writer(csvfile, delimiter=",")
        my_writer.writerows(data)

    return None


def read_csv(name: str):
    """
    Read csv file that contains 3D numpy array, and convert it into ndarray
    Parameters
    ----------
    name: name of the file, e.g.: "test_file.csv"

    Returns
    -------
    data_num: 3D numpy array

    """
    with open(name, "r") as csvFile:
        raw_data = csv.reader(csvFile)
        raw_data = list(raw_data)

    data_str = []
    for row in raw_data:
        nwrow = []
        for r in row:
            nwrow.append(eval(r))
        data_str.append(nwrow)

    data_num = np.array([[[float(k) for k in j] for j in i] for i in data_str])

    return data_num


def read_csv_single_line(name):
    with open(name, "r") as csvFile:
        raw_data = csv.reader(csvFile)
        raw_data = list(raw_data)

    for row in raw_data:
        nwrow = []
        for r in row:
            nwrow.append(eval(r))
    data = np.array(nwrow)
    return data


def target_moving(Nt, Nr, sz, scene, v, w, steps):
    test = SensorAssignment(Nt, Nr, sz, scene)
    test.set_actions(v, w)
    dt = 0.5
    u_tar = 1.3
    d_theta = 0.2
    target_radius = [
        np.sqrt(target.ground_truth[0] ** 2 + target.ground_truth[1] ** 2)
        for target in test.targets
    ] # calculates the distance of the target from the origin
    target_angle = [
        np.arctan2(target.ground_truth[1], target.ground_truth[0])
        for target in test.targets
    ] # calculates the heading angle of the target or theta
    times = np.linspace(0, steps * dt, steps)
    colors = ["r", "g", "b"]
    u = 1
    target_hist = np.zeros((Nt, steps, 2))
    robot_hist = np.zeros((test.Nr, steps, 2))
    error_hist = np.zeros((steps,))
    for i in range(test.Nr):
        test.robots[i].pos_hist = np.zeros((steps, 3))

    # setup the target trajectory
    for step in range(20): # since the steps is 20, will run 20 times
        t = times[step]

        # perform a step of tracking evaluation
        test.step() # the main point of this function is to update the covariance matrix with the EKF function 

        # use the pair with the best quality
        cov, robot, action, pairs = test.Greedy() # after the step function, we can use the greedy method to get the best pair since the covariance matrix is updated
        # pairs = {0: pairs[0], 1: pairs[1], 2: pairs[2]}
        pair = {pairs[0] : 0, pairs[1]: 1, pairs[2]: 2}

        plt.figure(step)
        plt.title(f"time step {step} at time {t} s with max_cov = {cov: 0.3f}")
        error_round = 0
        for target_ind in range(test.Nt):
            # get the target pose and target sigma
            target_loc = test.targets[target_ind].ground_truth
            target_est = test.targets[target_ind].estimated_location
            target_sig = test.sig_matrix[robot[target_ind]][action[target_ind]][
                         2 * target_ind: 2 * target_ind + 2, :
                         ] # get the covariance matrix of the target
            target_hist[target_ind, step, :] = target_loc # this target at this step is at this location (i think)
            
            plt.plot(
                target_loc[0], target_loc[1], colors[target_ind] + "*", markersize=10
            ) # this is the true location of the target represented by colored stars
            plt.xlim([-3, 13])
            plt.ylim([-3, 13])
            plt.axis("equal")
            plt.plot(target_est[0], target_est[1], colors[target_ind] + "s", mfc="w") # this is the estimated location of the target represented by white squares
            plt.plot(
                target_hist[target_ind, 0: step + 1, 0],
                target_hist[target_ind, 0: step + 1, 1],
                colors[target_ind] + ":",
            ) # this is the trajectory of the target represented by colored lines
            
            # get the covariance plot
            # confidence_ellipse(target_sig, target_est, colors[target_ind])
            
            # does this actually work? not sure if indexing the robot list returned form greedy method with the target_ind 
            # will give me the correct robot match for that target_ind
            #robot_pair = test.robots_comb[robot[target_ind]] # here I am getting the specific robot object that is tracking the target_ind 
            robot_pair = test.robots_comb[pair.get(target_ind)]
            # print(f"Target {target_ind} is assigned to robot: " , pair.get(target_ind))

            # i have to turn it into an array since robot_pair has no len attribute
            robot_pair = [robot_pair]
            
            error_round += find_error(target_loc, target_est)

            for robot_ind in range(len(robot_pair)): # this should always loop only once if scene 1
                robot_curr = robot_pair[robot_ind] # extract the robot object from the list
                robot_pos = robot_curr.location # get the location of the robot
                robot_curr.pos_hist[step] = robot_pos # store the location of the robot at this step
                robot_hist[robot_curr.id, step, :] = robot_pos[0:2] #storing x and y location of this robot id at this step
                action_taken = test.action_for_each_robot[action[target_ind]] # get the action taken by the robot for this target
                
                plt.plot(
                    robot_curr.pos_hist[0: step + 1, 0],
                    robot_curr.pos_hist[0: step + 1, 1],
                    "D", # this is the trajectory of the robot represented by empty squares
                    mec="k",
                    mfc="w",
                )
                plt.plot(
                    robot_curr.pos_hist[0: step + 1, 0],
                    robot_curr.pos_hist[0: step + 1, 1],
                    # "k:",# this is the trajectory of the robot represented by black lines
                )
                plt.plot(
                    robot_pos[0],
                    robot_pos[1],
                    "kD", # this is the current location of the robot represented by black diamonds
                )
                plt.plot(
                    [robot_pos[0], target_est[0]],
                    [robot_pos[1], target_est[1]],
                    colors[target_ind],
                )
                
                # update robot movement
                robot_next = robot_curr.get_action(action_taken, dt) # get the next location of the robot based on the action taken
                robot_curr.update_pos(robot_next) # update the location of the robot with robot_next that contains new_x, new_y, new_th

            ###### two ways of finding the angular change ########
            # first is use a constant linear vel and find the change in angle
            target_angle[target_ind] += u_tar / target_radius[target_ind] * dt 
            # This is equivalent to moving the target ^^^^
            # along the circumference of a circle centered at the origin.

            # another way is to use a constant angular change, but the linear velocity is not promised


            # target_angle[target_ind] += d_theta
            target_next = [
                np.cos(target_angle[target_ind]) * target_radius[target_ind],
                np.sin(target_angle[target_ind]) * target_radius[target_ind],
            ]
            test.targets[target_ind].update(target_next, target_sig)
        error_hist[step] = error_round

    plt.show()

    write_csv(robot_hist, "robot")
    write_csv(target_hist, "target")


def plot_result(file_name: str):
    data = read_csv(file_name)
    rounds = data.shape[0]
    types = data.shape[2]
    file_name_split = file_name.split(".")
    scene = int(file_name_split[0][-1])
    if types == 2:
        if_OPT = False
    elif types == 3:
        if_OPT = True
    else:
        raise Exception("Check the types number.")
    names = (
        [r"q(GREEDY)", r"q(HUNGARIAN)", r"$\frac{1}{3}$q(HUNGARIAN)"]
        if not if_OPT
        else [r"q(GREEDY)", r"q(HUNGARIAN)", r"q(OPT)", r"$\frac{1}{3}$q(OPT)"]
    )
    # colors = ["r", "g", "b"]
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]
    # https://matplotlib.org/stable/users/prev_whats_new/dflt_style_changes.html
    # plt.style.use('seaborn')  # I personally prefer seaborn for the graph style, but you may choose whichever you want.
    # params = {"ytick.color": "black",
    #           "xtick.color": "black",
    #           "axes.labelcolor": "black",
    #           "axes.edgecolor": "black",
    #           "text.usetex": True,
    #           "font.family": "serif",
    #           "font.serif": ["Computer Modern Serif"]}
    # plt.rcParams.update(params)

    bound = 1 / 2 if scene == 1 else 1 / 3
    Nt_lst = [i for i in range(1, rounds + 1)]
    Nt_lst_str = [str(i) for i in Nt_lst]
    pos = np.zeros((len(Nt_lst), 1))
    ax = plt.subplot()
    # ax = plt.axes()
    mean_val = np.zeros((len(Nt_lst), (len(names) - 1)))
    std_val = np.zeros((len(Nt_lst), (len(names) - 1)))
    x_pos = np.zeros((len(Nt_lst), (len(names) - 1)))
    # test_rlt = np.zeros((len(Nt_lst), 10, (len(names) - 1)))
    test_rlt = data

    linestyle = [(0, (5, 5)), (0, (3, 1, 1, 1)), (0, (1, 1)), 'solid']

    for t in Nt_lst:
        Nt = t
        ind = t - 1

        # print(test_rlt)
        # print(np.mean(test_rlt, axis=0))
        mean_val[ind] = np.mean(test_rlt[ind], axis=0)
        std_val[ind] = np.std(test_rlt[ind], axis=0)
        # pos_val = 2 * (2*t-3)
        pos_val = 2 * t - 1
        pos[ind] = pos_val
        x_pos[ind] = (
            [pos_val - 1 / 3, pos_val, pos_val + 1 / 3]
            if if_OPT
            else [pos_val - 1 / 3, pos_val]
        )

    # save the result
    # write_csv(test_rlt, 'ten_run_result' + str(rounds))

    mean_val = mean_val.T
    if if_OPT:
        OPT = mean_val[0, :] / mean_val[2, :]
        Hun = mean_val[0, :] / mean_val[1, :]
        print(f'OPT ratio is {np.mean(OPT)}, Hungarian ratio is {np.mean(Hun)}')
    else:
        Hun = mean_val[0, :] / mean_val[1, :]
        print(f'Hungarian ratio is {np.mean(Hun)}')
    std_val = std_val.T
    x_pos = x_pos.T
    for i in range(len(x_pos)):
        ax.plot(Nt_lst, mean_val[i], "-s", label=names[i], color=colors[i],
                linewidth=1.5,  # 5,
                markersize=2,  # 10,
                linestyle=linestyle[i]
                )
        (_, caps, _) = ax.errorbar(
            Nt_lst, mean_val[i], yerr=std_val[i], color=colors[i], linestyle="None",
            lw=1.5,
            capsize=2.5
        )
        print(colors[i])
        for cap in caps:
            cap.set_markeredgewidth(2.25)
    if if_OPT:
        ax.plot(
            Nt_lst,
            mean_val[2] * bound,  # third entry for mean_val is OPT
            "-s",
            label=names[-1],
            color=colors[-1],
            linewidth=2,
            markersize=2.5,
            linestyle=linestyle[-1]
        )
        # ax.errorbar(Nt_lst, mean_val[0], std_val[0], color=colors[-1], lw=3)
    else:
        ax.plot(
            Nt_lst,
            mean_val[-1] * bound,  # last entry for mean_val is Hungarian-
            "-s",
            label=names[-1],
            color=colors[-1],
            linewidth=2,
            markersize=2.5,
            linestyle=linestyle[-1]
        )
        # ax.errorbar(Nt_lst, mean_val[-1], yerr=std_val[-1], color=colors[-1], lw=3)

    # xticks = np.arange(0, Nt, 5)[1::]
    # ax.set_xticks(Nt_lst)
    # ax.set_xticklabels(Nt_lst)
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticks)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_xlabel(r'Number of Targets', fontsize=16)
    ax.set_ylabel(r"Tracking Quality", fontsize=16)
    ax.legend(fontsize=13)
    # print(np.arange(0, Nt, 5)[1::])
    # plt.grid()
    ratio = 1.0
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
    plt.savefig(file_name_split[0] + '.pdf', bbox_inches='tight')
    plt.show()


def single_run(Nt, Nr, sz, scene, v, w):
    test = SensorAssignment(Nt, Nr, sz, scene)
    test.set_actions(v, w)
    test.step()

    print("--- OPT Algorithm ---")
    t_opt_start = time.time()
    max = test.OPT()
    t_opt_elapse = time.time() - t_opt_start
    print(f"OPT Algorithm used {t_opt_elapse: 0.5f} sec")
    print("------------------------")
    print("--- Greedy Algorithm ---")
    t_grdy_start = time.time()
    test.Greedy()
    t_grdy_elapse = time.time() - t_grdy_start
    print(f"Greedy Algorithm used {t_grdy_elapse: 0.5f} sec")
    print("------------------------")
    t_hun_start = time.time()
    test.Hungarian()
    t_hun_elapse = time.time() - t_hun_start
    print(f"Hungarian Method used {t_hun_elapse: 0.5f} sec")


def cov_error_plot(num):
    cov_name = "cov_" + str(num) + ".csv"
    error_name = "error_avg_" + str(num) + ".csv"
    cov_data = read_csv_single_line(cov_name)
    error_data = read_csv_single_line(error_name)
    time = np.arange(0, 100 + 0.5, 0.5)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(time, cov_data, linewidth=2)
    ax[0].set_xlabel(r'Time ($s$)', fontsize=12)
    ax[0].set_ylabel(r'Tr($\Sigma_{T}$)', fontsize=14)
    ax[1].plot(time, error_data, linewidth=2)
    ax[1].set_xlabel(r'Time ($s$)', fontsize=12)
    ax[1].set_ylabel(r'$err_T$', fontsize=14)
    ax[0].tick_params(axis="both", which="major", labelsize=12)
    ax[1].tick_params(axis="both", which="major", labelsize=12)

    fig.tight_layout()
    plt.savefig("cov_error_" + str(num) + ".pdf")
    plt.show()


def main():
    Nt = 3
    Nr = Nt  # for bearing and range sensor
    sz = 10
    scene = 1
    v = [
        1,
        -1, 0
    ]
    w = [
        -0.7,
        0.7, 0
    ]
    steps = 20

    # test setup
    # test = SensorAssignment(Nt, Nr, sz, scene)
    # test.set_actions(v, w)
    # test.step()

    target_moving(Nt, Nr, sz, scene, v, w, steps)
    # ten_runs(Nt, Nr, sz, scene, v, w, 30, if_OPT=False, if_plot=False)
    # plot_result("ten_run_result8_scene_2.csv")
    # single_run(Nt, Nr, sz, scene, v, w)
    # plot_result('ten_run_result26_scene_2.csv')

    # cov_error_plot(2)


if __name__ == "__main__":
    main()
