import pandas as pd
from matplotlib import pyplot as plt
from numpy import hstack, vstack, linalg
from math import atan2
from prediction import compute_prediction_linear_kalman
from update import compute_update_linear_kalman_gps_pos, compute_update_linear_kalman_gps


def compute_timedelta(data):
    """

    :param data:
    :return:
    """

    diff = data.index.to_series().diff()
    time_delta = [time.total_seconds() for time in diff]

    return time_delta


class LinearKalman:

    def __init__(self, state_init, P_init, Q_init, R_init):
        self.state_init = state_init
        self.P_init = P_init
        self.Q_init = Q_init
        self.R_init = R_init

        self.P_mat_final = None

    def compute_kalman_gps_pos(self, gps_data):
        state_pred_list, state_upd_list = [], []
        P_mat_pred_list, P_mat_upd_list = [], []

        dt_list = compute_timedelta(data=gps_data)
        gps_values = gps_data.values

        state_var = self.state_init
        P_var = self.P_init
        Q_var = self.Q_init
        R_var = self.R_init

        for ind, dt in enumerate(dt_list):

            if ind is 0:
                state_pred_list.append(state_var)
                P_mat_pred_list.append(P_var)

                state_upd_list.append(state_var)
                P_mat_upd_list.append(P_var)

            else:

                state_pred, P_pred = compute_prediction_linear_kalman(state_vector=state_var,
                                                                      P_mat=P_var,
                                                                      Q_mat=Q_var,
                                                                      dt=dt)
                state_pred_list.append(state_pred)
                P_mat_pred_list.append(P_pred)

                state_upd, P_upd = compute_update_linear_kalman_gps_pos(state_vector=state_pred,
                                                                        meas_vector=gps_values,
                                                                        P_mat=P_pred,
                                                                        R_mat=R_var)
                state_upd_list.append(state_upd)
                P_mat_upd_list.append(P_upd)

                # for the newt loop
                state_var = state_upd
                P_var = P_upd
