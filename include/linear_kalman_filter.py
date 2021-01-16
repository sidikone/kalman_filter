import pandas as pd
from matplotlib import pyplot as plt
from numpy import hstack, vstack, linalg
from math import atan2
from prediction import compute_prediction_linear_kalman
from update import compute_update_linear_kalman_gps_pos, compute_update_linear_kalman_gps
from utils import cartesian_state_cv2DataFrame, cartesian_P_cv2DataFrame
from progress.bar import IncrementalBar
from utils import compute_timedelta


class LinearKalman:

    def __init__(self, state_init, P_init, Q_init, R_init):
        self.state_init = state_init
        self.P_init = P_init
        self.Q_init = Q_init
        self.R_init = R_init

        self.P_mat_final = None

    def compute_kalman_gps_pos(self, gps_data):

        my_bar = IncrementalBar('linear kalman gps-pos', max=len(gps_data))
        state_pred_list, state_upd_list = [], []
        P_pred_list, P_upd_list = [], []

        dt_list = compute_timedelta(data=gps_data)
        gps_values = gps_data.values

        state_var = self.state_init
        P_var = self.P_init
        Q_var = self.Q_init
        R_var = self.R_init

        for ind, dt in enumerate(dt_list):

            if ind is 0:
                state_pred_list.append(state_var)
                P_pred_list.append(P_var)

                state_upd_list.append(state_var)
                P_upd_list.append(P_var)

            else:

                state_pred, P_pred = compute_prediction_linear_kalman(state_vector=state_var,
                                                                      P_mat=P_var,
                                                                      Q_mat=Q_var,
                                                                      dt=dt)
                state_pred_list.append(state_pred)
                P_pred_list.append(P_pred)

                state_upd, P_upd = compute_update_linear_kalman_gps_pos(state_vector=state_pred,
                                                                        meas_vector=gps_values[ind],
                                                                        P_mat=P_pred,
                                                                        R_mat=R_var)
                state_upd_list.append(state_upd)
                P_upd_list.append(P_upd)

                # for the newt loop
                state_var = state_upd
                P_var = P_upd
            my_bar.next()

        my_bar.finish()
        state_frame = cartesian_state_cv2DataFrame(state_upd_list, gps_data)
        P_frame = cartesian_P_cv2DataFrame(P_upd_list, gps_data)
        return state_frame, P_frame

    def compute_kalman_gps(self, pos_data, spd_data):

        data = pd.concat([pos_data, spd_data], axis=1)
        my_bar = IncrementalBar('linear kalman gps', max=len(data))

        state_pred_list, state_upd_list = [], []
        P_pred_list, P_upd_list = [], []

        dt_list = compute_timedelta(data=data)
        gps_values = data .values

        state_var = self.state_init
        P_var = self.P_init
        Q_var = self.Q_init
        R_var = self.R_init

        for ind, dt in enumerate(dt_list):

            if ind is 0:
                state_pred_list.append(state_var)
                P_pred_list.append(P_var)

                state_upd_list.append(state_var)
                P_upd_list.append(P_var)

            else:

                state_pred, P_pred = compute_prediction_linear_kalman(state_vector=state_var,
                                                                      P_mat=P_var,
                                                                      Q_mat=Q_var,
                                                                      dt=dt)
                state_pred_list.append(state_pred)
                P_pred_list.append(P_pred)

                state_upd, P_upd = compute_update_linear_kalman_gps(state_vector=state_pred,
                                                                    meas_vector=gps_values[ind],
                                                                    P_mat=P_pred,
                                                                    R_mat=R_var)
                state_upd_list.append(state_upd)
                P_upd_list.append(P_upd)

                # for the newt loop
                state_var = state_upd
                P_var = P_upd
            my_bar.next()

        my_bar.finish()
        state_frame = cartesian_state_cv2DataFrame(state_upd_list, pos_data)
        P_frame = cartesian_P_cv2DataFrame(P_upd_list, pos_data)
        return state_frame, P_frame
