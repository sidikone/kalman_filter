import numpy as np
import pandas as pd


def state_dic(typ="ctrv"):
    out_dict = None
    if typ == "cv":
        out_dict = {"x": 0, "y": 1, "yaw": 2, "speed": 3}

    if typ == "ctrv":
        out_dict = {"x": 0, "y": 1, "yaw": 2, "speed": 3, "yaw_rate": 4}

    if typ == "ctra":
        out_dict = {"x": 0, "y": 1, "yaw": 2, "speed": 3, "yaw_rate": 4, "ax": 5}

    return out_dict


def compute_timedelta(data):
    """

    :param data:
    :return:
    """

    diff = data.index.to_series().diff()
    time_delta = [time.total_seconds() for time in diff]

    return time_delta


def cartesian_state_cv2DataFrame(state, ref_data):
    x_ekf, y_ekf = np.array(state)[:, 0], np.array(state)[:, 1]  # pos (m) ekf output
    vx_ekf, vy_ekf = (
        np.array(state)[:, 2],
        np.array(state)[:, 3],
    )  # yaw, speed ekf output
    ekf_index = ref_data.index

    data_frame = pd.DataFrame(
        {"x": x_ekf, "y": y_ekf, "vx": vx_ekf, "vy": vy_ekf},
        index=ekf_index[: len(x_ekf)],
    )

    return data_frame


def cartesian_P_cv2DataFrame(P, data):
    p_list = []
    for ind in range(len(P)):
        p_vect = np.diag(np.asarray(P[ind]))
        p_list.append(p_vect)

    p_x, p_y = np.array(p_list)[:, 0], np.array(p_list)[:, 1]  # pos (m) ekf output
    p_yaw, p_spd = (
        np.array(p_list)[:, 2],
        np.array(p_list)[:, 3],
    )  # yaw, speed ekf output

    ekf_p = pd.DataFrame(
        {"x": p_x, "y": p_y, "vx": p_yaw, "vy": p_spd, },
        index=data.index[: len(p_x)],
    )

    return ekf_p


def cv2DataFrame(state, ref_data):
    x_ekf, y_ekf = np.array(state)[:, 0], np.array(state)[:, 1]  # pos (m) ekf output
    yaw_ekf, spd_ekf = (
        np.array(state)[:, 2],
        np.array(state)[:, 3],
    )  # yaw, speed ekf output
    ekf_index = ref_data.index

    data_frame = pd.DataFrame(
        {"x": x_ekf, "y": y_ekf, "yaw": yaw_ekf, "speed": spd_ekf},
        index=ekf_index[: len(x_ekf)],
    )

    return data_frame


def ctrv2DataFrame(state, ref_data):
    x_ekf, y_ekf = np.array(state)[:, 0], np.array(state)[:, 1]  # pos (m) ekf output
    yaw_ekf, spd_ekf = (
        np.array(state)[:, 2],
        np.array(state)[:, 3],
    )  # yaw, speed ekf output
    yaw_rate_ekf = np.array(state)[:, 4]
    ekf_index = ref_data.index

    data_frame = pd.DataFrame(
        {
            "x": x_ekf,
            "y": y_ekf,
            "yaw": yaw_ekf,
            "speed": spd_ekf,
            "yaw_rate": yaw_rate_ekf,
        },
        index=ekf_index[: len(x_ekf)],
    )

    return data_frame


def P_cv2DataFrame(P, data):
    p_list = []
    for ind in range(len(P)):
        p_vect = np.diag(np.asarray(P[ind]))
        p_list.append(p_vect)

    p_x, p_y = np.array(p_list)[:, 0], np.array(p_list)[:, 1]  # pos (m) ekf output
    p_yaw, p_spd = (
        np.array(p_list)[:, 2],
        np.array(p_list)[:, 3],
    )  # yaw, speed ekf output

    ekf_p = pd.DataFrame(
        {"x": p_x, "y": p_y, "yaw": p_yaw, "speed": p_spd, },
        index=data.index[: len(p_x)],
    )

    return ekf_p


def P_ctrv2DataFrame(P, data):
    p_list = []
    for ind in range(len(P)):
        p_vect = np.diag(np.asarray(P[ind]))
        p_list.append(p_vect)

    p_x, p_y = np.array(p_list)[:, 0], np.array(p_list)[:, 1]  # pos (m) ekf output
    p_yaw, p_spd = (
        np.array(p_list)[:, 2],
        np.array(p_list)[:, 3],
    )  # yaw, speed ekf output
    p_yaw_rate = np.array(p_list)[:, 4]

    ekf_p = pd.DataFrame(
        {"x": p_x, "y": p_y, "yaw": p_yaw, "speed": p_spd, "yaw_rate": p_yaw_rate, },
        index=data.index[: len(p_x)],
    )

    return ekf_p
