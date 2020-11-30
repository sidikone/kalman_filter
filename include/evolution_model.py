from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import cv2DataFrame, ctrv2DataFrame


def unit_cv_polar(x_in, y_in, theta_in, speed_in, dt=1):
    """

    :param x_in:
    :param y_in:
    :param theta_in:
    :param speed_in:
    :param dt:
    :return:
    """

    dx = speed_in * np.cos(theta_in) * dt
    dy = speed_in * np.sin(theta_in) * dt

    x_out = x_in + dx
    y_out = y_in + dy

    theta_out = deepcopy(theta_in)
    speed_out = deepcopy(speed_in)

    return x_out, y_out, theta_out, speed_out


def unit_ctrv_polar(x_in, y_in, theta_in, speed_in, w_in, dt=1):
    """

    :param x_in:
    :param y_in:
    :param theta_in:
    :param speed_in:
    :param w_in:
    :param dt:
    :return:
    """

    if np.abs(w_in) <= 1e-10:

        dx = speed_in * np.cos(theta_in) * dt
        dy = speed_in * np.sin(theta_in) * dt

    else:
        ratio = speed_in / w_in
        dx = ratio * np.sin(dt * w_in) * np.cos(theta_in) + ratio * (
                np.cos(dt * w_in) * np.sin(theta_in) - np.sin(theta_in)
        )
        dy = ratio * (np.cos(theta_in) - np.cos(dt * w_in) * np.cos(theta_in)) + ratio * np.sin(
            dt * w_in
        ) * np.sin(theta_in)

    x_out = x_in + dx
    y_out = y_in + dy
    theta_out = theta_in + w_in * dt
    speed_out = deepcopy(speed_in)
    w_out = deepcopy(w_in)

    return x_out, y_out, theta_out, speed_out, w_out


def cv_imu_command_step(state_v_in, a_imu, w_imu, dt_imu):
    """

    :param state_v_in:
    :param a_imu:
    :param w_imu:
    :param dt_imu:
    :return:
    """
    x_t, y_t, theta_t, v_t = state_v_in[0], state_v_in[1], state_v_in[2], state_v_in[3]
    v_cmd = v_t + a_imu * dt_imu  # speed as a command
    theta_cmd = theta_t + w_imu * dt_imu  # theta as a command
    state_v_out = unit_cv_polar(x_in=x_t, y_in=y_t, theta_in=theta_cmd, speed_in=v_cmd, dt=dt_imu)

    return np.array(state_v_out)


def ctrv_imu_command_step(state_v_in, a_imu, w_imu, dt_imu):
    """

    :param state_v_in:
    :param a_imu:
    :param w_imu:
    :param dt:
    :return:
    """
    x_t, y_t, theta_t, v_t = state_v_in[0], state_v_in[1], state_v_in[2], state_v_in[3]
    v_cmd = v_t + a_imu * dt_imu  # speed as a command
    state_v_out = unit_ctrv_polar(
        x_in=x_t, y_in=y_t, theta_in=theta_t, speed_in=v_cmd, w_in=w_imu, dt=dt_imu
    )

    return np.array(state_v_out)


def batch_cv_integration(state_init, imu_accel, imu_gyro):
    save_state = []
    state_v_out = []
    dt = 1. / 100

    for count in range(len(imu_accel)):

        if count == 0:
            state_v = deepcopy(state_init)
            # save_state.append(state_v)

        else:
            state_v = deepcopy(state_v_out)

        state_v_out = cv_imu_command_step(
            state_v_in=state_v,
            a_imu=imu_accel[count],
            w_imu=imu_gyro[count],
            dt_imu=dt,
        )

        save_state.append(state_v_out)

    return cv2DataFrame(state=save_state, ref_data=imu_accel)


def batch_ctrv_integration(state_init, imu_accel, imu_gyro):
    save_state = []
    state_v_out = []
    dt = 1. / 100
    # dt_list = compute_timedelta(imu_accel)

    # for count, dt in enumerate(dt_list):
    for count in range(len(imu_accel)):
        if count == 0:
            state_v = deepcopy(state_init)
            # save_state.append(state_v)

        else:
            state_v = deepcopy(state_v_out)

        state_v_out = ctrv_imu_command_step(
            state_v_in=state_v,
            a_imu=imu_accel[count],
            w_imu=imu_gyro[count],
            dt_imu=dt,
        )

        save_state.append(state_v_out)

    return ctrv2DataFrame(state=save_state, ref_data=imu_accel)


def plot_ref_gps(ax):
    gps = pd.read_csv('ref_pos.csv')
    gps -= gps.iloc[0]

    ax.set_title('Plot trace')
    ax.plot(gps['ref_pos_x (m)'], gps['ref_pos_y (m)'], label='true position')
    ax.set_ylabel('Y (m)')
    ax.set_xlabel('X (m)')
    return None


def run_batch_cv_kalman(ax):
    init = np.array([0, 0, np.deg2rad(183), .0])
    accel = pd.read_csv('ref_accel.csv')
    gyro = pd.read_csv('ref_gyro.csv')
    em_cv = batch_cv_integration(state_init=init,
                                 imu_accel=accel['ref_accel_x (m/s^2)'],
                                 imu_gyro=np.deg2rad(gyro['ref_gyro_z (deg/s)']))
    ax.plot(em_cv['x'], em_cv['y'], '--', label='CV model')
    return None


def run_batch_ctrv_kalman(ax):
    init = np.array([0, 0, np.deg2rad(183), .0])
    accel = pd.read_csv('ref_accel.csv')
    gyro = pd.read_csv('ref_gyro.csv')
    em_cv = batch_ctrv_integration(state_init=init,
                                   imu_accel=accel['ref_accel_x (m/s^2)'],
                                   imu_gyro=np.deg2rad(gyro['ref_gyro_z (deg/s)']))
    ax.plot(em_cv['x'], em_cv['y'], marker='o', linestyle='--', fillstyle='none', label='CTRV model')
    return None


def main():
    fig, ax = plt.subplots()
    plot_ref_gps(ax=ax)
    run_batch_cv_kalman(ax=ax)
    run_batch_ctrv_kalman(ax=ax)
    ax.legend()
    plt.show()
    return None


if __name__ == '__main__':
    main()
