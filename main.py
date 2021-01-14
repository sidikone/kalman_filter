import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from include.evolution_model import batch_cv_integration, plot_ref_gps
from include.data_loader import SimuData
from include.linear_kalman_filter import LinearKalman
import datetime


def run_batch_cv_kalman(ax) -> None:
    init = np.array([0, 0, np.deg2rad(183), .0])
    accel = pd.read_csv('data/simu_dataset_1/ref_accel.csv')
    gyro = pd.read_csv('data/simu_dataset_1/ref_gyro.csv')

    dt = 1. / 100

    spd_from_accel = accel[['ref_accel_x (m/s^2)', 'ref_accel_y (m/s^2)']] * dt
    spd_from_accel.columns = ["ref_spd_x (m/s)", "ref_spd_y (m/s)"]

    spd_from_accel = spd_from_accel.cumsum()

    print(list(accel))
    print(accel.head(3))
    print(spd_from_accel.head(3))

    ax.plot(spd_from_accel)

    em_cv = batch_cv_integration(state_init=init,
                                 imu_accel=accel['ref_accel_x (m/s^2)'],
                                 imu_gyro=np.deg2rad(gyro['ref_gyro_z (deg/s)']))

    ax.plot(em_cv['x'], em_cv['y'], '--', label='CV model')


def main_data_loader() -> None:
    data_sim = SimuData(path="data/simu_dataset_1")

    data_sim.set_imu_frequency(fs=50)
    data_sim.set_gps_frequency(fs=10)

    data_sim.set_accel_data_noise(std=0.25)
    data_sim.set_gyro_data_noise(std=0.15)
    data_sim.set_gps_pos_data_noise(std=3.)
    data_sim.set_gps_spd_data_noise(std=0.50)

    accel, gyro = data_sim.get_imu_data()
    accel_name = list(accel)
    gyro_name = list(gyro)
    pos, spd = data_sim.get_gps_data()
    spd_name = list(spd)

    fig, ax = plt.subplots()
    ax.plot(pos["ref_pos_x (m)_noise"], pos["ref_pos_y (m)_noise"])
    ax.plot(pos["ref_pos_x (m)"], pos["ref_pos_y (m)"])
    plt.show()

    y_labels = ['ax (m/s²)', 'ay (m/s²)', 'az (m/s²)']
    fig, ax = plt.subplots(3, 1, sharex=True)
    for ind, name in enumerate(accel_name[3:]):
        ax[ind].plot(accel[name])

    for ind, name in enumerate(accel_name[:3]):
        ax[ind].plot(accel[name])
        ax[ind].set_ylabel(y_labels[ind])

    y_labels = ['gx (rad/s²)', 'gy (rad/s²)', 'gz (rad/s²)']
    fig, ax = plt.subplots(3, 1, sharex=True)
    for ind, name in enumerate(gyro_name[3:]):
        ax[ind].plot(gyro[name])

    for ind, name in enumerate(gyro_name[:3]):
        ax[ind].plot(gyro[name])
        ax[ind].set_ylabel(y_labels[ind])

    plt.show()

    y_labels = ['vx (m/s)', 'vy (m/s)', 'vz (m/s)', 'v_norm (m/s)']
    fig, ax = plt.subplots(4, 1, sharex=True)
    for ind, name in enumerate(spd_name[4:]):
        ax[ind].plot(spd[name])

    for ind, name in enumerate(spd_name[:4]):
        ax[ind].plot(spd[name])
        ax[ind].set_ylabel(y_labels[ind])

    plt.show()


def main() -> None:
    data_sim = SimuData(path="data/simu_dataset_1")
    data_sim.set_gps_frequency(fs=10)
    data_sim.set_gps_pos_data_noise(std=3.)
    data_sim.set_gps_spd_data_noise(std=0.50)
    pos, spd = data_sim.get_gps_data()

    pos_name = list(pos)
    c_time = (datetime.datetime.now()).strftime('%H:%M:%S')
    print(c_time)
    current_time = (datetime.datetime.now() + datetime.timedelta(minutes=5, seconds=30)).strftime('%H:%M:%S')
    next_time = (datetime.datetime.now() + datetime.timedelta(minutes=6)).strftime('%H:%M:%S')
    pos = pos.between_time(current_time, next_time)

    kal_m = LinearKalman(state_init=1, P_init=1, R_init=1, Q_init=2)
    kal_m.compute_kalman_gps_pos(gps_data=pos[["ref_pos_x (m)", "ref_pos_y (m)"]])

    print(pos.head(5))


def main_2() -> None:
    vect_in = [3, 5, 0.5, 0.75]
    Q_m = np.eye(4)
    P_m = 5 * np.eye(4)
    # F_mat = pdc.compute_F_matrix_linear_kalman(dt=0.25)
    # u_com = pdc.compute_command_vector_linear_kalman(spd=2, theta= np.deg2rad(15))
    # state_out, p_out = pdc.compute_prediction_linear_kalman(state_vector=vect_in, P_mat=P_m, spd=1,
    #                                                         theta=np.deg2rad(15), dt=.20, Q_mat=Q_m)
    #
    # R_m = 2 * np.eye(4)
    # meas = [1.5, 2.5, 0.1, 0.5]
    # pdc.compute_update_linear_kalman_gps(state_vector=state_out, meas_vector=meas, R_mat=R_m, P_mat=p_out)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
