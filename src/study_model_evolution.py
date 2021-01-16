from matplotlib import pyplot as plt
import numpy as np
from include.data_loader import SimuData
from include.evolution_model import batch_cv_integration, batch_ctrv_integration


def set_sensor_frequencies(fs_imu=25, fs_gps=100):
    return fs_imu, fs_gps


def set_init_cv(x=0, y=0, yaw=np.deg2rad(183), spd=0):
    return np.array([x, y, yaw, spd])


def set_init_ctrv(x=0, y=0, yaw=np.deg2rad(183), spd=0, y_rate=0):
    return np.array([x, y, yaw, spd, y_rate])


def plot_data(gps_data, cv_model, ctrv_model, disp=True) -> None:
    if disp:
        fig, ax = plt.subplots()
        ax.plot(gps_data["ref_pos_x (m)"], gps_data["ref_pos_y (m)"], label="true positions")
        ax.plot(cv_model["x"], cv_model["y"],  '--', label="cv model")
        ax.plot(ctrv_model["x"], ctrv_model["y"], '--', label="ctrv model")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend()
        plt.show()


def run_batch_cv_kalman(init, accel, gyro):
    em_cv = batch_cv_integration(state_init=init,
                                 imu_accel=accel['ref_accel_x (m/s^2)'],
                                 imu_gyro=np.deg2rad(gyro['ref_gyro_z (deg/s)']))
    return em_cv


def run_batch_ctrv_kalman(init, accel, gyro):
    em_ctrv = batch_ctrv_integration(state_init=init,
                                     imu_accel=accel['ref_accel_x (m/s^2)'],
                                     imu_gyro=np.deg2rad(gyro['ref_gyro_z (deg/s)']))
    return em_ctrv


def comparison_of_evolution_model() -> None:
    imu_fs, gps_fs = set_sensor_frequencies()

    data_sim = SimuData(path="../data/simu_dataset_1")
    data_sim.set_imu_frequency(fs=imu_fs)
    data_sim.set_gps_frequency(fs=gps_fs)
    accel, gyro = data_sim.get_imu_data()
    pos, _ = data_sim.get_gps_data()

    vect_init_cv = set_init_cv()
    vect_init_ctrv = set_init_ctrv()
    cv_model = run_batch_cv_kalman(init=vect_init_cv, accel=accel, gyro=gyro)
    ctrv_model = run_batch_ctrv_kalman(init=vect_init_ctrv, accel=accel, gyro=gyro)
    plot_data(pos, cv_model, ctrv_model, disp=False)


def main() -> None:
    comparison_of_evolution_model()


if __name__ == "__main__":
    main()
