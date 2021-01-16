from include.data_loader import SimuData
from matplotlib import pyplot as plt


def set_sensor_frequencies(fs_imu=50, fs_gps=10):
    return fs_imu, fs_gps


def set_sensor_noise(std_accel=0.25, stg_gyro=0.15, std_pos=3, std_spd=0.5):
    return std_accel, stg_gyro, std_pos, std_spd


def plot_gps_pos(data, disp=True, multi_plot=False) -> None:
    if disp:
        fig, ax = plt.subplots()
        ax.plot(data["pos_x (m)"], data["pos_y (m)"], label="noisy positions")
        ax.plot(data["ref_pos_x (m)"], data["ref_pos_y (m)"], label="true positions")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend()
        if not multi_plot:
            plt.show()


def plot_gps_spd(data, disp=True, multi_plot=False) -> None:
    data_name = list(data)
    y_labels = ['vx (m/s²)', 'vy (m/s²)', 'vz (m/s²)', 'v_norm (m/s)']
    if disp:
        fig, ax = plt.subplots(4, 1, sharex=True)
        for ind, name in enumerate(data_name[4:]):
            ax[ind].plot(data[name], label="noisy data")

        for ind, name in enumerate(data_name[:4]):
            ax[ind].plot(data[name], label="ref data")
            ax[ind].set_ylabel(y_labels[ind])
            ax[ind].legend()
        if not multi_plot:
            plt.show()


def plot_imu_accel(data, disp=True, multi_plot=False) -> None:
    data_name = list(data)
    y_labels = ['ax (m/s²)', 'ay (m/s²)', 'az (m/s²)']
    if disp:
        fig, ax = plt.subplots(3, 1, sharex=True)
        for ind, name in enumerate(data_name[3:]):
            ax[ind].plot(data[name], label="noisy data")

        for ind, name in enumerate(data_name[:3]):
            ax[ind].plot(data[name], label="ref data")
            ax[ind].set_ylabel(y_labels[ind])
            ax[ind].legend()
        if not multi_plot:
            plt.show()


def plot_imu_gyro(data, disp=True, multi_plot=False) -> None:
    data_name = list(data)
    y_labels = ['gx (m/s²)', 'gy (m/s²)', 'gz (m/s²)']
    if disp:
        fig, ax = plt.subplots(3, 1, sharex=True)
        for ind, name in enumerate(data_name[3:]):
            ax[ind].plot(data[name], label="noisy data")

        for ind, name in enumerate(data_name[:3]):
            ax[ind].plot(data[name], label="ref data")
            ax[ind].set_ylabel(y_labels[ind])
            ax[ind].legend()
        if not multi_plot:
            plt.show()


def run_data_simulation() -> None:
    data_sim = SimuData(path="../data/simu_dataset_1")

    imu_fs, gps_fs = set_sensor_frequencies()
    data_sim.set_imu_frequency(fs=imu_fs)
    data_sim.set_gps_frequency(fs=gps_fs)

    accel_std, gyro_std, pos_std, spd_std = set_sensor_noise()

    data_sim.set_accel_data_noise(std=accel_std)
    data_sim.set_gyro_data_noise(std=gyro_std)
    data_sim.set_gps_pos_data_noise(std=pos_std)
    data_sim.set_gps_spd_data_noise(std=spd_std)

    accel, gyro = data_sim.get_imu_data()
    pos, spd = data_sim.get_gps_data()

    plot_gps_pos(pos, disp=False)
    plot_gps_spd(spd, disp=False)
    plot_imu_accel(accel, disp=True, multi_plot=True)
    plot_imu_gyro(gyro, disp=True)


def main():
    run_data_simulation()


if __name__ == "__main__":
    main()
