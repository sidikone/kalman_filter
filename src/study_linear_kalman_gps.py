from matplotlib import pyplot as plt
import numpy as np
from include.data_loader import SimuData
from include.linear_kalman_filter import LinearKalman


def set_sensor_frequencies(fs_gps=25):
    return fs_gps


def set_init_cv(x=0, y=0, vx=.5, vy=.5):
    return np.array([x, y, vx, vy])


def __2_lc_matrix(mat, std_x, std_y):
    mat[0, 0] = std_x
    mat[1, 1] = std_y
    return mat


def __4_lc_matrix(mat, std_x, std_y, std_vx, std_vy):
    mat[0, 0] = std_x
    mat[1, 1] = std_y
    mat[2, 2] = std_vx
    mat[3, 3] = std_vy
    return mat


def set_P_init(P_x=100, P_y=100, P_vx=100, P_vy=100):
    mat = np.eye(4)
    return __4_lc_matrix(mat, P_x, P_y, P_vx, P_vy)


def set_Q_init(Q_x=1.5, Q_y=1.5, Q_vx=.5, Q_vy=.5):
    mat = np.eye(4)
    return __4_lc_matrix(mat, Q_x, Q_y, Q_vx, Q_vy)


def set_R_cv_pos(std_x=3, std_y=3):
    mat = np.eye(2)
    return __2_lc_matrix(mat, std_x, std_y)


def set_R_cv_gps(std_x=3, std_y=3, std_vx=.5, std_vy=.5):
    mat = np.eye(4)
    return __4_lc_matrix(mat, std_x, std_y, std_vx, std_vy)


def plot_data(gps_data, kalman_pos, kalman_gps, disp=True) -> None:
    if disp:
        fig, ax = plt.subplots()
        ax.plot(gps_data["ref_pos_x (m)"], gps_data["ref_pos_y (m)"], label="true positions")
        ax.plot(kalman_gps["x"], kalman_gps["y"], '--', label="kalman gps")
        ax.plot(kalman_pos["x"], kalman_pos["y"], '--', label="kalman position")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend()
        plt.show()


def comparison_of_evolution_model() -> None:
    gps_fs = set_sensor_frequencies()

    data_sim = SimuData(path="../data/simu_dataset_1")
    data_sim.set_gps_frequency(fs=gps_fs)
    pos, spd = data_sim.get_gps_data()

    vect_init = set_init_cv()
    P_init = set_P_init()
    Q_init = set_Q_init()

    R_pos_init = set_R_cv_pos()
    R_gps_init = set_R_cv_gps()

    kalman = LinearKalman(state_init=vect_init,
                          P_init=P_init,
                          Q_init=Q_init)

    ekf_pos, P_pos = kalman.compute_kalman_gps_pos(gps_data=pos[["ref_pos_x (m)", "ref_pos_y (m)"]],
                                                   R_init=R_pos_init)

    ekf_gps, P_gps = kalman.compute_kalman_gps(pos_data=pos[["ref_pos_x (m)", "ref_pos_y (m)"]],
                                               spd_data=spd[["ref_vel_x (m/s)", "ref_vel_y (m/s)"]],
                                               R_init=R_gps_init)

    plot_data(gps_data=pos,
              kalman_pos=ekf_pos,
              kalman_gps=ekf_gps)


def main() -> None:
    comparison_of_evolution_model()


if __name__ == "__main__":
    main()
