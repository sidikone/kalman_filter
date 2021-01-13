import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from include.evolution_model import batch_cv_integration, plot_ref_gps
from include.data_loader import SimuData


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

    # em_cv = batch_cv_integration(state_init=init,
    #                              imu_accel=accel['ref_accel_x (m/s^2)'],
    #                              imu_gyro=np.deg2rad(gyro['ref_gyro_z (deg/s)']))
    #
    # ax.plot(em_cv['x'], em_cv['y'], '--', label='CV model')


def main() -> None:
    data_sim = SimuData(path="data/simu_dataset_1", sampling_fs=100)
    accel, gyro = data_sim.get_imu_data(fs=5)
    kaw = data_sim.set_accel_data_noise(std=2)
    print(kaw.head(5))

    kaw = data_sim.set_gyro_data_noise(std=2)
    print(kaw.head(5))

    kaw = data_sim.set_gps_pos_data_noise(std=2)
    print(kaw.head(5))

    kaw = data_sim.set_gps_spd_data_noise(std=2)
    print(kaw.head(5))


    # fig, ax = plt.subplots()
    # plot_ref_gps(ax=ax)
    # fig2, ax2 = plt.subplots()
    # run_batch_cv_kalman(ax=ax2)
    # ax.legend()
    # plt.show()


# def main() -> None:
#     my_data = pd.DataFrame({"accel" : [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, -3, -3, 0, 0]})
#     my_data["speed"] = my_data.values*0.1
#     my_data["pos"] = my_data["speed"].cumsum()
#     print(my_data)
#     fig, ax = plt.subplots()
#     ax.plot(my_data["pos"] + 3, '.-')
#     ax.legend()
#     plt.show()


#
#
#     def add_noise(self, std):
#
#         self._noise_trig = True
#         self.noise = std
# #        self.noise_signal = self._signal + std * np.random.randn(len(self._time))
#         self.noise_signal = self._signal + np.random.normal(scale=np.sqrt(std), size=len(self._time))
#         return None
#
#     def _timestamp_data(self):
#
#         time_index = pd.date_range(
#             datetime.datetime.now(), periods=len(self._signal), freq=str(self._sampling_period)+'S', name="timestamp"
#         )
#         return time_index
#
#     def get_data_into_pandas_format(self):
#         time_index = self._timestamp_data()
#         self.dataFrame = pd.DataFrame({'Timestamp': time_index.values, 'raw_data': self._signal})
#         self.dataFrame = self.dataFrame.set_index('Timestamp')
#
#         if self._noise_trig:
#             self.dataFrame['noise_data'] = self.noise_signal
#
#         return self.dataFrame

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
