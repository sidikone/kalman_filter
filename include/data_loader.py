import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime


class SimuData:
    def __init__(self, path, sampling_fs):
        self.__path = None
        self.__accel_data = None
        self.__gyro_data = None
        self.__gps_pos_data = None
        self.__gps_spd_data = None
        self.__timestamp = None

        self.__sampling_dt = None
        self.__imu_dt = None
        self.__gps_dt = None
        self.__data_max_len = None

        # Add noise to different data
        self.__gps_pos_std = 0
        self.__gps_spd_std = 0
        self.__accel_std = 0
        self.__gyro_std = 0

        self.__initialize(path=path, sampling_fs=sampling_fs)

    def __initialize(self, path, sampling_fs) -> None:
        local_path = path + "/ref_accel.csv"
        self.__sampling_dt = 1. / sampling_fs
        self.__data_max_len = len(pd.read_csv(local_path))
        self.__timestamp = self.__timestamp_data()

        self.__gps_pos_data = pd.read_csv(path + "/ref_gps.csv")
        self.__accel_data = pd.read_csv(local_path)
        self.__gyro_data = pd.read_csv(path + "/ref_gyro.csv")

    def __timestamp_data(self):
        time_index = pd.date_range(datetime.datetime.now(), periods=self.__data_max_len,
                                   freq=str(self.__sampling_dt) + 'S', name="timestamp")
        return time_index

    def get_imu_data(self, fs=None):
        if fs is None:
            self.__imu_dt = self.__sampling_dt

        return self.__accel_data, self.__gyro_data

    def get_gps_data(self, fs=None):
        if fs is None:
            self.__gps_dt = self.__sampling_dt

        return self.__gps_pos_data, self.__gps_spd_data
