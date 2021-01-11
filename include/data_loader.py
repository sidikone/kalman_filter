import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime


class SimuData:
    def __init__(self, path, sampling_fs, imu_fs, gps_fs):

        self.__path = None
        self.__accel_data = None
        self.__gyro_data = None
        self.__gps_data = None
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

        self.__initialize(path=path, sampling_fs=sampling_fs, imu_fs=imu_fs, gps_fs=gps_fs)

    def __initialize(self, path, sampling_fs, imu_fs, gps_fs) -> None:

        local_path = path + "/ref_accel.csv"
        self.__sampling_dt = 1. / sampling_fs
        self.__imu_dt = 1. / imu_fs
        self.__gps_dt = 1. / gps_fs
        self.__data_max_len = len(pd.read_csv(local_path))
        self.__timestamp = self.__timestamp_data()

    def __timestamp_data(self):
        time_index = pd.date_range(datetime.datetime.now(), periods=self.__data_max_len,
                                   freq=str(self.__sampling_dt) + 'S', name="timestamp")
        return time_index

    def get_ref_imu_data(self, fs=None):
        dt = 1. / fs
        data = None
        if dt is self.__imu_dt:
            data = pd.read_csv(self.__path)
        else:
            data = pd.read_csv(self.__path)

        return data
