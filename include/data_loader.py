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
        self.__sampling_fs = None
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
        self.__sampling_fs = sampling_fs

        self.__data_max_len = len(pd.read_csv(local_path))
        self.__timestamp = self.__timestamp_data()

        self.__gps_pos_data = pd.read_csv(path + "/ref_pos.csv")
        self.__add_timestamp_to_data(self.__gps_pos_data, self.__timestamp)
        self.__gps_spd_data = pd.read_csv(path + "/ref_vel.csv")
        self.__add_timestamp_to_data(self.__gps_spd_data, self.__timestamp)
        self.__accel_data = pd.read_csv(local_path)
        self.__add_timestamp_to_data(self.__accel_data, self.__timestamp)
        self.__gyro_data = pd.read_csv(path + "/ref_gyro.csv")
        self.__add_timestamp_to_data(self.__gyro_data, self.__timestamp)

    def __timestamp_data(self):
        time_index = pd.date_range(datetime.datetime.now(), periods=self.__data_max_len,
                                   freq=str(self.__sampling_dt) + 'S', name="timestamp")
        return time_index

    @staticmethod
    def __resampling_data(data, fs_in, fs_out):
        return data.iloc[range(0, len(data), int(fs_in / fs_out))]

    @staticmethod
    def __add_timestamp_to_data(data, timestamp) -> None:
        data['Timestamp'] = timestamp.values
        data.set_index('Timestamp', inplace=True)

    def __add_noise(self, data, std):

        cols_name = data.columns
        cols_noise_name = [elt + '_noise' for elt in cols_name]
        data_noise = [self.__noise_definition(data=data[elt], std=std) for elt in cols_name]
        my_frame = pd.DataFrame.from_dict(dict(zip(cols_noise_name, data_noise)))
        return pd.concat([data, my_frame], axis=1)

    @staticmethod
    def __noise_definition(data, std):
        return data + np.random.normal(scale=np.sqrt(std), size=len(data))

    def get_imu_data(self, fs=None):
        data = ()
        if fs is None:
            data = (self.__accel_data, self.__gyro_data)

        else:
            self.__accel_data = self.__resampling_data(data=self.__accel_data, fs_in=self.__sampling_fs, fs_out=fs)
            self.__gyro_data = self.__resampling_data(data=self.__gyro_data, fs_in=self.__sampling_fs, fs_out=fs)
            data = (self.__accel_data, self.__gyro_data)
        return data

    def get_gps_data(self, fs=None):
        data = ()
        if fs is None:
            data = (self.__accel_data, self.__gyro_data)

        else:
            self.__gps_pos_data = self.__resampling_data(data=self.__gps_pos_data, fs_in=self.__sampling_fs, fs_out=fs)
            self.__gps_spd_data = self.__resampling_data(data=self.__gps_spd_data, fs_in=self.__sampling_fs, fs_out=fs)
            data = (self.__accel_data, self.__gyro_data)
        return data

    def set_accel_data_noise(self, std=0):
        return self.__add_noise(self.__accel_data, std=std)

    def set_gyro_data_noise(self, std=0):
        return self.__add_noise(self.__gyro_data, std=std)

    def set_gps_pos_data_noise(self, std=0):
        return self.__add_noise(self.__gps_pos_data, std=std)

    def set_gps_spd_data_noise(self, std=0):
        return self.__add_noise(self.__gps_spd_data, std=std)