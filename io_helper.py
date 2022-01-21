import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def current_block(sensor_reading, current_indices):
    current_sensor_readings = {}

    for sensor_name, sensor_data in sensor_reading.items():
        current_sensor_readings[sensor_name] = {}
        for series_name, series_array in sensor_data.items():
            current_sensor_readings[sensor_name][series_name] = [item for i, item in enumerate(series_array) if i in current_indices]

    return current_sensor_readings