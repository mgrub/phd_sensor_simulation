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
        

def split_sensor_readings(sensor_readings, split_indices, n_time):
    blockwise_indices = np.split(np.arange(n_time), split_indices)
    
    split_sensor_readings = []

    for current_indices in blockwise_indices:
        block_sensor_readings = {}

        for sensor_name, sensor_data in sensor_readings.items():
            block_sensor_readings[sensor_name] = {}

            for series_name, series_array in sensor_data.items():
                block_sensor_readings[sensor_name][series_name] = [item for i, item in enumerate(series_array) if i in current_indices]

        split_sensor_readings.append(block_sensor_readings) 

    return split_sensor_readings