
import os
import datetime
import traceback

from dtaianomaly.data import LazyDataLoader
from dtaianomaly.pipeline import Pipeline


def log_error(error_log_path: str, exception: Exception, data_loader: LazyDataLoader, pipeline: Pipeline = None, fit_on_X_train: bool = True) -> str:

    # Ensure the directory exists
    os.makedirs(error_log_path, exist_ok=True)

    # Set an intuitive name for the error file based on the given data loader and pipeline
    base_file_name = data_loader.__class__.__name__
    if pipeline is not None:
        base_file_name += f'-{pipeline.detector.__class__.__name__}'

    # Ensure that the file name is unique
    while True:
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if not os.path.exists(f'{error_log_path}/{base_file_name}-{now}.err'):
            break
    file_path = f'{error_log_path}/{base_file_name}-{now}.err'

    # Write away the logging
    with open(file_path, 'w') as error_file:

        # Create an error message as a string
        if pipeline is None:  # Didn't reach anomaly detection when error occurred.
            error_message = 'An error occurred while loading data!'
        else:
            error_message = 'An error occurred while detecting anomalies!'
        error_message += '\nCode to reproduce the error is at the bottom of this error-log.\n\n'

        # Add the error message
        error_message += 'Traceback (most recent call last):\n\n'
        error_message += '\n'.join(traceback.format_tb(exception.__traceback__))
        error_message += f'\n{exception.__class__.__name__}: {exception}'

        # Make sure the message is in python comments
        error_message = '# ' + error_message.replace('\n', '\n# ')

        # Write the error message
        error_file.write(error_message)

        # Add an empty line below the error message
        error_file.write('\n\n')

        # Add the imports to the file
        error_file.write('import numpy as np\n')
        error_file.write('from dtaianomaly.data import *\n')
        if pipeline is not None:
            error_file.write('from dtaianomaly.preprocessing import *\n')
            error_file.write('from dtaianomaly.anomaly_detection import *\n')
            error_file.write('from dtaianomaly.pipeline import Pipeline\n')

        # Ad an empty line
        error_file.write('\n')

        # Add code for loading the data
        error_file.write(f'data_loader = {data_loader}\n')
        error_file.write(f'data = data_loader.load()\n\n')

        # Add code for detecting anomalies
        if pipeline is not None:
            error_file.write(f'preprocessor = {pipeline.preprocessor}\n')
            error_file.write(f'detector = {pipeline.detector}\n')
            error_file.write('pipeline = Pipeline(\n'
                             '    preprocessor=preprocessor,\n'
                             '    detector=detector\n'
                             ')\n')
            if fit_on_X_train:
                train_data = 'X_train'
            else:
                train_data = 'X_test'
            error_file.write(f'y_pred = pipeline.fit(data.{train_data}, data.y_train).predict_proba(data.X_test)\n\n')

    return os.path.abspath(file_path)
