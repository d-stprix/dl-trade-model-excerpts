import json
import os
import time
import math
import threading
from queue import Queue
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

import numpy as np
import cupy as cp
import pandas as pd

# Get reference to dataset files.
with open('dataset_paths.json', 'r') as file:
    file_paths = json.load(file)

INTRADAY_FEATHER_FILE = file_paths['INTRADAY_FEATHER_FILE']
CONST_FEATHER_FILE = file_paths['CONST_FEATHER_FILE']
BACKTEST_FEATHER_FILE = file_paths['BACKTEST_FEATHER_FILE']
DAILY_FEATHER_FILE = file_paths['DAILY_FEATHER_FILE']
NPZ_FILEPATH = file_paths['NPZ_FILEPATH']
NPZ_SAMPLE_FILEPATH = file_paths['NPZ_SAMPLE_FILEPATH']
NPZ_LARGER_SAMPLE_FILEPATH = file_paths['NPZ_LARGER_SAMPLE_FILEPATH']

INTRADAY_COLS = ['normalized opens', 'normalized closes', 'normalized lows', 'normalized volume',
                 'normalized highs', 'normalized macd', 'normalized signal', 'RSI', 'normalized vwap',
                 'normalized 200ema', 'normalized bollinger plus', 'normalized bollinger minus', 'normalized tenkan',
                 'normalized kijun', 'normalized senkou a', 'normalized senkou b', 'spy close']
DAILY_COLS = ['open', 'close', 'low', 'high', 'volume', 'spy close']

DISABLE_PROCESS_POOL = True  # Disabled due to bug causing failure with large datasets.
SEQUENCE_LENGTH = 128
DAILY_SEQUENCE_LENGTH = 128
TRAIN_SIZE = 0.8  # Proportion of data used for training.

if SEQUENCE_LENGTH >= 390:
    raise ValueError('Intraday sequence length should not exceed 389!')
if not math.log2(SEQUENCE_LENGTH).is_integer():
    raise ValueError('SEQUENCE_LENGTH must be a power of 2.')
if DAILY_SEQUENCE_LENGTH > 128:
    raise ValueError('Daily sequence length should not exceed 128!')

INTRADAY_Q, DAILY_Q = Queue(), Queue()

# GPU memory pool.
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

def get_valid_sequences():
    """Opens intraday file and determines which trades are valid and can be placed in dataset.

    Valid trades have at least SEQUENCE_LENGTH intraday candles. Each row of the feather files consist of one
    intraday/daily candle for each trade. The rows following the trades' last candle correspond to the first candles of
    the next trades. The sequence lengths are determined by reading the 'start' column of the respective feather files,
    which is a boolean value signalling whether this candle represents the first candle of a trade. By subtracting the
    indices between successive True values, each trade's corresponding sequence length can be determined.

    Returns:
        valid_sequences: Index array of trades that are considered valid (np.ndarray).
        intraday_indices: Index array pointing to the positions of valid intraday candles in the dataset (np.ndarray).
        daily_indices: Index array pointing to the positions of valid daily candles in the dataset (np.ndarray).
    """
    # Process intraday sequences.
    sequence_start = pd.read_feather(INTRADAY_FEATHER_FILE, columns=['start']).to_numpy()
    start_indices = np.flatnonzero(sequence_start)
    end_indices = np.append(start_indices[1:], sequence_start.size)
    print(f'{start_indices.size} trades')
    valid_sequences = np.flatnonzero(end_indices - start_indices >= SEQUENCE_LENGTH)
    print(f'{valid_sequences.size} valid trades')
    end_indices = end_indices[valid_sequences]
    intraday_indices = np.empty(end_indices.size*SEQUENCE_LENGTH, dtype=np.int64)
    start, end = 0, SEQUENCE_LENGTH
    for end_index in end_indices:
        intraday_indices[start:end] = np.arange(end_index-SEQUENCE_LENGTH, end_index)
        start, end = end, end + SEQUENCE_LENGTH

    # Process daily sequences.
    daily_start = pd.read_feather(DAILY_FEATHER_FILE, columns=['start']).to_numpy()
    start_indices = np.flatnonzero(daily_start)
    end_indices = np.append(start_indices[1:], daily_start.size)
    end_indices = end_indices[valid_sequences]
    daily_indices = np.full(end_indices.size*DAILY_SEQUENCE_LENGTH, 0, dtype=np.int64)
    start, end = 0, DAILY_SEQUENCE_LENGTH
    for end_index in end_indices:
        daily_indices[start:end] = np.arange(end_index-DAILY_SEQUENCE_LENGTH, end_index)
        start, end = end, end + SEQUENCE_LENGTH

    return valid_sequences, intraday_indices, daily_indices

def load_constant_data(valid_indices):
    """Loads constant data from corresponding feather file.

    Constant data refers to features where only one value exists per instance, e.g. symbol, strategy (MACD crossover,
    RSI, etc). The output array consists of constant features, followed by the profitability of each trade with a
    specific target/stoploss combination (labels).

    Args:
        valid_indices: The indices of the trades to be considered (np.ndarray).

    Returns:
        constant_data: The output array (np.ndarray).
        num_targets: The number of target/stoploss combinations, ie. the last num_targets columns represent labels
        (int).
    """
    print('Loading constant data.')

    strategy_id = {'strategy_id': pd.read_feather(CONST_FEATHER_FILE, columns=['strategy id']).to_numpy()[valid_indices]}
    one_hot_encode(strategy_id, ['strategy_id'])
    strategy_id = strategy_id['strategy_id']
    num_strategies = strategy_id.shape[1]
    original_constant_data = pd.read_feather(CONST_FEATHER_FILE)
    final_header = list(original_constant_data.columns)[-1]
    num_targets = int(''.join([char for char in final_header if char.isdigit()])) + 1

    original_constant_data = original_constant_data.to_numpy()[valid_indices]
    new_shape = (original_constant_data.shape[0], original_constant_data.shape[1]+num_strategies-1)
    constant_data = np.empty(new_shape, dtype=np.float32)
    constant_data[:, 0] = original_constant_data[:, 0]  # 'symbol'
    constant_data[:, 1:1+num_strategies] = strategy_id  # Add one-hot encoded strategy id.
    constant_data[:, 1+num_strategies:] = original_constant_data[:, 2:]  # Skip original strategy ids.
    return constant_data, num_targets

def load_backtest_data(valid_indices, year):
    """Loads data used in backtesting. Also adds the years as the last item in the array."""
    print('Loading backtest data.')
    backtest_data = pd.read_feather(BACKTEST_FEATHER_FILE).to_numpy()[valid_indices]
    year = year.reshape(-1, 1)  # Make as 2D
    backtest_data = np.concatenate([backtest_data, year + 2000], axis=1)
    return backtest_data

def background_load_files(intraday_indices, daily_indices):
    """Loads intraday and daily feather files in the background.

    Several GB of data must be loaded. To speed up the script, data is loaded before it is required.
    """
    t1 = threading.Thread(target=intraday_col_loader, args=(intraday_indices,))
    t2 = threading.Thread(target=daily_col_loader, args=(daily_indices,))
    t1.start()
    t2.start()

def daily_col_loader(daily_indices):
    """Loads and adds daily data to the daily queue."""
    for col in DAILY_COLS:
        col_data = pd.read_feather(DAILY_FEATHER_FILE, columns=[col]).to_numpy()[daily_indices].astype(np.float32)
        DAILY_Q.put(col_data)

def intraday_col_loader(intraday_indices):
    """Loads and adds intraday data to the intraday queue."""
    for col in INTRADAY_COLS:
        col_data = pd.read_feather(INTRADAY_FEATHER_FILE, columns=[col]).to_numpy()[intraday_indices].astype(np.float32)
        INTRADAY_Q.put(col_data)

def decompose_intraday_times(intraday_indices):
    """Separates integer_time data (YYMMDDhhmm) to day, month, day of month, and hour_min (hhmm) variables

    Obtains data by loading the integer time column of the intraday sequence file. Sets invalid hour_min values to
    NaN.

    Args:
        intraday_indices: Indices of the intraday feather file rows used (np.ndarray).

    Returns:
        hour_min: The hour_min values (np.ndarray).
        constant_time: A dict containing days, months, and days of month of the dates that each trade occurs
        (dict[str, np.ndarray]).
    """
    print('Decomposing intraday times.')
    integer_time = np.squeeze(pd.read_feather(INTRADAY_FEATHER_FILE, columns=['integer time']).to_numpy()[intraday_indices])
    hour_min = (integer_time % 1e4).astype(np.float32)
    invalid_hour_min = np.flatnonzero((hour_min == 0) | (hour_min % 1 != 0) | (hour_min == 929) | (hour_min % 100 >= 60))
    print(f'Number of invalid hour_min values: {len(invalid_hour_min)}')
    hour_min[invalid_hour_min] = np.nan
    end_indices = np.arange(SEQUENCE_LENGTH-1, integer_time.shape[0], SEQUENCE_LENGTH)
    const_times = integer_time[end_indices]
    const_day = (((const_times // 1e4) % 100) - 1).astype(np.float32)  # Subtract 1 to start at 0 for embedding layer.
    const_month = ((const_times // 1e6) % 100).astype(np.float32)
    const_day_month = ((const_times // 1e4) % 10000).astype(np.float32)
    year = const_times // 1e8
    constant_times = {'day': const_day, 'month': const_month, 'day_month': const_day_month}
    return year, hour_min, constant_times

def process_hour_min(hour_min):
    """Applies linear encoding to hour_min and replaces actual value with inverse (for embedding layer).

    Args:
        hour_min: The hour_min values (np.ndarray).

    Returns:
        hour_min: The inverse of the hour_min, i.e. the hour_min integer ID (np.ndarray).
        linear_hour_min: The linear encoding of the hour_min (np.ndarray).
    """
    unique_hour_min, inverse = np.unique(hour_min, return_inverse=True)
    hour_min = inverse.astype(np.float32)
    if np.isnan(unique_hour_min[-1]):  # Set invalid trade hour_mins to NaN - later drop trades.
        hour_min[hour_min == np.amax(inverse)] = np.nan
    linear_hour_min = (2 * hour_min / np.nanmax(hour_min)) - 1  # Value between -1 and 1.
    return hour_min, linear_hour_min


def decompose_daily_times(daily_indices):
    """Separates integer_day data (YYMMDD) to days, month and weekday variables.

    Obtains data by loading the integer day column of the daily sequence file. Uses CuPy to accelerate calculations
    with GPU.

    Args:
        daily_indices: Indices of the daily feather file rows used (np.ndarray).

    Returns:
        A dict containing days, months, days of month and weekdays corresponding to all datapoints in the daily
        dataset (dict[str, np.ndarray]).
    """
    print('Decomposing daily times.')
    integer_day = np.squeeze(pd.read_feather(DAILY_FEATHER_FILE, columns=['integer day']).to_numpy())[daily_indices]
    weekday = np.squeeze(pd.read_feather(DAILY_FEATHER_FILE, columns=['weekday']).to_numpy()[daily_indices])
    integer_day = cp.array(integer_day)
    day = ((integer_day % 100) - 1).astype(np.float32)  # Subtract 1 to start at 0 for embedding layer.
    day = cp.asnumpy(day)
    month = ((integer_day // 1e2) % 100).astype(np.float32)
    month = cp.asnumpy(month)
    day_month = (integer_day % 10000).astype(np.float32)
    day_month = cp.asnumpy(day_month)
    del integer_day
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    return {'day': day, 'month': month, 'day_month': day_month, 'weekday': weekday}

def cyclical_encode_date(day_month):
    """Applies cyclical encoding to the days of the month.

    The cyclical encoding uses sine and cosine transformations to map each unique day of the month to a point on the
    unit circle. Allows algorithms to understand the cyclicity of days.

    Args:
        day_month: An array containing days of the month for all datapoints (np.ndarray).

    Returns:
        unique_day_month: An array containing the unique days of the month; usually 0-31 (np.ndarray).
        sin_date: The sine transformation (np.ndarray)
        cos_date: The cosine transformation (np.ndarray)
    """
    print('Creating cyclical encodings for dates.')
    unique_day_month, inverse = np.unique(day_month, return_inverse=True)
    scaled_day_month = inverse / (len(unique_day_month) + 1)  # Value between 0 and 1.
    sin_date = np.sin(2 * math.pi * scaled_day_month)
    cos_date = np.cos(2 * math.pi * scaled_day_month)
    return unique_day_month, sin_date, cos_date

def update_constant_data(old_constant_data, const_times, unique_day_month):
    """Processes temporal information of the constant values and adds to constants array.

    Steps:
    1) One-hot encodes the month.
    2) Cyclically encodes the day of the month. Uses unique_day_month for scaling to ensure consistency with daily-
    candle processing.
    3) Allocates a new (larger) array for storing all constant values.
    4) Adds the processed temporal information to the start of the array and the rest of the data (unprocessed) to the
    end.

    Args:
        old_constant_data: The initial constant dataset (np.ndarray).
        const_times: A dict containing days, months, and days of month of the dates that each trade occurs
        (dict[str, np.ndarray]).
        unique_day_month: An array containing the unique days of the month; usually 0-31 (np.ndarray).

    Returns:
        constant_data: An array containing the constant dataset (np.ndarray).
    """
    print('Adding timestamp data to constant array.')
    one_hot_encode(const_times, ['month'])
    day_size = 1
    cyclical_day_month_size = 2
    month_size = const_times['month'].shape[1]
    time_size = day_size + cyclical_day_month_size + month_size

    # Cyclically encode day_month (of this particular trade)
    inverse = np.searchsorted(unique_day_month, const_times['day_month'])
    scaled_day_month = inverse / (len(unique_day_month) + 1)  # Value between 0 and 1.
    sin_date = np.sin(2 * math.pi * scaled_day_month)
    cos_date = np.cos(2 * math.pi * scaled_day_month)

    num_instances = old_constant_data.shape[0]
    num_features = old_constant_data.shape[1] + time_size
    constant_data = np.full((num_instances, num_features), np.nan, dtype=np.float32)

    constant_data[:, 0] = old_constant_data[:, 0]  # Symbol id. This must go first! (goes into embedding layer)
    constant_data[:, 1] = const_times['day']  # This must go second!
    constant_data[:, 2] = sin_date
    constant_data[:, 3] = cos_date
    constant_data[:, 4:4+month_size] = const_times['month']
    constant_data[:, 1+time_size:] = old_constant_data[:, 1:]
    return constant_data

def one_hot_encode(data, keys):
    """One-hot encodes the values in a dict. Replaces original dict value with output.

    Multiprocessing is disabled by default due to a bug which causes an error if the dataset is large. To enable, set
    DISABLE_PROCESS_POOL to False.

    Args:
        data: A dict containing values to one-hot encode (dict).
        keys: An iterable of keys indicating which items in the dict to one-hot encode (Iterable).
    """
    print(f'One hot encoding {keys}.')
    data_inputs = (data[key] for key in keys)

    if DISABLE_PROCESS_POOL:  # Process pool doesn't work if too much data to transfer.
        for key, data_input in zip(keys, data_inputs):
            data[key] = one_hot_process(key, data_input)[1]  # Returns a tuple with result in index 1.
    else:
        with ProcessPoolExecutor() as executor:
            results = executor.map(one_hot_process, keys, data_inputs)
        for key, result in results:
            data[key] = result

def one_hot_process(key, data):
    """One-hot encodes the data input.

    Args:
        key: The name of the corresponding dict key. This is returned unchanged for the calling function to identify
        the output; since the process pool executes non-deterministically (str).
        data: The data to one-hot encode (np.ndarray).

    Returns:
        key: The dict key (str).
        out: An array containing one-hot encoded values (np.ndarray).
    """
    num_items = data.shape[0]
    unique_items, inverse = np.unique(data, return_inverse=True)
    out = np.full((num_items, len(unique_items)), False, dtype=np.bool_)
    out[np.arange(num_items), inverse] = True
    return key, out

def create_daily_array(daily_times):
    """Creates an array containing the daily dataset.

    Steps:
    1) Preprocesses temporal data (adds second dimension).
    2) Calculates size of output array.
    3) Allocates output array.
    4) Adds temporal data to the beginning of the output array.
    5) Obtains all other daily data from the daily queue (which was updated asynchronously by background_load_files())
    and adds to output array.

    Args:
        daily_times: A dict containing days, months, days of month, and weekdays corresponding to all datapoints
        in the daily dataset (dict[str, np.ndarray]).

    Returns:
        daily_data: An array containing the daily dataset (np.ndarray).
    """
    print('Creating daily array.')
    arr_size = len(DAILY_COLS)  # +2 for cyclically encoded dates.

    del daily_times['day_month']  # Not used in final dataset.
    for key in ['day', 'sin_date', 'cos_date']:
        daily_times[key] = daily_times[key][:, np.newaxis]  # To have second dimension.

    for i, (key, time_var) in enumerate(daily_times.items()):
        if i == 0:
            num_items = time_var.shape[0]
        arr_size += time_var.shape[1]

    daily_data = np.full((num_items, arr_size), np.nan, dtype=np.float32)
    start, end = 0, 0
    for (key, time_var) in daily_times.items():
        end += time_var.shape[1]
        daily_data[:, start:end] = time_var
        start = end
    for i in range(len(DAILY_COLS)):
        print(f'processing {i}')
        col_data = DAILY_Q.get(block=True)
        end += 1
        daily_data[:, start:end] = col_data
        start = end
    print(f'Done daily data')
    print(daily_data.shape)
    return daily_data

def create_intraday_array(hour_min, linear_hour_min):
    """Creates an array containing the intraday dataset.

    Steps:
    1) Allocates output array.
    2) Adds temporal data to the beginning of the output array.
    3) Obtains all other intraday data from the intraday queue (which was updated asynchronously by
    background_load_files()) and adds to output array.

    Args:
        hour_min: The integer ID of each hour_min; where hour_min is hhmm (np.ndarray).
        linear_hour_min: The linear encoding of hour_min (np.ndarray).

    Returns:
        intraday_data: An array containing the intraday dataset (np.ndarray).
    """
    print('Creating intraday array.')
    arr_size = len(INTRADAY_COLS) + 2  # +2 for hour_min and linear_hour_min
    num_items = hour_min.shape[0]

    intraday_data = np.full((num_items, arr_size), np.nan, dtype=np.float32)
    intraday_data[:, 0] = hour_min
    intraday_data[:, 1] = linear_hour_min

    start, end = 2, 2
    for i in range(len(INTRADAY_COLS)):
        print(f'processing {i}')
        col_data = INTRADAY_Q.get(block=True)
        print('obtained col data')
        end += 1
        intraday_data[:, start:end] = col_data
        print('added to array')
        start = end
    print('Done intraday day')
    print(intraday_data.shape)
    return intraday_data

def split_dataset(data):
    """Splits the dataset into training, validation and test sets and returns each set.

    TRAIN_SIZE is the proportion of the data used for training. The remaining instances are divided evenly among the
    validation and test sets.

    Args:
        data: The dataset (np.ndarray).

    Returns:
        The input dataset divided into training, validation and test sets (Tuple[np.ndarray, np.ndarray, np.ndarray]).
    """
    valid_size = (1 - TRAIN_SIZE) / 2  # Validation and test sets have the same size.
    train_end = math.floor(data.shape[0] * TRAIN_SIZE)
    valid_end = train_end + math.floor(data.shape[0] * valid_size)
    return data[:train_end], data[train_end:valid_end], data[valid_end:]

def save_arrays(intraday_data, daily_data, constant_data, backtest_data, num_targets):
    """Saves datasets to NPZ files.

    Creates three files; with the first containing the entire dataset, the second comprising the first 10k
    instances, and the last comprising the first 100k instances. Begins by splitting the datasets between training,
    validation. The files are then saved to the specified filepaths.

    Args:
        intraday_data: The intraday dataset (np.ndarray).
        daily_data: The daily dataset (np.ndarray).
        constant_data: The constants dataset (np.ndarray).
        backtest_data: The backtest dataset(np.ndarray).
        num_targets: The number of target/stoploss combinations (int).
    """
    print('Creating main dataset.')
    train_sequence, valid_sequence, test_sequence = split_dataset(intraday_data)
    train_daily, valid_daily, test_daily = split_dataset(daily_data)
    train_constant, valid_constant, test_constant = split_dataset(constant_data)
    train_backtest, valid_backtest, test_backtest = split_dataset(backtest_data)

    num_targets = np.array([num_targets])
    print('Saving binary file.')
    np.savez(NPZ_FILEPATH, train_sequence=train_sequence, valid_sequence=valid_sequence, test_sequence=test_sequence,
             train_daily=train_daily, valid_daily=valid_daily, test_daily=test_daily, train_constant=train_constant,
             valid_constant=valid_constant, test_constant=test_constant, train_backtest=train_backtest,
             valid_backtest=valid_backtest, test_backtest=test_backtest, num_targets=num_targets)
    print('Creating sample dataset.')
    train_sequence, valid_sequence, test_sequence = split_dataset(intraday_data[:10000])
    train_daily, valid_daily, test_daily = split_dataset(daily_data[:10000])
    train_constant, valid_constant, test_constant = split_dataset(constant_data[:10000])
    train_backtest, valid_backtest, test_backtest = split_dataset(backtest_data[:10000])
    print('Saving binary file.')
    np.savez(NPZ_SAMPLE_FILEPATH, train_sequence=train_sequence, valid_sequence=valid_sequence, test_sequence=test_sequence,
             train_daily=train_daily, valid_daily=valid_daily, test_daily=test_daily, train_constant=train_constant,
             valid_constant=valid_constant, test_constant=test_constant, train_backtest=train_backtest,
             valid_backtest=valid_backtest, test_backtest=test_backtest, num_targets=num_targets)

    print('Creating larger sample dataset.')
    train_sequence, valid_sequence, test_sequence = split_dataset(intraday_data[:100000])
    train_daily, valid_daily, test_daily = split_dataset(daily_data[:100000])
    train_constant, valid_constant, test_constant = split_dataset(constant_data[:100000])
    train_backtest, valid_backtest, test_backtest = split_dataset(backtest_data[:100000])
    print('Saving binary file.')
    np.savez(NPZ_LARGER_SAMPLE_FILEPATH, train_sequence=train_sequence, valid_sequence=valid_sequence, test_sequence=test_sequence,
             train_daily=train_daily, valid_daily=valid_daily, test_daily=test_daily, train_constant=train_constant,
             valid_constant=valid_constant, test_constant=test_constant, train_backtest=train_backtest,
             valid_backtest=valid_backtest, test_backtest=test_backtest, num_targets=num_targets)

def load_arrays(sample, evaluation):
    """Loads the dataset into memory (not including backtesting data).

    Args:
        sample: The dataset ID. 0 for the full dataset, 1 for the first 100k instances, and 2 for the first 10k
        instances (int).
        evaluation: A boolean indicating whether this set is for training or testing. If True, the test set is returned,
        otherwise the training and validation sets are returned (bool).

    Returns:
        data: The requested dataset (dict[np.ndarray]).
        num_targets: The number of target/stoploss combinations (int).
    """
    print('Loading dataset...')
    if sample == 0:
        binary_file = np.load(NPZ_FILEPATH)
    elif sample == 1:
        binary_file = np.load(NPZ_LARGER_SAMPLE_FILEPATH)
    elif sample == 2:
        binary_file = np.load(NPZ_SAMPLE_FILEPATH)
    if evaluation:
        data = {'test_sequence': binary_file['test_sequence'], 'test_daily': binary_file['test_daily'],
                'test_constant': binary_file['test_constant']}
    else:
        data = {'train_sequence': binary_file['train_sequence'], 'train_daily': binary_file['train_daily'],
               'train_constant': binary_file['train_constant'], 'valid_sequence': binary_file['valid_sequence'],
               'valid_daily': binary_file['valid_daily'], 'valid_constant': binary_file['valid_constant']}
    num_targets = binary_file['num_targets'][0]
    return data, num_targets

def get_backtest_data(sample=0):
    """Loads and returns the backtesting dataset.

    Used when running backtests. Concatenates the training, validation and test sets and indicates the start index
    of each set. The backtest arrays provide information on the overlap between trades for the backtesting algorithm
    to be able to limit trading based on limited buying power when running simulations.

    Args:
        sample: The dataset ID. 0 for the full dataset, 1 for the first 100k instances, and 2 for the first 10k
        instances (int).

    Returns:
        backtest_items: An array containing information used in backtesting (np.ndarray).
        start_indices: A dict containing the start indices of the training, validation and test sets (dict[str, int])
    """
    if sample == 0:
        binary_file = np.load(NPZ_FILEPATH)
    elif sample == 1:
        binary_file = np.load(NPZ_LARGER_SAMPLE_FILEPATH)
    elif sample == 2:
        binary_file = np.load(NPZ_SAMPLE_FILEPATH)

    print('Loading backtest data...')
    train_backtest = binary_file['train_backtest']
    valid_backtest = binary_file['valid_backtest']
    test_backtest = binary_file['test_backtest']
    backtest_items = np.concatenate((train_backtest, valid_backtest, test_backtest), axis=0)

    train_length = train_backtest.shape[0]
    valid_length = valid_backtest.shape[0]

    start_indices = {'train': 0, 'validation': train_length, 'test': train_length+valid_length}
    return backtest_items, start_indices

def get_constant_data(sample=0):
    """Loads and returns the constants dataset.

    Used when running backtests. Concatenates the training, validation and test sets.

    Args:
        sample: The dataset ID. 0 for the full dataset, 1 for the first 100k instances, and 2 for the first 10k
        instances (int).

    Returns:
         constant_items: An array containing the constants dataset (np.ndarray).
         num_targets: The number of target/stoploss combinations (int).
    """
    if sample == 0:
        binary_file = np.load(NPZ_FILEPATH)
    elif sample == 1:
        binary_file = np.load(NPZ_LARGER_SAMPLE_FILEPATH)
    elif sample == 2:
        binary_file = np.load(NPZ_SAMPLE_FILEPATH)

    print('Loading constant data...')
    train_constant = binary_file['train_constant']
    valid_constant = binary_file['valid_constant']
    test_constant = binary_file['test_constant']
    constant_items = np.concatenate((train_constant, valid_constant, test_constant), axis=0)

    num_targets = binary_file['num_targets'][0]
    return constant_items, num_targets

def get_intraday_data(sample=0):
    """Loads and returns the intraday dataset.

    Used when running backtests. Concatenates the training, validation and test sets.

    Args:
        sample: The dataset ID. 0 for the full dataset, 1 for the first 100k instances, and 2 for the first 10k
        instances (int).

    Returns:
        sequence_items: An array containing the intraday dataset (np.ndarray).
    """
    if sample == 0:
        binary_file = np.load(NPZ_FILEPATH)
    elif sample == 1:
        binary_file = np.load(NPZ_LARGER_SAMPLE_FILEPATH)
    elif sample == 2:
        binary_file = np.load(NPZ_SAMPLE_FILEPATH)

    print('Loading intraday sequence data...')
    train_sequence = binary_file['train_sequence']
    valid_sequence = binary_file['valid_sequence']
    test_sequence = binary_file['test_sequence']
    return np.concatenate((train_sequence, valid_sequence, test_sequence), axis=0)

def get_daily_data(sample=0):
    """Loads and returns the daily dataset.

    Used when running backtests. Concatenates the training, validation and test sets.

    Args:
        sample: The dataset ID. 0 for the full dataset, 1 for the first 100k instances, and 2 for the first 10k
        instances (int).

    Returns:
        daily_items: An array containing the intraday dataset (np.ndarray).
    """
    if sample == 0:
        binary_file = np.load(NPZ_FILEPATH)
    elif sample == 1:
        binary_file = np.load(NPZ_LARGER_SAMPLE_FILEPATH)
    elif sample == 2:
        binary_file = np.load(NPZ_SAMPLE_FILEPATH)

    print('Loading daily data...')
    train_daily = binary_file['train_daily']
    valid_daily = binary_file['valid_daily']
    test_daily = binary_file['test_daily']
    return np.concatenate((train_daily, valid_daily, test_daily), axis=0)

def create_datasets():
    """Creates the dataset for the intraday models.

    Reads trade data from feather files, performs tasks such as one-hot encoding and positional encoding, and saves the
    processed data to a binary file. Only trades with sequence_length intraday candles are included (e.g. if
    sequence_length = 128, then trades before 11:38 are excluded).
    """
    print('Creating dataset...')
    valid_indices, intraday_indices, daily_indices = get_valid_sequences()
    constant_data, num_targets = load_constant_data(valid_indices)
    background_load_files(intraday_indices, daily_indices)
    year, hour_min, const_time = decompose_intraday_times(intraday_indices)
    backtest_data = load_backtest_data(valid_indices, year)
    hour_min, linear_hour_min = process_hour_min(hour_min)
    daily_times = decompose_daily_times(daily_indices)
    unique_day_month, daily_times['sin_date'], daily_times['cos_date'] = cyclical_encode_date(daily_times['day_month'])
    constant_data = update_constant_data(constant_data, const_time, unique_day_month)
    one_hot_encode(daily_times, ['month', 'weekday'])
    daily_data = create_daily_array(daily_times)
    intraday_data = create_intraday_array(hour_min, linear_hour_min)
    daily_data = daily_data.reshape((-1, DAILY_SEQUENCE_LENGTH, daily_data.shape[1]))
    intraday_data = intraday_data.reshape((-1, SEQUENCE_LENGTH, intraday_data.shape[1]))
    # Drops trades with invalid intraday candles.
    # Did not create dedicated function to avoid duplicate arrays in memory.
    invalid_indices = np.any(np.isnan(intraday_data[:, :, 0]), axis=1)
    print(f'Dropping invalid indices ({np.count_nonzero(invalid_indices)} trades)')
    intraday_data = intraday_data[~invalid_indices]
    daily_data = daily_data[~invalid_indices]
    constant_data = constant_data[~invalid_indices]
    backtest_data = backtest_data[~invalid_indices]
    save_arrays(intraday_data, daily_data, constant_data, backtest_data, num_targets)

def get_dataset(sample=0, evaluation=False):
    """Loads the datasets from disk.

    Creates and saves a new dataset if none exists.
    """
    if not os.path.exists(NPZ_FILEPATH) or not os.path.exists(NPZ_SAMPLE_FILEPATH) or not \
            os.path.exists(NPZ_LARGER_SAMPLE_FILEPATH):
        create_datasets()
    return load_arrays(sample, evaluation)
