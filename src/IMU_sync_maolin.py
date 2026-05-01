#!/usr/bin/env python3
"""
IMU Data Synchronization Script

What this script does:
1. Parse alignment.json
2. Read Xsens IMU txt files without pandas
3. Compute packet_sync_num from the first sensor file in each segment
4. Trim the beginning of every sensor file using effective packet timing
5. Convert PacketCounter into a continuous counter
   - raw wrapped counter: 65535 -> 0 stays continuous
   - already modified counter with values > 65535 also stays continuous
6. Interpolate all missing packets inside the remaining range
7. Save each sensor as .npz with the same base filename and folder structure

Output keys per sensor file:
- sensor_name
- rate_hz
- packet_counter_modulus
- packet_counter
- clip_id
- id
- is_interpolated
- acc
- freeacc
- gyr
- mag
- pressure
- quat

Usage:
    python IMU_sync_maolin.py --alignment_json alignment.json --imu_path IMU --output_folder IMU_Sync
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np


PACKET_COUNTER_MODULUS = 65536
IMU_RATE_HZ = 40
NUM_COLUMNS = 18
FLOAT_DTYPE = np.float32
INT_DTYPE = np.int32


class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.log_file = open(log_path, 'w', encoding='utf-8')

    def log(self, message):
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()

    def close(self):
        self.log_file.close()


logger = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Synchronize Xsens IMU txt files and export npz files.'
    )
    parser.add_argument('--alignment_json', type=Path, help='Path to alignment.json')
    parser.add_argument('--imu_path', type=Path, help='Path to IMU folder')
    parser.add_argument(
        '--output_folder',
        type=Path,
        nargs='?',
        default=None,
        help='Optional output folder. Default: IMU_New beside the IMU folder.'
    )
    return parser.parse_args()


def parse_alignment_json(alignment_path):
    with open(alignment_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    alignment_data = {}

    if 'participants' in data:
        for participant_obj in data['participants']:
            participant_name = participant_obj['name']
            alignment_data[participant_name] = {}

            if 'segments' in participant_obj:
                for segment_obj in participant_obj['segments']:
                    segment_num = str(segment_obj['segment'])
                    alignment_integral = segment_obj.get('alignment_integral(seconds)', 0)
                    alignment_data[participant_name][segment_num] = alignment_integral

    return alignment_data


def read_txt_sensor_file(txt_path):
    rows = []

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('//'):
                continue
            if 'PacketCounter' in line:
                continue

            parts = [part.strip() for part in line.split(',')]
            if len(parts) != NUM_COLUMNS:
                continue
            rows.append(parts)

    if len(rows) == 0:
        return {
            'packet_counter_raw': np.zeros((0,), dtype=np.int64),
            'values': np.zeros((0, NUM_COLUMNS - 1), dtype=FLOAT_DTYPE),
        }

    packet_counter_raw = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
    values = np.asarray([[float(x) for x in row[1:]] for row in rows], dtype=FLOAT_DTYPE)

    return {
        'packet_counter_raw': packet_counter_raw,
        'values': values,
    }


def normalize_packet_counter(packet_counter_raw):
    if packet_counter_raw.size == 0:
        return np.zeros((0,), dtype=np.int64)

    continuous = np.zeros(packet_counter_raw.shape[0], dtype=np.int64)
    continuous[0] = int(packet_counter_raw[0])

    saw_value_above_modulus = bool(np.any(packet_counter_raw >= PACKET_COUNTER_MODULUS))

    if saw_value_above_modulus:
        for i in range(1, packet_counter_raw.shape[0]):
            current_value = int(packet_counter_raw[i])
            previous_value = int(continuous[i - 1])

            if current_value >= previous_value:
                continuous[i] = current_value
            else:
                wrapped_candidate = current_value + PACKET_COUNTER_MODULUS
                if wrapped_candidate >= previous_value:
                    continuous[i] = wrapped_candidate
                else:
                    k = (previous_value - current_value + PACKET_COUNTER_MODULUS - 1) // PACKET_COUNTER_MODULUS
                    continuous[i] = current_value + k * PACKET_COUNTER_MODULUS
    else:
        offset = 0
        previous_raw = int(packet_counter_raw[0])
        for i in range(1, packet_counter_raw.shape[0]):
            current_raw = int(packet_counter_raw[i])
            if current_raw < previous_raw:
                offset += PACKET_COUNTER_MODULUS
            continuous[i] = current_raw + offset
            previous_raw = current_raw

    return continuous


def get_effective_packet_count(continuous_packet_counter):
    if continuous_packet_counter.size == 0:
        return 0
    return int(continuous_packet_counter[-1] - continuous_packet_counter[0] + 1)


def calculate_packet_sync_num(alignment_integral, first_sensor_data):
    continuous = normalize_packet_counter(first_sensor_data['packet_counter_raw'])
    effective_packet_count = get_effective_packet_count(continuous)
    packet_sync_num = int(effective_packet_count - (alignment_integral - 1) * IMU_RATE_HZ)

    logger.log(
        '    Formula: {0} - ({1} - 1) * {2} = {3}'.format(
            effective_packet_count,
            alignment_integral,
            IMU_RATE_HZ,
            packet_sync_num,
        )
    )
    logger.log(
        '    Effective packet count: {0}'.format(effective_packet_count)
    )

    return packet_sync_num


def calculate_trim_row_count(continuous_packet_counter, effective_packets_to_trim):
    if effective_packets_to_trim <= 0 or continuous_packet_counter.size == 0:
        return 0

    elapsed_packets = 0

    for row_index in range(1, continuous_packet_counter.shape[0]):
        previous_packet = int(continuous_packet_counter[row_index - 1])
        current_packet = int(continuous_packet_counter[row_index])
        packet_delta = current_packet - previous_packet
        elapsed_packets += packet_delta

        if elapsed_packets >= effective_packets_to_trim:
            return row_index

    return int(continuous_packet_counter.shape[0])


def trim_sensor_data(sensor_data, packet_sync_num):
    packet_counter_raw = sensor_data['packet_counter_raw']
    values = sensor_data['values']

    if packet_counter_raw.size == 0:
        return sensor_data

    continuous = normalize_packet_counter(packet_counter_raw)

    if packet_sync_num > 0:
        trim_row_count = calculate_trim_row_count(continuous, packet_sync_num)
    else:
        trim_row_count = 0

    trimmed_packet_counter_raw = packet_counter_raw[trim_row_count:]
    trimmed_values = values[trim_row_count:]

    return {
        'packet_counter_raw': trimmed_packet_counter_raw,
        'values': trimmed_values,
    }


def normalize_quaternion(quat):
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        return quat.copy()
    return quat / norm


def slerp_quaternion(q0, q1, t):
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)

    dot = float(np.dot(q0, q1))

    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = max(min(dot, 1.0), -1.0)

    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return normalize_quaternion(result).astype(FLOAT_DTYPE)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)

    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    result = s0 * q0 + s1 * q1
    return normalize_quaternion(result).astype(FLOAT_DTYPE)


def interpolate_linear_vector(continuous_packet_counter, values, full_packet_counter, output_dim):
    output = np.zeros((full_packet_counter.shape[0], output_dim), dtype=FLOAT_DTYPE)
    x = continuous_packet_counter.astype(np.float64)
    x_full = full_packet_counter.astype(np.float64)

    for dim in range(output_dim):
        y = values[:, dim].astype(np.float64)
        output[:, dim] = np.interp(x_full, x, y).astype(FLOAT_DTYPE)

    return output


def interpolate_quaternion(continuous_packet_counter, quat_values, full_packet_counter):
    total_length = full_packet_counter.shape[0]
    output = np.zeros((total_length, 4), dtype=FLOAT_DTYPE)

    start_packet = int(full_packet_counter[0])
    index_map = {}
    for i in range(continuous_packet_counter.shape[0]):
        index_map[int(continuous_packet_counter[i]) - start_packet] = i

    observed_positions = sorted(index_map.keys())

    if len(observed_positions) == 0:
        return output

    if len(observed_positions) == 1:
        only_quat = normalize_quaternion(quat_values[0]).astype(FLOAT_DTYPE)
        output[:] = only_quat
        return output

    for pos in observed_positions:
        original_index = index_map[pos]
        output[pos] = normalize_quaternion(quat_values[original_index]).astype(FLOAT_DTYPE)

    for pair_index in range(len(observed_positions) - 1):
        left_pos = observed_positions[pair_index]
        right_pos = observed_positions[pair_index + 1]

        left_quat = output[left_pos]
        right_quat = output[right_pos]
        gap = right_pos - left_pos

        if gap <= 1:
            continue

        for step in range(1, gap):
            t = float(step) / float(gap)
            output[left_pos + step] = slerp_quaternion(left_quat, right_quat, t)

    return output


def build_npz_arrays(trimmed_sensor_data):
    packet_counter_raw = trimmed_sensor_data['packet_counter_raw']
    values = trimmed_sensor_data['values']

    if packet_counter_raw.size == 0:
        return {
            'packet_counter': np.zeros((0,), dtype=np.int64),
            'clip_id': np.zeros((0,), dtype=INT_DTYPE),
            'id': np.zeros((0,), dtype=INT_DTYPE),
            'is_interpolated': np.zeros((0,), dtype=bool),
            'acc': np.zeros((0, 3), dtype=FLOAT_DTYPE),
            'freeacc': np.zeros((0, 3), dtype=FLOAT_DTYPE),
            'gyr': np.zeros((0, 3), dtype=FLOAT_DTYPE),
            'mag': np.zeros((0, 3), dtype=FLOAT_DTYPE),
            'pressure': np.zeros((0, 1), dtype=FLOAT_DTYPE),
            'quat': np.zeros((0, 4), dtype=FLOAT_DTYPE),
        }

    continuous_packet_counter = normalize_packet_counter(packet_counter_raw)
    full_packet_counter = np.arange(
        int(continuous_packet_counter[0]),
        int(continuous_packet_counter[-1]) + 1,
        dtype=np.int64,
    )

    is_interpolated = np.ones(full_packet_counter.shape[0], dtype=bool)
    observed_positions = continuous_packet_counter - full_packet_counter[0]
    is_interpolated[observed_positions.astype(np.int64)] = False

    acc = interpolate_linear_vector(continuous_packet_counter, values[:, 0:3], full_packet_counter, 3)
    freeacc = interpolate_linear_vector(continuous_packet_counter, values[:, 3:6], full_packet_counter, 3)
    gyr = interpolate_linear_vector(continuous_packet_counter, values[:, 6:9], full_packet_counter, 3)
    mag = interpolate_linear_vector(continuous_packet_counter, values[:, 9:12], full_packet_counter, 3)
    pressure = interpolate_linear_vector(continuous_packet_counter, values[:, 12:13], full_packet_counter, 1)
    quat = interpolate_quaternion(continuous_packet_counter, values[:, 13:17], full_packet_counter)

    clip_id = np.zeros(full_packet_counter.shape[0], dtype=INT_DTYPE)
    local_id = np.arange(full_packet_counter.shape[0], dtype=INT_DTYPE)

    return {
        'packet_counter': full_packet_counter,
        'clip_id': clip_id,
        'id': local_id,
        'is_interpolated': is_interpolated,
        'acc': acc,
        'freeacc': freeacc,
        'gyr': gyr,
        'mag': mag,
        'pressure': pressure,
        'quat': quat,
    }


def save_sensor_npz(output_npz_path, sensor_name, sensor_arrays):
    np.savez_compressed(
        output_npz_path,
        sensor_name=np.asarray(sensor_name),
        rate_hz=np.asarray(IMU_RATE_HZ, dtype=INT_DTYPE),
        packet_counter_modulus=np.asarray(PACKET_COUNTER_MODULUS, dtype=INT_DTYPE),
        packet_counter=sensor_arrays['packet_counter'],
        clip_id=sensor_arrays['clip_id'],
        id=sensor_arrays['id'],
        is_interpolated=sensor_arrays['is_interpolated'],
        acc=sensor_arrays['acc'],
        freeacc=sensor_arrays['freeacc'],
        gyr=sensor_arrays['gyr'],
        mag=sensor_arrays['mag'],
        pressure=sensor_arrays['pressure'],
        quat=sensor_arrays['quat'],
    )


def process_segment(segment_path, alignment_integral, output_segment_path):
    txt_files = sorted(segment_path.glob('*.txt'))

    if not txt_files:
        logger.log('No txt files found in {0}'.format(segment_path))
        return

    output_segment_path.mkdir(parents=True, exist_ok=True)

    mtb_files = sorted(segment_path.glob('*.mtb'))
    for mtb_file in mtb_files:
        shutil.copy2(mtb_file, output_segment_path / mtb_file.name)
        logger.log('  Copied: {0}'.format(mtb_file.name))

    logger.log('  Step 1: Reading txt sensor files...')
    sensor_data_by_file = {}
    for txt_file in txt_files:
        sensor_data = read_txt_sensor_file(txt_file)
        sensor_data_by_file[txt_file] = sensor_data
        logger.log(
            '    Loaded: {0} ({1} rows)'.format(
                txt_file.name,
                sensor_data['packet_counter_raw'].shape[0],
            )
        )

    first_txt_file = txt_files[0]
    first_sensor_data = sensor_data_by_file[first_txt_file]
    logger.log('  Step 2: Calculating packet_sync_num from {0}...'.format(first_txt_file.name))
    packet_sync_num = calculate_packet_sync_num(alignment_integral, first_sensor_data)
    logger.log(
        '  Segment {0}: alignment_integral={1}, packet_sync_num={2}'.format(
            segment_path.name,
            alignment_integral,
            packet_sync_num,
        )
    )

    logger.log('  Step 3: Trimming, interpolating, and saving npz files...')
    for txt_file in txt_files:
        trimmed_sensor_data = trim_sensor_data(sensor_data_by_file[txt_file], packet_sync_num)
        sensor_arrays = build_npz_arrays(trimmed_sensor_data)
        output_npz_path = output_segment_path / '{0}.npz'.format(txt_file.stem)
        save_sensor_npz(output_npz_path, txt_file.stem, sensor_arrays)

        logger.log(
            '    Saved: {0} ({1} frames, {2} interpolated)'.format(
                output_npz_path.name,
                sensor_arrays['packet_counter'].shape[0],
                int(sensor_arrays['is_interpolated'].sum()),
            )
        )


def main():
    global logger

    args = parse_args()

    alignment_json_path = args.alignment_json
    imu_path = args.imu_path

    if not alignment_json_path.exists():
        raise FileNotFoundError('alignment.json not found at {0}'.format(alignment_json_path))

    if not imu_path.exists():
        raise FileNotFoundError('IMU folder not found at {0}'.format(imu_path))

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = imu_path.parent / 'IMU_sync_log_{0}.txt'.format(timestamp)
    logger = Logger(log_path)

    logger.log('=' * 60)
    logger.log('IMU Data Synchronization Script')
    logger.log('Started at: {0}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    logger.log('=' * 60)
    logger.log('Alignment JSON: {0}'.format(alignment_json_path))
    logger.log('IMU folder: {0}'.format(imu_path))
    logger.log('Log file: {0}'.format(log_path))
    logger.log('')

    logger.log('Parsing alignment.json...')
    alignment_data = parse_alignment_json(alignment_json_path)
    logger.log('Found {0} participants in alignment.json'.format(len(alignment_data)))
    logger.log('')

    if args.output_folder is not None:
        if args.output_folder.is_absolute():
            output_imu_path = args.output_folder
        else:
            output_imu_path = imu_path.parent / args.output_folder
    else:
        output_imu_path = imu_path.parent / 'IMU_Sync'

    output_imu_path.mkdir(exist_ok=True)
    logger.log('Output directory: {0}'.format(output_imu_path))
    logger.log('')

    participants_folder = imu_path / 'Participants'
    if not participants_folder.exists():
        logger.close()
        raise FileNotFoundError('Participants folder not found at {0}'.format(participants_folder))

    for participant_folder in sorted(participants_folder.glob('*')):
        if not participant_folder.is_dir():
            continue

        folder_name = participant_folder.name
        logger.log('\nProcessing folder: {0}'.format(folder_name))

        if '-' in folder_name:
            participant_name = folder_name.split('-')[0].lower()
        else:
            logger.log('  Skipping special case folder (no hyphen): {0}'.format(folder_name))
            continue

        matching_key = None
        for key in alignment_data.keys():
            if key.lower() == participant_name:
                matching_key = key
                break

        if matching_key is None:
            continue

        logger.log('  Matched with participant: {0}'.format(matching_key))
        participant_alignment = alignment_data[matching_key]

        for segment_folder in sorted(participant_folder.glob('*')):
            if not segment_folder.is_dir():
                continue

            segment_num = segment_folder.name
            alignment_integral = None
            for segment_key, align_val in participant_alignment.items():
                if segment_num in segment_key or str(segment_num) == str(segment_key):
                    alignment_integral = align_val
                    break

            if alignment_integral is None:
                continue

            logger.log('  Processing segment: {0}'.format(segment_num))
            output_participant_path = output_imu_path / folder_name
            output_segment_path = output_participant_path / segment_num
            process_segment(segment_folder, alignment_integral, output_segment_path)

    logger.log('')
    logger.log('=' * 60)
    logger.log('Processing complete! Output saved to: {0}'.format(output_imu_path))
    logger.log('Finished at: {0}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    logger.log('=' * 60)
    logger.close()


if __name__ == '__main__':
    main()
