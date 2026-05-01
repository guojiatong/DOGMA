#!/usr/bin/env python3
"""
IMU Data Synchronization Script
Processes IMU txt files based on alignment data and converts to pkl format.

Usage:
    python IMU_sync.py /path/to/alignment.json /path/to/IMU [output_folder]

If output_folder is omitted, the script writes to IMU_New beside the IMU folder.
"""

import json
import os
import sys
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime


# Set up logging
class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.log_file = open(log_path, 'w', encoding='utf-8')
        
    def log(self, message):
        """Print to console and write to log file"""
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()


logger = None
PACKET_COUNTER_MODULUS = 65536


def parse_alignment_json(alignment_path):
    """
    Parse alignment.json to extract alignment_integral values.
    
    Returns:
        dict: {participant: {segment: alignment_integral}}
    """
    with open(alignment_path, 'r') as f:
        data = json.load(f)
    
    alignment_data = {}
    
    # Handle the actual JSON structure with "participants" array
    if "participants" in data:
        for participant_obj in data["participants"]:
            participant_name = participant_obj["name"]
            alignment_data[participant_name] = {}
            
            if "segments" in participant_obj:
                for segment_obj in participant_obj["segments"]:
                    segment_num = str(segment_obj["segment"])
                    # Use the field name with parentheses
                    alignment_integral = segment_obj.get("alignment_integral(seconds)", 0)
                    alignment_data[participant_name][segment_num] = alignment_integral
    
    return alignment_data


def extract_packet_numbers_from_dataframe(df):
    """
    Extract PacketCounter values from the first column of the processed pkl file.
    """
    if df.empty:
        return []

    return pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().astype(int).tolist()


def calculate_trim_row_count(packet_numbers, effective_packets_to_trim):
    """
    Convert an effective packet trim count into a row trim count by walking the
    PacketCounter timeline, including missing packets implied by discontinuities.

    Returns:
        int: number of rows to remove from the beginning
    """
    if effective_packets_to_trim <= 0 or not packet_numbers:
        return 0

    elapsed_packets = 0

    for row_index, (previous_packet, current_packet) in enumerate(
        zip(packet_numbers, packet_numbers[1:]),
        start=1
    ):
        packet_delta = (current_packet - previous_packet) % PACKET_COUNTER_MODULUS
        elapsed_packets += packet_delta if packet_delta > 0 else 1

        if elapsed_packets >= effective_packets_to_trim:
            return row_index

    return len(packet_numbers)


def get_effective_packet_count_from_pkl(pkl_path):
    """
    Count total packets for timing by combining recorded rows with dropped
    packets inferred from PacketCounter discontinuities.
    
    Args:
        pkl_path: Path to the pkl file
    
    Returns:
        int: effective packet count used for synchronization timing
    """
    try:
        df = pd.read_pickle(pkl_path)

        if df.empty:
            logger.log(f"      WARNING: {pkl_path.name} is empty")
            return 0

        packet_numbers = extract_packet_numbers_from_dataframe(df)

        if not packet_numbers:
            logger.log(f"      WARNING: No valid PacketCounter values found in {pkl_path.name}")
            return 0

        recorded_packets = len(packet_numbers)
        missing_packets = 0
        discontinuity_count = 0
        discontinuity_examples = []

        for previous_packet, current_packet in zip(packet_numbers, packet_numbers[1:]):
            packet_delta = (current_packet - previous_packet) % PACKET_COUNTER_MODULUS

            if packet_delta > 1:
                missing_between_packets = packet_delta - 1
                missing_packets += missing_between_packets
                discontinuity_count += 1

                if len(discontinuity_examples) < 5:
                    discontinuity_examples.append(
                        f"{previous_packet} -> {current_packet} (missing {missing_between_packets})"
                    )

        effective_packet_count = recorded_packets + missing_packets

        logger.log(f"      Counted {recorded_packets} recorded rows from {pkl_path.name}")
        logger.log(f"      First packet: {packet_numbers[0]}, Last packet: {packet_numbers[-1]}")

        if discontinuity_count:
            logger.log(
                f"      Detected {discontinuity_count} PacketCounter discontinuities "
                f"with {missing_packets} missing packets total"
            )
            for example in discontinuity_examples:
                logger.log(f"        Gap example: {example}")
            if discontinuity_count > len(discontinuity_examples):
                logger.log(
                    f"        ... plus {discontinuity_count - len(discontinuity_examples)} more discontinuities"
                )
        else:
            logger.log("      PacketCounter is continuous; no missing packets detected")

        logger.log(f"      Effective packet count for timing: {effective_packet_count}")
        return effective_packet_count
    except Exception as e:
        logger.log(f"      ERROR reading {pkl_path.name}: {e}")
        return 0


def calculate_packet_sync_num(alignment_integral, pkl_files):
    """
    Calculate packet_sync_num based on formula:
    effective_packet_count - (alignment_integral - 1) * 40
    
    Args:
        alignment_integral: float, from alignment.json
        pkl_files: list of paths to pkl files
    
    Returns:
        int: packet_sync_num (can be negative)
    """
    if not pkl_files:
        logger.log(f"    WARNING: No pkl files provided!")
        return 0
    
    # Use first pkl file (all files have same number of packets)
    pkl_file = pkl_files[0]
    logger.log(f"    Reading PacketCounter continuity from: {pkl_file.name} (using first file only)")
    
    # Count total packets from pkl file, including dropped packets inferred from gaps
    effective_packet_count = get_effective_packet_count_from_pkl(pkl_file)
    
    # Apply the existing trim formula with the effective packet count
    packet_sync_num = int(effective_packet_count - (alignment_integral - 1) * 40)
    
    # Log details
    logger.log(
        f"    Formula: {effective_packet_count} - ({alignment_integral} - 1) * 40 = {packet_sync_num}"
    )
    logger.log(
        f"    Effective packet count: {effective_packet_count} "
        f"(all {len(pkl_files)} sensor files assumed to share the same timing)"
    )
    
    # Log if negative
    if packet_sync_num < 0:
        logger.log(f"    WARNING: packet_sync_num is NEGATIVE ({packet_sync_num})")
    
    return packet_sync_num


def remove_rows_from_pkl(pkl_path, packet_sync_num):
    """
    Remove the leading rows needed to cover packet_sync_num effective packets,
    including time lost to missing PacketCounter values.
    
    Args:
        pkl_path: Path to the pkl file
        packet_sync_num: Number of effective packets to remove from the beginning
    """
    try:
        # Read the pkl file
        df = pd.read_pickle(pkl_path)
        original_rows = len(df)
        packet_numbers = extract_packet_numbers_from_dataframe(df)
        
        # Remove the prefix that covers packet_sync_num effective packets if positive
        if packet_sync_num > 0:
            trim_row_count = calculate_trim_row_count(packet_numbers, packet_sync_num)

            if trim_row_count < len(df):
                df = df.iloc[trim_row_count:]
                logger.log(
                    f"      Removed {trim_row_count} rows from {pkl_path.name} "
                    f"to cover {packet_sync_num} effective packets ({original_rows} -> {len(df)} rows)"
                )
            else:
                logger.log(
                    f"      WARNING: packet_sync_num ({packet_sync_num}) consumes the full file "
                    f"({len(df)} rows) in {pkl_path.name}"
                )
                df = pd.DataFrame(columns=df.columns)  # Empty dataframe
        elif packet_sync_num < 0:
            logger.log(f"      packet_sync_num is negative ({packet_sync_num}), keeping all {original_rows} rows in {pkl_path.name}")
        else:
            logger.log(f"      packet_sync_num is 0, keeping all {original_rows} rows in {pkl_path.name}")
        
        # Reset index and save back
        df = df.reset_index(drop=True)
        df.to_pickle(pkl_path)
        
    except Exception as e:
        logger.log(f"      ERROR processing {pkl_path.name}: {e}")


def process_txt_file(txt_path, output_pkl_path):
    """
    Process a single txt file:
    1. Remove comment rows (starting with //)
    2. Skip header row (contains "PacketCounter") - don't include in data
    3. Keep all columns from data rows only
    4. Save as pkl file (without row removal - that happens later)
    """
    data_rows = []
    
    # Read file and filter out comments, empty lines, and header
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments, empty lines, and header row
            if line.startswith('//') or not line or 'PacketCounter' in line:
                continue
            data_rows.append(line)
    
    if len(data_rows) == 0:
        logger.log(f"Warning: No data in {txt_path}")
        return
    
    # Parse data and keep all columns
    all_data = []
    for line in data_rows:
        # Split by comma and keep all elements
        parts = [part.strip() for part in line.split(',')]
        all_data.append(parts)
    
    # Create DataFrame with all columns (no header)
    df = pd.DataFrame(all_data)
    
    # Save as pickle
    df.to_pickle(output_pkl_path)
    logger.log(f"Processed: {txt_path.name} -> {output_pkl_path.name} ({len(all_data)} rows, {len(all_data[0]) if all_data else 0} columns)")


def process_segment(segment_path, alignment_integral, output_segment_path):
    """
    Process all txt files in a segment folder.
    """
    # Get all txt files
    txt_files = list(segment_path.glob('*.txt'))
    
    if not txt_files:
        logger.log(f"No txt files found in {segment_path}")
        return
    
    # Create output directory
    output_segment_path.mkdir(parents=True, exist_ok=True)
    
    # Copy mtb file if exists
    mtb_files = list(segment_path.glob('*.mtb'))
    for mtb_file in mtb_files:
        shutil.copy2(mtb_file, output_segment_path / mtb_file.name)
        logger.log(f"  Copied: {mtb_file.name}")
    
    # Step 1: Process each txt file to create initial pkl files (no row removal yet)
    logger.log(f"  Step 1: Creating initial pkl files...")
    pkl_files = []
    for txt_file in txt_files:
        output_pkl_path = output_segment_path / f"{txt_file.stem}.pkl"
        process_txt_file(txt_file, output_pkl_path)
        pkl_files.append(output_pkl_path)
    
    # Step 2: Calculate packet_sync_num from the created pkl files
    packet_sync_num = calculate_packet_sync_num(alignment_integral, pkl_files)
    logger.log(f"  Segment {segment_path.name}: alignment_integral={alignment_integral}, packet_sync_num={packet_sync_num}")
    
    # Step 3: Apply row removal to all pkl files based on packet_sync_num
    if packet_sync_num != 0:
        logger.log(f"  Step 2: Applying row removal (packet_sync_num={packet_sync_num})...")
        for pkl_file in pkl_files:
            remove_rows_from_pkl(pkl_file, packet_sync_num)
    else:
        logger.log(f"  packet_sync_num is 0, no row removal needed")


def main():
    global logger
    
    if len(sys.argv) not in (3, 4):
        print("Usage: python IMU_sync.py /path/to/alignment.json /path/to/IMU [output_folder]")
        sys.exit(1)
    
    alignment_json_path = Path(sys.argv[1])
    imu_path = Path(sys.argv[2])
    
    if not alignment_json_path.exists():
        print(f"Error: alignment.json not found at {alignment_json_path}")
        sys.exit(1)
    
    if not imu_path.exists():
        print(f"Error: IMU folder not found at {imu_path}")
        sys.exit(1)
    
    # Initialize logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = imu_path.parent / f"IMU_sync_log_{timestamp}.txt"
    logger = Logger(log_path)
    
    logger.log("="*60)
    logger.log("IMU Data Synchronization Script")
    logger.log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("="*60)
    logger.log(f"Alignment JSON: {alignment_json_path}")
    logger.log(f"IMU folder: {imu_path}")
    logger.log(f"Log file: {log_path}")
    logger.log("")
    
    # Parse alignment data
    logger.log("Parsing alignment.json...")
    alignment_data = parse_alignment_json(alignment_json_path)
    logger.log(f"Found {len(alignment_data)} participants in alignment.json")
    logger.log("")
    
    # Create output directory
    if len(sys.argv) == 4:
        output_arg = Path(sys.argv[3])
        if output_arg.is_absolute():
            output_imu_path = output_arg
        else:
            output_imu_path = imu_path.parent / output_arg
    else:
        output_imu_path = imu_path.parent / "IMU_New"

    output_imu_path.mkdir(exist_ok=True)
    logger.log(f"Output directory: {output_imu_path}")
    logger.log("")
    
    # Navigate to IMU/Participants folder
    participants_folder = imu_path / "Participants"
    if not participants_folder.exists():
        logger.log(f"Error: Participants folder not found at {participants_folder}")
        logger.close()
        sys.exit(1)
    
    # Process each participant
    for participant_folder in sorted(participants_folder.glob('*')):
        if not participant_folder.is_dir():
            continue
        
        folder_name = participant_folder.name
        logger.log(f"\nProcessing folder: {folder_name}")
        
        # Extract participant name from folder name
        # Format is either "dogname" (special case like Bella) or "participant-dogname"
        if '-' in folder_name:
            participant_name = folder_name.split('-')[0].lower()
        else:
            # Special case like "Bella" - skip for now
            logger.log(f"  Skipping special case folder (no hyphen): {folder_name}")
            continue
        
        # Get alignment data for this participant (exact match, case-insensitive)
        matching_key = None
        for key in alignment_data.keys():
            if key.lower() == participant_name:
                matching_key = key
                break
        
        if matching_key is None:
            # No alignment data found for this participant, skip silently
            continue
        
        logger.log(f"  Matched with participant: {matching_key}")
        participant_alignment = alignment_data[matching_key]
        
        # Process each segment
        for segment_folder in sorted(participant_folder.glob('*')):
            if not segment_folder.is_dir():
                continue
            
            segment_num = segment_folder.name
            
            # Find matching alignment data
            alignment_integral = None
            for segment_key, align_val in participant_alignment.items():
                if segment_num in segment_key or str(segment_num) == str(segment_key):
                    alignment_integral = align_val
                    break
            
            if alignment_integral is None:
                # No alignment data for this segment, skip silently
                continue
            
            logger.log(f"  Processing segment: {segment_num}")
            
            # Create output path
            # Keep the participant_dogname format in output
            output_participant_path = output_imu_path / folder_name
            output_segment_path = output_participant_path / segment_num
            
            # Process segment
            process_segment(segment_folder, alignment_integral, output_segment_path)
    
    logger.log("")
    logger.log("="*60)
    logger.log(f"Processing complete! Output saved to: {output_imu_path}")
    logger.log(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("="*60)
    logger.close()


if __name__ == "__main__":
    main()
