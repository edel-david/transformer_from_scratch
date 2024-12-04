#!/usr/bin/env python3

import os
import random
import argparse

segment_start_marker = "<|SEGMENTSTART|>"
segment_end_marker = "<|SEGMENTEND|>"


def load_segments(file_path: str, start_marker: str, end_marker: str) -> list:
    """Load segments from a single file, splitting by segment markers."""

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Split based on segments, ignoring any empty splits
    segments = [
        f"{start_marker}\n{segment.strip()}"
        for segment in text.split(start_marker)
        if segment.strip()
    ]
    return segments


def split_data(
    segments: list,
    train_ratio: float = 0.8,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
) -> tuple:
    """Shuffle and split segments into train, validation, and test sets based on given ratios."""

    random.shuffle(segments)
    total = len(segments)

    # Calculate the number of segments for each set
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_segments = segments[:train_end]
    val_segments = segments[train_end:val_end]
    test_segments = segments[val_end:]

    return train_segments, test_segments, val_segments


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--start-marker", type=str, default=segment_start_marker)
    parser.add_argument("--end-marker", type=str, default=segment_end_marker)

    args = parser.parse_args()

    # Gather all segmented text files
    segmented_files = [f for f in os.listdir(args.input_dir) if f.endswith(".txt")]

    segments_all = []
    for file_name in segmented_files:
        file_path = os.path.join(args.input_dir, file_name)
        segments_all.extend(
            load_segments(file_path, args.start_marker, args.end_marker)
        )

    # Split the segments into train, test, and val sets
    segments_train, segments_test, segments_val = split_data(
        segments_all, args.train_ratio, args.test_ratio, args.val_ratio
    )

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Write the segments to the respective files

    with open(
        os.path.join(args.output_dir, "train.txt"), "w", encoding="utf-8"
    ) as train_file:
        train_file.write("\n".join(segments_train))

    with open(
        os.path.join(args.output_dir, "test.txt"), "w", encoding="utf-8"
    ) as test_file:
        test_file.write("\n".join(segments_test))

    with open(
        os.path.join(args.output_dir, "val.txt"), "w", encoding="utf-8"
    ) as val_file:
        val_file.write("\n".join(segments_val))
