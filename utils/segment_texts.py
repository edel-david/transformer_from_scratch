#!/usr/bin/env python3

import os
import argparse

segment_start_marker = "<|SEGMENTSTART|>"
segment_end_marker = "<|SEGMENTEND|>"


def mark_segments(text: str, start_marker: str, end_marker: str) -> str:

    # Split text into lines

    lines = text.strip().splitlines()
    text_segmented_list = [f"{start_marker}\n"]
    segment_open = True

    # Temporary storage for lines in the current segment
    line_buffer = []

    # Flag to track if we've encountered a multi-line block
    multi_line_detected = False

    for line in lines:
        # Add the current line to the buffer
        line_buffer.append(line + "\n")

        # Check if we have a non-empty line that indicates multi-line text

        if line.strip():
            # Mark that we've seen a multi-line block
            if multi_line_detected or len(line_buffer) > 2:
                multi_line_detected = True
        else:

            # If we encounter an empty line and have a multi-line block, end the segment
            if multi_line_detected:

                # Remove trailing empty lines from line_buffer
                while line_buffer and not line_buffer[-1].strip():
                    line_buffer.pop()

                text_segmented_list.extend(line_buffer)

                # Close and start a new segment
                text_segmented_list.append(f"{end_marker}\n{start_marker}\n")
                line_buffer.clear()

                # Keep segment open after restart
                segment_open = True

                # Reset for next block
                multi_line_detected = False

            else:
                # Append to current buffer if it's just a single line header
                text_segmented_list.extend(line_buffer)
                line_buffer.clear()

    # Add remaining lines in the buffer at the end
    if line_buffer:
        text_segmented_list.extend(line_buffer)

    # Close final segment
    if segment_open:
        text_segmented_list.append(f"{end_marker}\n")

    return "".join(text_segmented_list)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--start-marker", type=str, default=segment_start_marker)
    parser.add_argument("--end-marker", type=str, default=segment_end_marker)

    args = parser.parse_args()

    # List all .txt files in the specified directory
    input_files = [f for f in os.listdir(args.input_dir) if f.endswith(".txt")]
    input_files = [f for f in input_files if "segmented" not in f]  # swapped not position

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for file_name in input_files:
        file_path_in = os.path.join(args.input_dir, file_name)

        # Read the content of each file
        with open(file_path_in, "r", encoding="utf-8") as file:
            content = file.read()

        # Apply the mark_segments function
        content_segmented = mark_segments(content, args.start_marker, args.end_marker)

        # Save the segmented content to a new file
        file_name_out = file_name.replace(".txt", "_segmented.txt")
        file_path_out = os.path.join(args.output_dir, file_name_out)

        with open(file_path_out, "w", encoding="utf-8") as new_file:
            new_file.write(content_segmented)

        print(f"Processed {file_name}")
