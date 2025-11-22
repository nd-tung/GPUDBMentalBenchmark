#!/usr/bin/env python3

import sys

if len(sys.argv) < 2:
    print("Usage: python test_string_pred.py <lineitem.tbl> [shipmode]")
    sys.exit(1)

file_path = sys.argv[1]
shipmode = sys.argv[2] if len(sys.argv) > 2 else 'AIR'

print(f"Counting rows where l_shipmode = '{shipmode}'...")

count = 0
with open(file_path, 'r') as f:
    for line in f:
        fields = line.split('|')
        if len(fields) > 14:
            # l_shipmode is column 14 (0-indexed)
            mode = fields[14].strip()
            if mode == shipmode:
                count += 1

print(f"COUNT(*) = {count}")

# Also show distinct l_shipmode values
distinct_modes = set()
with open(file_path, 'r') as f:
    for line in f:
        fields = line.split('|')
        if len(fields) > 14:
            mode = fields[14].strip()
            distinct_modes.add(mode)

print(f"\nDistinct l_shipmode values: {sorted(distinct_modes)}")
print(f"Counts by shipmode:")
for mode in sorted(distinct_modes):
    mode_count = 0
    with open(file_path, 'r') as f:
        for line in f:
            fields = line.split('|')
            if len(fields) > 14 and fields[14].strip() == mode:
                mode_count += 1
    print(f"  {mode}: {mode_count}")
