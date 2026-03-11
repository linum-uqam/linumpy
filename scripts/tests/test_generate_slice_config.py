#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import csv
from pathlib import Path


def test_help(script_runner):
    ret = script_runner.run(['linum_generate_slice_config.py', '--help'])
    assert ret.success


def test_from_shifts_file(script_runner, tmp_path):
    """Test generating slice config from an existing shifts file."""
    # Create a sample shifts file
    shifts_file = tmp_path / 'shifts_xy.csv'
    with open(shifts_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fixed_id', 'moving_id', 'x_shift', 'y_shift', 'x_shift_mm', 'y_shift_mm'])
        writer.writerow([0, 1, 10, 5, 0.01, 0.005])
        writer.writerow([1, 2, 8, 3, 0.008, 0.003])
        writer.writerow([2, 3, 12, 7, 0.012, 0.007])
    
    output = tmp_path / 'slice_config.csv'
    ret = script_runner.run(['linum_generate_slice_config.py', str(shifts_file),
                             str(output), '--from_shifts', '--exclude_first', '0'])
    assert ret.success
    assert output.exists()

    # Verify the content
    with open(output, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 4  # slices 0, 1, 2, 3
    for row in rows:
        assert row['use'] == 'true'
        assert row['slice_id'] in ['00', '01', '02', '03']


def test_from_shifts_file_with_exclude(script_runner, tmp_path):
    """Test generating slice config with exclusions."""
    # Create a sample shifts file
    shifts_file = tmp_path / 'shifts_xy.csv'
    with open(shifts_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fixed_id', 'moving_id', 'x_shift', 'y_shift', 'x_shift_mm', 'y_shift_mm'])
        writer.writerow([0, 1, 10, 5, 0.01, 0.005])
        writer.writerow([1, 2, 8, 3, 0.008, 0.003])
        writer.writerow([2, 3, 12, 7, 0.012, 0.007])
    
    output = tmp_path / 'slice_config.csv'
    ret = script_runner.run(['linum_generate_slice_config.py', str(shifts_file),
                             str(output), '--from_shifts', '--exclude_first', '0',
                             '--exclude', '1', '2'])
    assert ret.success
    assert output.exists()
    
    # Verify the content
    with open(output, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    assert len(rows) == 4
    for row in rows:
        if row['slice_id'] in ['01', '02']:
            assert row['use'] == 'false'
        else:
            assert row['use'] == 'true'

