#!/usr/bin/env python3
"""Summarize recorded ramp/hold/return data without ROS or robot access."""
import argparse
import json
import math
import statistics
from pathlib import Path


def summarize(data, tolerance=0.003):
    rows = data['records']
    ramp, hold = data.get('ramp_s', 2.), data.get('hold_s', 4.)
    peak_end, end = ramp + hold, 2*(ramp + hold)
    if not data.get('complete') or not rows or rows[-1]['t'] < end-0.2:
        raise ValueError('incomplete ramp/hold/return recording')
    anchor = rows[0]['observation']['sdk_eef'][:3]
    axis = data['axis']
    def delta(row):
        return [row['observation']['sdk_eef'][i] - anchor[i] for i in range(3)]
    def norm(v):
        return math.sqrt(sum(x*x for x in v))
    def mean_delta(lo, hi):
        selected = [delta(row) for row in rows if lo <= row['t'] < hi]
        return [statistics.mean(v[i] for v in selected) for i in range(3)]
    stable = mean_delta(peak_end-2, peak_end)
    returned = mean_delta(end-2, end)
    stable_error = norm([stable[i] - (data['amplitude_m'] if i == axis else 0) for i in range(3)])
    errors = [norm([v - (row['target_offset'] if i == axis else 0)
                    for i, v in enumerate(delta(row))]) for row in rows]
    stable_max = max(e for row, e in zip(rows, errors) if peak_end-2 <= row['t'] < peak_end)
    return_max = max(e for row, e in zip(rows, errors) if end-2 <= row['t'] < end)
    final_grip = statistics.mean(row['observation']['state']['gripper'][0]
                                for row in rows if peak_end-0.5 <= row['t'] < peak_end)
    return {
        'measurement': 'encoder FK; not externally measured Cartesian accuracy',
        'tolerance_m': tolerance,
        'stable_displacement_xyz_m': stable,
        'stable_position_error_m': stable_error,
        'return_offset_xyz_m': returned,
        'return_position_error_m': norm(returned),
        'stable_window_max_error_m': stable_max,
        'return_window_max_error_m': return_max,
        'trajectory_max_error_m': max(errors),
        'trajectory_mean_error_m': statistics.mean(errors),
        'settled_tracking_passed': stable_max <= tolerance and return_max <= tolerance,
        'continuous_tracking_passed': max(errors) <= tolerance,
        'single_finger_feedback_last_0p5s_peak_hold_m': final_grip,
        'max_joint_correction_rad': max(abs(v) for row in rows
            for v in row.get('executor', {}).get('q_correction', [0])),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(json.loads(args.recording.read_text())), indent=2))
