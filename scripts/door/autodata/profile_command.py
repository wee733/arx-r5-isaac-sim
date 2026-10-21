#!/usr/bin/env python3
"""Measure one command continuously; JSONL survives a failed simulation.

GPU readings cover the whole device; RSS sums can count shared pages twice.
Cgroup current usage includes the file cache; its memory.peak raw samples cover
the container's lifetime, while the summary's sampled peaks cover this command.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import resource
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    argv = args.command[1:] if args.command[:1] == ['--'] else args.command
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    metadata = dict(argv=argv, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    sample_interval_requested_s=1., gpu_scope='whole_device_not_process_only',
                    exit_detection_resolution_s=1.,
                    cgroup_scope='container_current_usage_including_cache; raw_memory_peak_is_lifetime',
                    rss_scope='child_process_tree_sum_shared_pages_may_be_counted_twice')
    (args.output/'command.json').write_text(json.dumps(metadata, indent=2))
    peak_rss = peak_gpu = peak_cgroup = 0
    count = 0
    with (args.output/'stdout.log').open('w') as log, (args.output/'resources.jsonl').open('w') as samples:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        while True:
            tick = time.monotonic()
            processes = {}
            for entry in Path('/proc').iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    status = dict(line.split(':', 1) for line in (entry/'status').read_text().splitlines())
                    processes[int(entry.name)] = (int(status['PPid']), int(status.get('VmRSS', '0 kB').split()[0]))
                except (OSError, ValueError, KeyError):
                    pass
            children = {proc.pid}
            for _ in range(10):
                found = {pid for pid, (parent, _) in processes.items() if parent in children}
                if found <= children:
                    break
                children |= found
            rss = sum(processes.get(pid, (0, 0))[1] for pid in children) * 1024
            row = dict(elapsed_s=tick-start, process_tree_rss_bytes=rss, pids=sorted(children))
            peak_rss = max(peak_rss, rss)
            for key in ('memory.current', 'memory.peak'):
                try:
                    row['cgroup_'+key.replace('.', '_')+'_bytes'] = int(Path('/sys/fs/cgroup', key).read_text())
                except (OSError, ValueError):
                    pass
            peak_cgroup = max(peak_cgroup, row.get('cgroup_memory_current_bytes', 0))
            try:
                gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,power.draw',
                                      '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
                if gpu.returncode:
                    raise ValueError(gpu.stderr.strip() or 'nvidia-smi returned no metrics')
                row['gpu_devices'] = [dict(zip(('memory_used_mib','utilization_percent','power_w'),
                                               map(float, line.split(',')))) for line in gpu.stdout.strip().splitlines()]
                peak_gpu = max([peak_gpu, *(x['memory_used_mib'] for x in row['gpu_devices'])])
            except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                row['gpu_sampling_error'] = str(error)
            samples.write(json.dumps(row)+'\n')
            samples.flush()
            count += 1
            if proc.poll() is not None:
                break
            time.sleep(max(0., 1.-(time.monotonic()-tick)))
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    metadata.update(exit_code=proc.returncode, wall_seconds=time.monotonic()-start, samples=count,
                    ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    sampled_peak_process_tree_rss_bytes=peak_rss, sampled_peak_gpu_used_mib=peak_gpu,
                    sampled_peak_cgroup_memory_current_bytes=peak_cgroup,
                    child_rusage_maxrss_bytes=usage.ru_maxrss*1024,
                    children_user_cpu_seconds=usage.ru_utime, children_system_cpu_seconds=usage.ru_stime)
    (args.output/'summary.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata), flush=True)
    return proc.returncode


if __name__ == '__main__':
    raise SystemExit(main())
