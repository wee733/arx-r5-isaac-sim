#!/usr/bin/env python3
"""Repack a GR00T safetensors checkpoint without changing tensor bytes.

Only inference weights and processor configuration are exported. Optimizer,
RNG, trainer logs and machine-specific experiment configuration stay local.
No PyTorch/GPU dependency is required. Each tensor is copied intact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct


def header(path):
    with path.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        return json.loads(stream.read(length)), 8 + length


def digest_range(path, offset, size):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        stream.seek(offset)
        while size:
            block = stream.read(min(size, 8 * 1024 * 1024))
            if not block:
                raise ValueError(f'Truncated tensor in {path}')
            digest.update(block)
            size -= len(block)
    return digest.hexdigest()


def export(source, target, max_bytes):
    if target.exists():
        raise FileExistsError(target)
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    headers = {name: header(source / name) for name in set(index['weight_map'].values())}
    groups, group, size = [], [], 0
    for key, name in index['weight_map'].items():
        item = headers[name][0][key]
        start, end = item['data_offsets']
        tensor_bytes = end - start
        if tensor_bytes + 1024 * 1024 > max_bytes:
            raise ValueError(f'Tensor exceeds shard limit: {key}')
        if group and size + tensor_bytes + 1024 * 1024 > max_bytes:
            groups.append(group)
            group, size = [], 0
        group.append((key, name, item))
        size += tensor_bytes
    if group:
        groups.append(group)
    target.mkdir(parents=True)
    weight_map, total_size, checked = {}, 0, 0
    for number, group in enumerate(groups, 1):
        name = f'model-{number:05d}-of-{len(groups):05d}.safetensors'
        metadata, offset = {'__metadata__': {'format': 'pt'}}, 0
        for key, _, item in group:
            size = item['data_offsets'][1] - item['data_offsets'][0]
            metadata[key] = {**item, 'data_offsets': [offset, offset + size]}
            offset += size
            weight_map[key] = name
        encoded = json.dumps(metadata, separators=(',', ':')).encode()
        encoded += b' ' * (-len(encoded) % 8)
        output = target / name
        originals = {}
        with output.open('wb') as stream:
            stream.write(struct.pack('<Q', len(encoded)))
            stream.write(encoded)
            for key, original, item in group:
                start, end = item['data_offsets']
                remaining = end - start
                digest = hashlib.sha256()
                with (source / original).open('rb') as input_stream:
                    input_stream.seek(headers[original][1] + start)
                    while remaining:
                        block = input_stream.read(min(remaining, 8 * 1024 * 1024))
                        if not block:
                            raise ValueError(f'Truncated tensor: {key}')
                        stream.write(block)
                        digest.update(block)
                        remaining -= len(block)
                originals[key] = digest.hexdigest()
        saved, data_offset = header(output)
        for key, expected in originals.items():
            start, end = saved[key]['data_offsets']
            assert digest_range(output, data_offset + start, end - start) == expected, key
            checked += 1
        assert output.stat().st_size <= max_bytes
        total_size += offset
        print(f'Validated {name}: {output.stat().st_size} bytes', flush=True)
    (target / 'model.safetensors.index.json').write_text(json.dumps({
        'metadata': {'total_size': total_size}, 'weight_map': weight_map,
    }, indent=2) + '\n')
    for name in ('config.json', 'processor_config.json', 'statistics.json', 'embodiment_id.json'):
        shutil.copy2(source / name, target / name)
    (target / 'export_validation.json').write_text(json.dumps({
        'source_checkpoint': source.name, 'tensor_count': checked,
        'tensor_bytes': total_size, 'shards': len(groups),
        'tensor_bytes_verified_equal': True,
        'validation': 'Every exported tensor was read back and compared to source tensor bytes using SHA256.',
        'model_inference_test_executed': False,
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    parser.add_argument('--max-shard-bytes', type=int, default=1_900_000_000)
    args = parser.parse_args()
    export(args.source, args.target, args.max_shard_bytes)
