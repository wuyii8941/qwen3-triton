"""A relocated launcher must bind input paths before touching CUDA."""
import argparse
import importlib.util
from pathlib import Path
import sys

import pytest
import torch


def test_input_root_preserves_cli_path_binding_before_cuda(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        '_qwen3_input_root_entry', root / 'scripts/train_qwen3_triton_8b.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    launch = tmp_path / 'launcher'
    data = tmp_path / 'frozen_inputs'
    launch.mkdir(); data.mkdir()
    monkeypatch.chdir(launch)
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setattr(sys, 'argv', [
        'train_qwen3_triton_8b.py', '--variant', 'hf',
        '--protocol', 'protocol.json', '--input-manifest', 'inputs.json',
        '--output', 'result', '--reference-raw-dir', 'reference',
        '--input-root', str(data)])
    captured = {}
    parse = argparse.ArgumentParser.parse_args

    def capture(parser, *args, **kwargs):
        parsed = parse(parser, *args, **kwargs)
        captured['args'] = parsed
        return parsed

    class BeforeCUDA(Exception):
        pass

    def stop_before_cuda(device):
        assert device == 0
        raise BeforeCUDA

    initialized = torch.cuda.is_initialized()
    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', capture)
    monkeypatch.setattr(torch.cuda, 'set_device', stop_before_cuda)
    with pytest.raises(BeforeCUDA):
        entry.main()
    args = captured['args']
    assert Path.cwd() == data
    assert args.protocol == launch / 'protocol.json'
    assert args.input_manifest == launch / 'inputs.json'
    assert args.output == launch / 'result'
    assert args.reference_raw_dir == [launch / 'reference']
    assert entry.ROOT == root
    assert torch.cuda.is_initialized() == initialized
