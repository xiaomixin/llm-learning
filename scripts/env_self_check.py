"""
Colab/本地环境自检脚本
用法：
    !python scripts/env_self_check.py

检查：
    1. Python 版本 ≥ 3.10
    2. GPU 可见 + CUDA 版本
    3. 关键包版本
    4. torch.matmul 在 GPU 上能跑
    5. Drive 路径（仅 Colab）
"""
import importlib
import os
import platform
import sys
import time


EXPECTED = {
    'torch': '2.0',
    'numpy': '1.24',
    'pandas': '2.0',
    'sklearn': '1.2',
    'matplotlib': '3.6',
    'seaborn': '0.12',
}


def check(ok: bool, msg: str) -> None:
    mark = 'PASS' if ok else 'FAIL'
    symbol = '+' if ok else 'x'
    print(f'  [{symbol}] {mark}  {msg}')


def main() -> None:
    print('=' * 60)
    print('Environment self-check')
    print('=' * 60)
    print(f'Platform : {platform.platform()}')
    print(f'Python   : {sys.version.split()[0]}')

    check(sys.version_info >= (3, 10), 'Python >= 3.10')

    print('\nPackages:')
    for pkg, min_ver in EXPECTED.items():
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, '__version__', '?')
            ok = ver >= min_ver
            check(ok, f'{pkg:12s} = {ver:10s} (need >= {min_ver})')
        except ImportError:
            check(False, f'{pkg:12s} NOT INSTALLED')

    print('\nPyTorch / CUDA:')
    try:
        import torch
        check(True, f'torch {torch.__version__}')
        gpu = torch.cuda.is_available()
        check(gpu, 'CUDA available' if gpu else 'No GPU — switch Runtime to GPU')
        if gpu:
            dev = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            check(True, f'GPU: {dev} ({mem:.1f} GB)')

            # quick matmul benchmark
            a = torch.randn(4096, 4096, device='cuda')
            b = torch.randn(4096, 4096, device='cuda')
            torch.cuda.synchronize()
            t0 = time.time()
            _ = a @ b
            torch.cuda.synchronize()
            ms = (time.time() - t0) * 1000
            check(ms < 500, f'matmul 4096x4096 = {ms:.1f} ms (need < 500 on GPU)')
    except ImportError:
        check(False, 'torch NOT INSTALLED')

    print('\nFilesystem:')
    drive = '/content/drive/MyDrive'
    if os.path.exists('/content'):
        check(os.path.exists(drive), f'Drive mounted at {drive}' if os.path.exists(drive) else 'Drive NOT mounted — run drive.mount()')
        proj = f'{drive}/transformer-anomaly'
        check(os.path.exists(proj), f'Project dir exists: {proj}')
    else:
        check(True, 'Running locally (not Colab)')

    print('\n' + '=' * 60)
    print('Done. Fix any [x] items before proceeding.')


if __name__ == '__main__':
    main()
