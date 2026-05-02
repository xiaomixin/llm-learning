"""
Kaggle 数据集一键下载脚本
用法：
    # 在 Colab 中
    !python scripts/download_kaggle.py --dataset mlg-ulb/creditcardfraud --out data/

    # 本地
    python scripts/download_kaggle.py --dataset mlg-ulb/creditcardfraud --out ./data

前置：
    - ~/.kaggle/kaggle.json 已存在，或设置 KAGGLE_USERNAME + KAGGLE_KEY 环境变量
    - 或者把 kaggle.json 放在当前目录/Drive，脚本会自动发现
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


CANDIDATE_KAGGLE_JSON = [
    '/content/kaggle.json',
    '/content/drive/MyDrive/kaggle.json',
    './kaggle.json',
    os.path.expanduser('~/.kaggle/kaggle.json'),
]


def ensure_kaggle_json() -> None:
    target = Path.home() / '.kaggle' / 'kaggle.json'
    if target.exists():
        target.chmod(0o600)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    for src in CANDIDATE_KAGGLE_JSON:
        if os.path.exists(src) and Path(src) != target:
            shutil.copy(src, target)
            target.chmod(0o600)
            return
    if os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'):
        return
    sys.exit('ERROR: kaggle.json not found. Download from Kaggle → Settings → Create New API Token.')


def ensure_kaggle_cli() -> None:
    try:
        subprocess.run(['kaggle', '--version'], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print('Installing kaggle CLI...')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kaggle'], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='mlg-ulb/creditcardfraud',
                    help='Kaggle dataset slug (owner/name)')
    ap.add_argument('--competition', default=None,
                    help='Kaggle competition slug (mutex with --dataset)')
    ap.add_argument('--out', default='./data', help='Output directory')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ensure_kaggle_cli()
    ensure_kaggle_json()

    if args.competition:
        cmd = ['kaggle', 'competitions', 'download', '-c', args.competition, '-p', str(out)]
    else:
        cmd = ['kaggle', 'datasets', 'download', '-d', args.dataset, '-p', str(out), '--unzip']

    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    print(f'\nDone. Contents of {out}:')
    for p in sorted(out.iterdir()):
        size = p.stat().st_size / 1e6
        print(f'  {p.name:40s} {size:8.2f} MB')


if __name__ == '__main__':
    main()
