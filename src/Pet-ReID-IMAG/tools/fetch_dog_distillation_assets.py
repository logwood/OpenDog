"""Fetch selected public pilot videos, checking their recorded length and SHA256.

The manifest records the original author, license and source page. User-supplied
test videos have no download URL and must already be in the local asset archive.
Existing mismatched files are never replaced. No new videos are chosen here.
"""
import argparse
from pathlib import Path
import json
import urllib.request

from prepare_dog_video_distillation import sha256


def fetch(manifest, root):
    root = Path(root).resolve()
    rows = json.loads(Path(manifest).read_text(encoding='utf8'))['sources']
    for row in rows:
        if row['split'] == 'excluded':
            continue
        path = (root/row['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'Asset path escapes root: {row["path"]}')
        if path.is_file():
            if sha256(path) != row['sha256']:
                raise ValueError(f'Existing asset hash mismatch: {path}')
            print(f'Verified {row["id"]}', flush=True)
            continue
        url = row.get('download_url')
        if not url:
            raise FileNotFoundError(f'Local regression asset is missing: {path}')
        if not url.startswith('https://'):
            raise ValueError(f'Expected HTTPS download URL: {row["id"]}')
        path.parent.mkdir(parents=True,exist_ok=True)
        part = path.with_name(path.name+'.part')
        request = urllib.request.Request(url,headers={'User-Agent':'OpenDogResearch/1.0 (https://github.com/logwood/OpenDog)'})
        with urllib.request.urlopen(request,timeout=90) as response, part.open('wb') as target:
            while block := response.read(1024*1024):
                target.write(block)
        if sha256(part) != row['sha256'] or (row.get('bytes') and part.stat().st_size != row['bytes']):
            raise ValueError(f'Download verification failed; incomplete file retained at {part}')
        part.replace(path)
        print(f'Downloaded and verified {row["id"]}',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args()
    fetch(args.manifest,args.root)
