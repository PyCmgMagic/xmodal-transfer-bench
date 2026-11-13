"""Small, dependency-free helpers shared by the portable data tools."""
import hashlib
import json
from pathlib import Path

ARMS = ('original', 'repeat10', 'multiform10')
MODALITIES = ('text', 'vision')
DIRECTIONS = ('T2T', 'T2V', 'V2T', 'V2V')


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except Exception as exc:
                    raise ValueError(f'{path}:{number}: invalid JSON') from exc


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def content_digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')


def inside(root, relative):
    """Resolve a portable package member, rejecting traversal and absolute paths."""
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Not a package-relative path: {relative}')
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f'Path escapes package: {relative}')
    return result


def dataset_paths():
    for arm in ARMS:
        for modality in MODALITIES:
            yield f'train/{arm}/{modality}.jsonl'
    for direction in DIRECTIONS:
        yield f'eval/{direction}.jsonl'


def relocate(row, root):
    if 'images' in row:
        row['images'] = [str(inside(root, path)) for path in row['images']]
        for path in row['images']:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    return row
