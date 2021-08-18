import multiprocessing as mp
import traceback
from glob import glob
from pathlib import Path

import pandas as pd
import trimesh
from mesh_to_sdf import BadMeshException, sample_sdf_near_surface
from mesh_to_sdf.utils import scale_to_unit_sphere
from tqdm import tqdm

# noinspection PyUnresolvedReferences
import monkey_patches
from render import coords_to_image


# noinspection PyShadowingNames
def worker(key_path, key, points_num, out_queue: mp.Queue):
    mesh = scale_to_unit_sphere(trimesh.load(key_path[key]))
    try:
        coords, real_sdf = sample_sdf_near_surface(mesh, number_of_points=points_num)
    except BadMeshException:
        out_queue.put((key, None, None))
        return

    df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'z': coords[:, 2],
        'real_sdf': real_sdf,
    }, copy=False)

    render_coords = coords[real_sdf <= 0]
    img = coords_to_image(render_coords)

    out_queue.put((key, df, img))


if __name__ == '__main__':
    store_path = Path('../datasets/shapenet300000.h5')
    points_num = 30000  # 2 * 64 ** 3
    max_workers = 4
    out_queue = mp.Queue()

    with open('../datasets/skip_keys.txt', 'r') as f:
        skip_keys = set(f.read().splitlines())

    paths = list(map(Path, glob('../datasets/ShapeNetCore.v2/02691156/*/models/*', recursive=True)))
    keys = ['_' + path.parts[path.parts.index('models') - 1] for path in paths]
    key_path = dict(zip(keys, paths))

    if store_path.exists():
        store = pd.HDFStore(store_path, 'r')
        store_keys = {k.lstrip('/') for k in store.keys()}
        store.close()
    else:
        store_keys = set()

    remove_keys = store_keys.intersection(skip_keys)
    store = pd.HDFStore(store_path, 'a')
    for k in tqdm(remove_keys):
        tqdm.write(f'remove {k}')
        store.remove(k)
    store.close()

    workers = 0
    todo_keys = list(set(keys).difference(store_keys).difference(skip_keys))

    with tqdm(total=len(todo_keys)) as bar:
        for in_key in todo_keys:
            # noinspection PyBroadException
            try:
                kwargs = {
                    'key': in_key,
                    'key_path': key_path,
                    'points_num': points_num,
                    'out_queue': out_queue,
                }

                # f(**kwargs)
                mp.Process(target=worker, kwargs=kwargs).start()
                workers += 1

                if workers >= max_workers or in_key in todo_keys[-max_workers:]:
                    out_key, df, img = out_queue.get()
                    workers -= 1
                    bar.update(1)

                    if df is None:
                        tqdm.write(f'bad {out_key}')
                    else:
                        img.convert('RGB').save(f'../datasets/points/{out_key}.jpg', 'JPEG')
                        store = pd.HDFStore(store_path, 'a')
                        store[out_key] = df
                        store.close()

            except Exception:
                tqdm.write(f'error {in_key}')
                traceback.print_exc()
