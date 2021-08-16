import multiprocessing as mp
import traceback
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from mesh_to_sdf import sample_sdf_near_surface
from tqdm import tqdm

from render import coords_to_image


def f(key_path, key, points_num, out_queue: mp.Queue):
    scene = trimesh.load(key_path[key])
    coords, real_sdf = sample_sdf_near_surface(scene, number_of_points=points_num)
    mask = ~np.isnan(real_sdf)
    coords = coords[mask]
    real_sdf = real_sdf[mask]

    render_coords = coords[real_sdf <= 0]
    img = coords_to_image(render_coords)
    img.convert('RGB').save(f'../datasets/points/{key}.jpg', 'JPEG')

    df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'z': coords[:, 2],
        'real_sdf': real_sdf,
    }, copy=False)

    out_queue.put((key, df))


if __name__ == '__main__':
    store_path = Path('../datasets/shapenet.h5')
    points_num = 60000  # 2 * 64 ** 3
    max_workers = 4
    out_queue = mp.Queue()

    paths = list(map(Path, glob('../datasets/ShapeNetCore.v2/02691156/*/models/*', recursive=True)))
    keys = ['_' + path.parts[path.parts.index('models') - 1] for path in paths]
    key_path = dict(zip(keys, paths))


    def run(**kwargs):
        mp.Process(
            target=f,
            kwargs={
                **kwargs,
                **{
                    'key_path': key_path,
                    'points_num': points_num,
                    'out_queue': out_queue,
                }
            }
        ).start()


    if store_path.exists():
        store = pd.HDFStore(store_path, 'r')
        done_keys = {k.lstrip('/') for k in store.keys()}
        store.close()
    else:
        done_keys = {}

    todo_keys = list(set(keys).difference(done_keys))

    for in_key in todo_keys[:max_workers]:
        run(key=in_key)
    workers = max_workers

    for in_key in tqdm(todo_keys[max_workers:], smoothing=0.1):
        try:
            if workers >= max_workers:
                out_key, df = out_queue.get()
                tqdm.write(f'{in_key} {points_num - len(df)}')
                workers -= 1

                store = pd.HDFStore(store_path, 'a')
                store[out_key] = df
                store.close()

            run(key=in_key)
            workers += 1

        except Exception:
            tqdm.write(f'skip {in_key}:')
            traceback.print_exc()
