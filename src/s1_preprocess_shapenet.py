import math
import multiprocessing as mp
import traceback
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import mesh_to_sdf.surface_point_cloud as spc
import numpy as np
import pandas as pd
import trimesh
from mesh_to_sdf import BadMeshException, sample_sdf_near_surface
from mesh_to_sdf.utils import sample_uniform_points_in_unit_sphere, scale_to_unit_sphere
from tqdm import tqdm

from render import coords_to_image


# noinspection PyPep8Naming
def SurfacePointCloud__get_sdf(self, query_points, use_depth_buffer, sample_count=11, return_gradients=False):
    if use_depth_buffer:
        distances, indices = self.kd_tree.query(query_points)
        distances = distances.astype(np.float32).reshape(-1)
        inside = ~self.is_outside(query_points)
        distances[inside] *= -1

    else:
        distances, indices = self.kd_tree.query(query_points, k=sample_count)
        distances = distances.astype(np.float32)

        closest_points = self.points[indices]
        direction_from_surface = query_points[:, np.newaxis, :] - closest_points
        inside = np.einsum('ijk,ijk->ij', direction_from_surface, self.normals[indices]) < 0
        votes_ratio = np.sum(inside, axis=1) / sample_count
        inside = votes_ratio > 0.5
        all_or_nothing = np.isclose(votes_ratio, 0) | np.isclose(votes_ratio, 1)
        # inside1 = self.mesh.ray.contains_points(query_points)

        distances = distances[:, 0]
        distances[inside] *= -1
        distances[~all_or_nothing] = np.nan

    return distances


# noinspection PyPep8Naming
def SurfacePointCloud__sample_sdf_near_surface(
        self, number_of_points=500000, use_scans=True, sign_method='normal',
        normal_sample_count=11, min_size=0, return_gradients=False
):
    dtype = np.float32  # TODO dtype
    surface_num = int(number_of_points * 0.94)
    surface_num_nan = surface_num * 3
    surface_points_nan = self.get_random_surface_points(surface_num_nan // 2, use_scans=True).astype(dtype)
    surface_query_nan = np.concatenate((
        surface_points_nan + np.random.normal(scale=0.0025, size=(surface_num_nan // 2, 3)),
        surface_points_nan + np.random.normal(scale=0.00025, size=(surface_num_nan // 2, 3))
    ))

    unit_num = number_of_points - surface_num
    unit_num_nan = unit_num * 20
    unit_query_nan = sample_uniform_points_in_unit_sphere(unit_num_nan).astype(dtype)

    def get_sdf_in_batches(query):
        if sign_method == 'normal':
            sdf = self.get_sdf_in_batches(query, use_depth_buffer=False, sample_count=normal_sample_count)
        elif sign_method == 'depth':
            sdf = self.get_sdf_in_batches(query, use_depth_buffer=True)
        else:
            raise ValueError('Unknown sign determination method: {:s}'.format(sign_method))
        return sdf

    def get_query_sdf(query_nan):
        sdf_nan = get_sdf_in_batches(query_nan)
        mask = ~np.isnan(sdf_nan)
        query = query_nan[mask]
        sdf = sdf_nan[mask]
        return query, sdf

    more_surface_query, more_surface_sdf = get_query_sdf(surface_query_nan)
    more_unit_query, more_unit_sdf = get_query_sdf(unit_query_nan)

    if more_surface_sdf.shape[0] < surface_num:
        raise BadMeshException()

    surface_idx = np.random.choice(more_surface_sdf.shape[0], surface_num, replace=False)
    surface_query = more_surface_query[surface_idx]
    surface_sdf = more_surface_sdf[surface_idx]

    ####################################################################################################################
    inside_part = trimesh.convex.convex_hull(self.mesh).volume / (4/3 * np.pi)
    need_inside_num = int(inside_part * unit_num)
    need_outside_num = unit_num - need_inside_num

    inside_mask = more_unit_sdf <= 0
    outside_mask = more_unit_sdf > 0

    actual_inside_num = np.count_nonzero(inside_mask)
    actual_outside_num = np.count_nonzero(outside_mask)

    if actual_inside_num < need_inside_num or actual_outside_num < need_outside_num:
        raise BadMeshException()

    unit_idx = np.concatenate((
        np.random.choice(np.where(inside_mask)[0], need_inside_num, replace=False),
        np.random.choice(np.where(outside_mask)[0], need_outside_num, replace=False)
    ))
    unit_query = more_unit_query[unit_idx]
    unit_sdf = more_unit_sdf[unit_idx]
    ####################################################################################################################

    query = np.concatenate((surface_query, unit_query))
    sdf = np.concatenate((surface_sdf, unit_sdf))
    return query, sdf


spc.SurfacePointCloud.get_sdf = SurfacePointCloud__get_sdf
spc.SurfacePointCloud.sample_sdf_near_surface = SurfacePointCloud__sample_sdf_near_surface


def f(key_path, key, points_num, out_queue: mp.Queue):
    mesh = scale_to_unit_sphere(trimesh.load(key_path[key]))
    try:
        coords, real_sdf = sample_sdf_near_surface(mesh, number_of_points=points_num, min_size=0.015)
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
    store_path = Path('../datasets/shapenet30000.h5')
    points_num = 30000  # 2 * 64 ** 3
    max_workers = 4
    out_queue = mp.Queue()

    paths = list(map(Path, glob('../datasets/ShapeNetCore.v2/02691156/*/models/*', recursive=True)))
    keys = ['_' + path.parts[path.parts.index('models') - 1] for path in paths]
    key_path = dict(zip(keys, paths))

    skip_keys = {
        '_17ac3afd54143b797172a40a4ca640fe',
        '_2066f1830765183544bb6b89d63deb6f',
        '_240136d2ae7d2dec9fe69c7ccc27d2bf',
        '_24bdf389877fb7f21b1f694e36340ebb',
        '_2af04ef09d49221b85e5214b0d6a7',
        '_310f0957ae1436d88025a4ffa6c0c22b',
        '_3b82e575165383903c83f6e156ad107a',
        '_45bd6c0723e555d4ba2821676102936',
        '_461891f9231fc09a3d21c2160f47f16',
        '_81e6b629264dad5daf2c6c19cc41708a',
        '_84b84d39581a6766925c10fa44a32fd7',
        '_8e2e03ed888e0eace4f2488af2c37f8d',
        '_ad66ac8155a316422068b7c500584ade',
        '_e53547a01129eef87eda1e12bd28fb7',
        '_f31be358cb57dffffe198fc7b510f52f',
        '_fa27e66018f82cf6e549ab640f51dca9',
        '_fa9d933674352c3711a50c43722a30cb',
        '_4e4128a2d12c818e5f38952c9fdf4604',
        '_5d5aefde5935fc9eaa5d0ddd6a2781ea',
        '_f50eba69a3be1a1e536cfc00d8c31ac5',
    }


    def run(**kwargs):
        kwargs1 = {
            **kwargs,
            **{
                'key_path': key_path,
                'points_num': points_num,
                'out_queue': out_queue,
            }
        }

        # f(**kwargs1)
        mp.Process(target=f, kwargs=kwargs1).start()


    if store_path.exists():
        store = pd.HDFStore(store_path, 'r')
        done_keys = {k.lstrip('/') for k in store.keys()}
        store.close()
    else:
        done_keys = {}

    todo_keys = list(set(keys).difference(done_keys).difference(skip_keys))

    for in_key in todo_keys[:max_workers]:
        run(key=in_key)
    workers = max_workers

    for in_key in tqdm(todo_keys[max_workers:], smoothing=0.1):
        try:
            if workers >= max_workers:
                out_key, df, img = out_queue.get()
                workers -= 1

                if df is None:
                    tqdm.write(f'bad {out_key}:')
                else:
                    img.convert('RGB').save(f'../datasets/points/{out_key}.jpg', 'JPEG')
                    store = pd.HDFStore(store_path, 'a')
                    store[out_key] = df
                    store.close()

            run(key=in_key)
            workers += 1

        except Exception:
            tqdm.write(f'skip {in_key}:')
            traceback.print_exc()
