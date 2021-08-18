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
    store_path = Path('../datasets/shapenet300001.h5')
    points_num = 30000  # 2 * 64 ** 3
    max_workers = 4
    out_queue = mp.Queue()
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
        '_12877bdca58ccbea402991f646f01d6c',
        '_77ee6ccca238ceec5144962e2c20b832',
        '_130422d62194cf026c8067e570d6af71',
        '_78ceee44d207e826c87f6dba8af25d8b',
        '_166d333d38897d1513d521050081b441',
        '_7f6af37cd64377e1cabcecce1c335df1',
        '_1678946724380812de689e373096b0e3',
        '_7f6e55daa567aade3a1cddb101d3e1ea',
        '_172764bea108bbcceae5a783c313eb36',
        '_844d36a369cdeed3ac4f72bf08dc79a6',
        '_189f045faacc1b5f9a8993cdad554625',
        '_85cf5445f44a49f5cf35fd096a07bc43',
        '_19a624cf1037fc75cda1835f53ae7d53',
        '_87348bdeb8bc74139ebd7d1067b13a',
        '_1d5beedb73951ef68649ad8da70da1e',
        '_87d37c43f41871fb4dd260a1dd3357bc',
        '_253a1aead30731904c3a35cee92bb95b',
        '_895ae296f701f5e2ee7ff700fef7cd22',
        '_27317e8e93190374780ee0648cf79a36',
        '_8a84a26158da1db7668586dcfb752ad',
        '_2ba980d080f89581ab2a0ebad7754fba',
        '_8eab40ab482a46f04369ac253fd9f7b2',
        '_337658edebb67c301ce9f50324082ee4',
        '_90769434f2b588c1b675ec196a869fe5',
        '_38edc8fad5a5c0f0ac4f72bf08dc79a6',
        '_91bd6e91455f85fddcf9f917545742df',
        '_3921f5f454c5733e96e161ce7a956a8',
        '_93e0290ab5eede3a883f7527225435dc',
        '_395afa94dd4d549670e6bd9d4e2b211f',
        '_99ee9ae50909ac0cd3cd0742a4ec7e9b',
        '_3f69370401d4dc9a275386e1d3ac388e',
        '_9a0f4dd21a4ca19bf1cb19f636b1c2bd',
        '_3f9cab3630160be9f19e1980c1653b79',
        '_a4ea22087dec2f32c7575c9089791ff',
        '_4008286f2fe8b6a97c44cd1ce3d402d0',
        '_a9cdbca070047fe61e9dc95dd6a9f6',
        '_4031ee8a4631762c9566f663623859aa',
        '_a9dff753cf97f9c5354ab1decf4fe605',
        '_444d67950ff9a4cc1139bebb00fe5be8',
        '_ac0d1320328f5636b819f3a4f3895504',
        '_446f9144536c0e47f0c7ca80512a3ddf',
        '_b29c650e4d7582d11ae96ac7591d0dc5',
        '_4982bea0a007c19593b2f224b3acb952',
        '_b94b4edc5e4701093ba0eea71939b1f2',
        '_4b623f70c9853080aac5531514d15662',
        '_bc7ead8b45952ab8822054a0a020bf4a',
        '_4bb41171f7e6505bc32f927674bfca67',
        '_bcaf04bfae3afc1f4d48ad32fb72c8ce',
        '_52ca6970fb09b561f9f7510373841dd9',
        '_bfd606459cace196e7ee2e25a3cfaa4d',
        '_530d0edd6e2881653023dc1d1218bb2d',
        '_cff4a52406b84da7aaeb49542c2cc445',
        '_552d76fdb54e5d57cf7cf1e30e2f5267',
        '_d56fba80d39bdff738decdbba236bc1d',
        '_560b4af718f67902ac4f72bf08dc79a6',
        '_d75a4d4d25863f5062747c704d78d4f8',
        '_57f1dfb095bbe82cafc7bdb2f8d1ea84',
        '_e02485f093835f45c1b64d86df61366a',
        '_5abba5b86814b98a9f4ab5ced9b9495',
        '_e115f4f824e28246becc132ee053f2fa',
        '_5b048655453b37467584cbfee85fb982',
        '_e30e25fe047ce1ea10b08ceced9a0113',
        '_5e77ccd7794465dacbbcf57b22894cc3',
        '_e4e1b542f8732ae1c6768d9a3f27965',
        '_5f9b4ffc555c9915a3451bc89763f63c',
        '_e5a7a353d5fa8df844b2fa2cac0778f5',
        '_5fc53108f6bf2f45893f875739da1b24',
        '_e8c1e738997275799de8e648621673e1',
        '_64f01ada275ba83df1218670339368db',
        '_e8ceb64406509714e5dcd47593c60221',
        '_66e60b297f733435fff6684ee288fa59',
        '_e9704a563a5a0a3f5a4b8d382b9decda',
        '_6720352c366eba1a60370f16a3e15e76',
        '_ecc50d702133b1531e9da99095f71c63',
        '_68d3c213b16ee2a6b5f20f5912ee034d',
        '_f25ffb9cf92236fb9671f5163e7f6535',
        '_724be1fb093a31a1ac8c46f8a114a34b',
        '_f44c0e1e55a3469494f3355d9c061b5a',
        '_755b0ee19aa7037453e01eb381ca65',
        '_f4b734236ec678d269e10d525d6df27',
        '_775120d01da7a7cc666b8bccf7d1f46a',
        '_fb62efc64c58d1e5e0d07a8ce78b9182',
        '_3b82e575165383903c83f6e156ad107a',
        '_2a856d0462a38e2518b14db3b83de9ff',
        '_e7e73007e0373933c4c280b3db0d6264',
        '_7175100f99a61c9646322bce65ca3756',
        '_fb01b45a0659af80c1006ed55bc1a3fc',
    }

    paths = list(map(Path, glob('../datasets/ShapeNetCore.v2/02691156/*/models/*', recursive=True)))
    keys = ['_' + path.parts[path.parts.index('models') - 1] for path in paths]
    key_path = dict(zip(keys, paths))


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
        store_keys = {k.lstrip('/') for k in store.keys()}
        store.close()
    else:
        store_keys = {}

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
            try:
                run(key=in_key)
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
