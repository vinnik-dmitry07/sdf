import mesh_to_sdf.surface_point_cloud as spc
import numpy as np
import trimesh.convex
from mesh_to_sdf import BadMeshException
from mesh_to_sdf.utils import sample_uniform_points_in_unit_sphere
from pyglet.libs.win32.constants import WM_SIZE, SIZE_MINIMIZED
from pyglet.window.win32 import Win32EventHandler
import pyglet.window.win32 as win32


# noinspection PyPep8Naming,PyUnusedLocal
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


# noinspection PyPep8Naming,PyUnusedLocal
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
    inside_part = trimesh.convex.convex_hull(self.mesh).volume / (4 / 3 * np.pi)
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

    res_query = np.concatenate((surface_query, unit_query))
    res_sdf = np.concatenate((surface_sdf, unit_sdf))
    return res_query, res_sdf


# noinspection PyPep8Naming,PyUnusedLocal
@Win32EventHandler(WM_SIZE)
def Win32Window___event_size(self, msg, wParam, lParam):
    if not self._dc:
        # Ignore window creation size event (appears for fullscreen
        # only) -- we haven't got DC or HWND yet.
        return None

    if wParam == SIZE_MINIMIZED:
        # Minimized, not resized.
        self._hidden = True
        self.dispatch_event('on_hide')
        return 0
    if self._hidden:
        # Restored
        self._hidden = False
        self.dispatch_event('on_show')
    w, h = self._get_location(lParam)
    if w and h:  # bugfix
        if not self._fullscreen:
            self._width, self._height = w, h
        self._update_view_location(self._width, self._height)
        self.switch_to()
        self.dispatch_event('on_resize', self._width, self._height)
    return 0


spc.SurfacePointCloud.get_sdf = SurfacePointCloud__get_sdf
spc.SurfacePointCloud.sample_sdf_near_surface = SurfacePointCloud__sample_sdf_near_surface
win32.Win32Window._event_size = Win32Window___event_size
