import warnings

import numpy as np
import trimesh
from PIL import Image
from mesh_to_sdf import sample_sdf_near_surface, get_surface_point_cloud, scale_to_unit_sphere
from mesh_to_sdf.utils import sample_uniform_points_in_unit_sphere
from trimesh.proximity import closest_point


def rotate(scene):
    matrix = trimesh.transformations.rotation_matrix(
        angle=np.radians(145.0),
        direction=[0, 1, 0.1],
        point=scene.centroid
    )
    camera_old, _geometry = scene.graph[scene.camera.name]
    camera_new = np.dot(matrix, camera_old)
    scene.graph[scene.camera.name] = camera_new


def render(scene, resolution):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        bytes_ = scene.save_image(resolution)
    image = Image.open(trimesh.util.wrap_as_stream(bytes_))
    return image


def mesh_to_image(mesh, resolution):
    scene = mesh.scene()
    rotate(scene)
    image = render(scene, resolution)
    return image


def coords_to_image(coords, resolution=(1024, 1024)):
    cloud = trimesh.points.PointCloud(coords)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        scene = cloud.scene()
    rotate(scene)
    image = render(scene, resolution)
    return image


def view_objects(objects):
    import pyrender
    scene = pyrender.Scene()
    for o in objects:
        scene.add(o)
    pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)


def to_mesh(scene):
    mesh = trimesh.util.concatenate([
        trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
        for g in scene.geometry.values()
    ])
    return mesh


def sample_sdf(
        mesh, number_of_points, use_scans=True, surface_point_method='scan', sign_method='normal',
        scan_count=100, scan_resolution=400, sample_point_count=10000000, return_gradients=False
):
    bounding_radius = 1
    surface_point_cloud = get_surface_point_cloud(
        mesh, surface_point_method, bounding_radius, scan_count, scan_resolution, sample_point_count,
        calculate_normals=sign_method == 'normal' or return_gradients
    )

    query_points = []
    surface_sample_count = int(number_of_points * 47 / 50) // 2
    surface_points = surface_point_cloud.get_random_surface_points(surface_sample_count, use_scans=use_scans)

    query_points.append(surface_points + np.random.normal(scale=0.0025, size=(surface_sample_count, 3)))
    query_points.append(surface_points + np.random.normal(scale=0.00025, size=(surface_sample_count, 3)))

    unit_sphere_sample_count = number_of_points - surface_points.shape[0] * 2
    unit_sphere_points = sample_uniform_points_in_unit_sphere(unit_sphere_sample_count)
    query_points.append(unit_sphere_points)
    query_points = np.concatenate(query_points).astype(np.float32)

    # closest, distance, triangle_id = closest_point(mesh, query_points)
    #
    # distances, indices = surface_point_cloud.kd_tree.query(query_points, k=1)
    # closest_points = surface_point_cloud.points[indices]

    # surface_point_cloud.kd_tree.query()
    sdf = trimesh.proximity.signed_distance(mesh, query_points)

    return query_points, sdf


if __name__ == '__main__':
    a_mesh = a_scene = trimesh.load(
        # '../datasets/ShapeNetCore.v2/02691156/2a3d485b0214d6a182389daa2190d234/models/model_normalized1.obj'
        '../datasets/ShapeNetCore.v2/02691156/229cd78f9ac3e77c8b96ae1a0a8b84ec/models/model_normalized.obj'
    )
    # a_mesh = to_mesh(a_scene)
    # print(a_mesh.is_watertight)
    a_mesh = scale_to_unit_sphere(a_mesh)

    # mesh_to_image(a_mesh)

    a_coords, a_sdf = sample_sdf_near_surface(
        a_mesh,
        # sign_method='depth',
        number_of_points=30000 * 2,
        # normal_sample_count=11,
        # min_size=0.015,  # less than 1.5% are negative
    )

    mask = ~np.isnan(a_sdf)
    a_coords = a_coords[mask]
    a_sdf = a_sdf[mask]
    import pyrender
    view_objects([
        pyrender.Mesh.from_points(a_coords),
        # pyrender.Mesh.from_trimesh(a_mesh),
    ])
    # sdf_to_image(lambda _: a_sdf, a_coords)

    # cloud = trimesh.points.PointCloud(trimesh.sample.volume_mesh(a_mesh, 64**3))
    # scene = cloud.scene()
    # # scene.add_geometry(a_mesh)
    # rotate(scene)
    # render(scene)
