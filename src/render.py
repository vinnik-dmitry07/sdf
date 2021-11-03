import warnings

import numpy as np
import trimesh
from PIL import Image


def rotate(scene):
    matrix = trimesh.transformations.rotation_matrix(
        angle=np.radians(145.0),
        direction=[0, 1, 0.1],
        point=scene.centroid
    )
    camera_old, _geometry = scene.graph[scene.camera.name]
    camera_new = np.dot(matrix, camera_old)
    scene.graph[scene.camera.name] = camera_new
    return scene


def render(scene, resolution):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        bytes_ = scene.save_image(resolution)
    image = Image.open(trimesh.util.wrap_as_stream(bytes_))
    return image


def mesh_to_image(mesh, resolution=None):
    if resolution is None:
        resolution = (1024, 1024)
    scene = rotate(mesh.scene())
    image = render(scene, resolution)
    return image


def points_to_image(points, resolution=None):
    cloud = trimesh.points.PointCloud(points)
    # cloud = points_to_cloud(points)  todo bug https://github.com/mikedh/trimesh/issues/1198
    scene = rotate(cloud.scene())
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
