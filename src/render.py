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
