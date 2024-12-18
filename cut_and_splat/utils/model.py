import copy

import numpy as np
import open3d.geometry

from cut_and_splat.utils.geometry import Plane, align_vectors, calculate_geometric_center, distances_to_plane, \
    get_translation_matrix, get_reflection_matrix
from scene import GaussianModel
from utils.sh_utils import SH2RGB
import open3d as o3d


def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    source_temp.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([source_temp, target])


def model_to_pointcloud(model: 'GaussianModelWrapper') -> open3d.geometry.PointCloud:
    points = model.foreground_model.get_xyz.cpu().numpy()
    colors = SH2RGB(model.foreground_model.get_features[:, 0, :].cpu().numpy())
    colors = np.clip(colors, 0.0, 1.0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def estimate_alignment(target: 'GaussianModelWrapper', source: 'GaussianModelWrapper'):
    shift = get_model_center(target.foreground_model, geometric=True) - get_model_center(source.foreground_model, geometric=True)

    translation = get_translation_matrix(shift[0], shift[1], shift[2])
    reflection = get_reflection_matrix(target.ground_plane.normal, get_model_center(target.foreground_model, geometric=True))

    return reflection @ translation


def merge_models(target: 'GaussianModelWrapper', source: 'GaussianModelWrapper'):
    """
    Merge two GaussianModelWrappers. The second model is merged into the first.
    """
    pcd_t = model_to_pointcloud(target)
    pcd_s = model_to_pointcloud(source)

    base_radius = 0.04
    voxel_radius = [base_radius, base_radius / 2, base_radius / 4]
    max_iter = [50, 50, 50]

    current_transformation = estimate_alignment(target, source)
    draw_registration_result_original_color(pcd_s, pcd_t, current_transformation)

    best_fitness = None
    best_transform = current_transformation

    for scale in range(3):
        iterations = max_iter[scale]
        radius = voxel_radius[scale]

        source_down = pcd_s
        target_down = pcd_t

        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=100))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=100))

        print("3-3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-7,
                                                              relative_rmse=1e-7,
                                                              max_iteration=iterations))
        print(result_icp)

        current_transformation = result_icp.transformation

        if best_fitness is None or result_icp.fitness > best_fitness:
            best_transform = current_transformation
            best_fitness = result_icp.fitness

    draw_registration_result_original_color(pcd_s, pcd_t, best_transform)
    print(f"Best fitness: {best_fitness}")

    return target


def model_subset(other: GaussianModel, indices: list):
    new_model = GaussianModel(other.max_sh_degree)

    new_model._xyz = other._xyz[indices]
    new_model._features_dc = other._features_dc[indices]
    new_model._features_rest = other._features_rest[indices]
    new_model._opacity = other._opacity[indices]
    new_model._scaling = other._scaling[indices]
    new_model._rotation = other._rotation[indices]

    new_model.active_sh_degree = other.max_sh_degree

    return new_model


def get_model_center(model: GaussianModel, geometric: bool = False):
    points = model.get_xyz.detach().cpu().numpy()
    if not geometric:
        center = np.average(points, axis=0)
    else:
        center = calculate_geometric_center(points)
    return center


def get_low_center(model: GaussianModel, ground_plane: Plane, ratio: float = 0.2):
    points = model.get_xyz.detach().cpu().numpy()
    distances = distances_to_plane(points, ground_plane)

    threshold = ratio * np.max(distances)
    indices = np.where(distances < threshold)[0]

    # find center of all low points
    center = calculate_geometric_center(points[indices])
    return center


def get_model_base(model: GaussianModel, ground_plane: Plane):
    center = get_low_center(model, ground_plane, ratio=0.2)
    base_point = ground_plane.project_point(center)

    return base_point


def adjust_ground_plane(model: GaussianModel, ground_plane: Plane):
    points = model.get_xyz.detach().cpu().numpy()
    distances = distances_to_plane(points, ground_plane)

    closest_i = np.argmin(distances)

    adjusted_plane = Plane(ground_plane.normal, points[closest_i])

    return adjusted_plane


def show_model(model, extra_points=None):
    pcd = model_to_pointcloud(model)

    geometries = [pcd]

    if extra_points is not None:
        extra_pcd = o3d.geometry.PointCloud()
        extra_pcd.points = o3d.utility.Vector3dVector(extra_points)
        extra_pcd.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0, 0], (len(extra_points), 1)))

        geometries.append(extra_pcd)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    frame.translate(model.foreground_base)
    frame.rotate(align_vectors(np.array([0, 0, 1]), model.ground_plane.normal), center=model.foreground_base)

    geometries.append(frame)

    o3d.visualization.draw_geometries(geometries, lookat=model.foreground_base, up=model.ground_plane.normal, front=np.array([1.0, 0.0, 0.0]), zoom=0.1)