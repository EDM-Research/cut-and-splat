import math
import os
import time

import numpy as np
import open3d as o3d

from cut_and_splat.utils.camera import get_intrinsics
from cut_and_splat.utils.geometry import normal_of_bbox, vector_angle, flip_bbox, Plane, find_closest_axis, \
    count_points_above_plane


class PlacementPlane(Plane):
    """
    Datastructure that describes a plane in 3D space with a normal and center
    The datastructure also contains:
    - the 3D points on the plane
    - intrinsics of the camera used to capture the image, used to map 3d point to 2d
    """
    def __init__(self, normal: np.array, center: np.array, points_3d: np.array, cx: float, cy: float, f: float, volume: float):
        super().__init__(normal, center)
        self.points_3d = points_3d
        self.f = f
        self.cx = cx
        self.cy = cy
        self.volume = volume

    def get_point_2d(self, index: int) -> (int, int):
        y, x, z = self.points_3d[index]
        return int(((self.f * x) / z) + self.cx), int(((-self.f * y) / z) + self.cy)


class PlaneFinder:
    """
    Class for finding planes in depth maps
    """
    def __init__(self, filter_top: bool = True):
        """
        filter_top specifies whether planes in the upper part of the scene should be excluded
        """
        self.filter_top = filter_top
        self.up = np.array([0, 0, 1])

    @staticmethod
    def depth_to_points(depth: np.array) -> (np.array, dict):
        """
        Convert the given depth map to a point cloud
        Additionally, return a mapping between the index of the 3D points and their 2D image locations
        """
        f, cx, cy = get_intrinsics(depth.shape[0], depth.shape[1])
        u, v = np.indices(depth.shape)
        d = depth.copy()

        mask = (d != 0)
        d[mask] = d[mask]

        z = d[mask]
        x = (u[mask] - cx) * z / f
        y = (v[mask] - cy) * z / f

        points = np.column_stack((y, -x, -z))

        return points

    @staticmethod
    def prepare_point_cloud(points: np.array, image: np.array = None) -> o3d.geometry.PointCloud:
        """
        Given a list of 3D points and a color image, return a prepared open3d point cloud
        Outliers are removed and the normals are estimated for further processing
        If a color image is specified, colors are assigned to the point cloud
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if image is not None:
            colors = np.array(image).reshape(-1, 3)
            pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        return pcd

    @staticmethod
    def get_manual_normal(pcd: o3d.geometry.PointCloud):
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()
        # Picked point #84 (-0.00, 0.01, 0.01) to add in queue.
        # Picked point #119 (0.00, 0.00, -0.00) to add in queue.
        # Picked point #69 (-0.01, 0.02, 0.01) to add in queue.
        vis.destroy_window()
        selected_indices = vis.get_picked_points()  # [84, 119, 69]

        points = np.array(pcd.points)

        normal = points[selected_indices[1]] - points[selected_indices[0]]
        normal = normal / np.linalg.norm(normal)

        return normal

    def find_planes(self, depth: np.array, override_scene_normal: np.array = None) -> (list, np.array):
        """
        Find horizontal planes in an rgb image using monocular depth estimation
        """
        points = self.depth_to_points(depth)
        pcd = self.prepare_point_cloud(points)
        points = np.asarray(pcd.points)

        oboxes = pcd.detect_planar_patches(
            normal_variance_threshold_deg=10,
            coplanarity_deg=75,
            outlier_ratio=0.75,
            min_plane_edge_length=0,
            min_num_points=100,
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
        )

        if override_scene_normal is not None:
            scene_normal = override_scene_normal
        else:
            scene_box = pcd.get_oriented_bounding_box()
            scene_normal = normal_of_bbox(scene_box)

        if vector_angle(scene_normal, self.up) > math.pi / 2:
            scene_normal = 1.0 - scene_normal

        found_planes = []
        geometries = []

        for box in oboxes:
            plane_normal = normal_of_bbox(box)
            plane_angle = vector_angle(plane_normal, scene_normal)

            # invert the normal in case the angle is too large
            if plane_angle > math.pi / 2:
                plane_normal = -1 * plane_normal
                plane_angle = vector_angle(plane_normal, scene_normal)
                box = flip_bbox(box)

            # filter planes that are close to the ceiling
            p_above = count_points_above_plane(Plane(plane_normal, box.center), points) / len(points)
            if self.filter_top and p_above < 0.3:
                continue

            # keep this plane in case the angle is small enough
            if 0 < plane_angle < 1.0:
                plane_points_3d = np.asarray(pcd.crop(box).points)
                f, cx, cy = get_intrinsics(depth.shape[0], depth.shape[1])
                if len(plane_points_3d) > 5:
                    found_planes.append(PlacementPlane(plane_normal, box.center, plane_points_3d, f=f, cx=cx, cy=cy, volume=box.volume()))

                # collect meshes for debugging purposes
                if os.environ.get("DEBUG", '0') == '1':
                    mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(box, scale=[1, 1, 0.0001])
                    mesh.paint_uniform_color(box.color)
                    geometries.append(mesh)
                    geometries.append(box)
                    geometries.append(self.get_frame(box, size=0.2))

        if os.environ.get("DEBUG", '0') == '1':
            self.show_debug_visualization(pcd, None, geometries)

        return found_planes, scene_normal

    @staticmethod
    def get_frame(box: o3d.geometry.OrientedBoundingBox, size=1.0):
        obb_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        obb_frame.translate(box.center)
        obb_frame.rotate(box.R, center=box.center)

        return obb_frame

    @staticmethod
    def show_debug_visualization(pcd: o3d.geometry.PointCloud, scene_box: o3d.geometry.OrientedBoundingBox = None, geometries: list = None):
        """
        Shows a debug visualization of the given point cloud and a list of debug geometries
        """
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)

        to_render = [pcd, world_frame]

        if scene_box is not None:
            obb_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
            obb_frame.translate(scene_box.center)
            obb_frame.rotate(scene_box.R, center=scene_box.center)
            to_render += scene_box

        if geometries is not None:
            to_render += geometries

        o3d.visualization.draw_geometries(to_render)
