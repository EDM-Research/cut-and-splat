import numpy as np

from cut_and_splat.utils.geometry import align_vectors, Plane
from cut_and_splat.utils.model import get_model_center, model_subset, get_model_base
from scene import GaussianModel
import open3d as o3d

from utils.sh_utils import SH2RGB


def get_indices(points, cropped_points):
    # Convert points and cropped_points to indices of the cropped points in first array
    dtype = np.dtype((np.void, points.dtype.itemsize * points.shape[1]))
    points_view = points.view(dtype).ravel()
    cropped_points_view = cropped_points.view(dtype).ravel()

    # Use np.in1d to find the indices of matches
    mask = np.in1d(points_view, cropped_points_view)

    # Get the indices where the mask is True
    indices = np.where(mask)[0]

    return indices


def select_model(model: GaussianModel, indices: list, base_plane: Plane) -> (GaussianModel, Plane):
    """
    Create a subset of the given GaussianModel based on the given indices.
    Also update the center of the given plane to the projection of the center of the cropped model on the plane.
    """
    foreground_model = model_subset(model, indices)
    model_center = get_model_center(foreground_model)

    if base_plane.distance(model_center) < 0:
        print("Flipping plane normal, so it is on the same side as object")
        base_plane.flip_normal()

    base_plane.center = get_model_base(foreground_model, base_plane)

    return foreground_model, base_plane


class ModelExtractor:
    """
    Class for extracting the foreground object from a Gaussian Splatting model
    """
    @staticmethod
    def build_pointcloud(points: np.array, colors: np.array = None) -> o3d.geometry.PointCloud:
        """
        Builds an open3d point cloud from a set of 3D points and optionally a set of colors for those points
        """
        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def extract(self, model: GaussianModel, manual_preprocess: bool = False) -> (GaussianModel, Plane, list):
        """
        Extract the foreground object from the Gaussian Splatting model contained by this class
        """
        points = model.get_xyz.detach().cpu().numpy()
        colors = SH2RGB(model.get_features[:, 0, :].detach().cpu().numpy())
        indices = np.arange(len(points))

        original_pcd = self.build_pointcloud(points, colors)

        pcd, inliers = self.center_radius_filter(original_pcd, percentage=0.2)
        indices = indices[inliers]

        if manual_preprocess:
            print(f"Interactive filtering... ({len(inliers)} points)")
            pcd, inliers = self.filter_interactive(pcd)
            indices = indices[inliers]

        print(f"Filtering out base plane... ({len(inliers)} points)")
        base_plane, pcd_nofloor, inliers = self.filter_plane(pcd)
        indices = indices[inliers]

        print(f"Statistic filtering... ({len(inliers)} points)")
        pcd, inliers = self.filter_statistical(pcd_nofloor)
        indices = indices[inliers]

        print(f"Cluster filtering... ({len(inliers)} points)")
        pcd, inliers = self.filter_cluster(pcd)
        indices = indices[inliers]

        pcd, inliers = self.filter_interactive(pcd)
        indices = indices[inliers]

        self.show_model(pcd, base_plane=base_plane)

        foreground_model, base_plane = select_model(model, indices, base_plane)

        return foreground_model, base_plane, indices

    def extract_interactive(self, model: GaussianModel) -> (GaussianModel, Plane, list):
        points = model.get_xyz.detach().cpu().numpy()
        colors = SH2RGB(model.get_features[:, 0, :].detach().cpu().numpy())

        original_pcd = self.build_pointcloud(points, colors)
        plane_model, plane_points = original_pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000)
        base_plane = Plane.from_equation(*plane_model)

        _, indices = self.filter_interactive(original_pcd)

        foreground_model, base_plane = select_model(model, indices, base_plane)

        return foreground_model, base_plane, indices

    @staticmethod
    def filter_plane(pcd: o3d.geometry.PointCloud):
        """
        Filter the largest plane from the given open3d point cloud.
        The filtered point cloud and the indices of the points that are kept are returned
        """
        plane_model, plane_points = pcd.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=1000)
        inliers = np.setdiff1d(np.arange(len(pcd.points)), plane_points)
        base_plane = Plane.from_equation(*plane_model)
        pcd = pcd.select_by_index(inliers)

        return base_plane, pcd, inliers

    @staticmethod
    def filter_statistical(pcd: o3d.geometry.PointCloud):
        """
        Remove floating points using a statistical filter.
        The filtered point cloud and the indices of the points that are kept are returned
        """
        _, inliers = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.5)
        pcd = pcd.select_by_index(inliers)

        return pcd, inliers

    @staticmethod
    def filter_interactive(pcd: o3d.geometry.PointCloud):
        points = np.asarray(pcd.points)
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        cropped_points = np.array(vis.get_cropped_geometry().points)

        indices = get_indices(points, cropped_points)

        # Get the indices of the cropped points in the original model
        # TODO: This is pretty slow, check for optimization
        #mask = np.all(points[:, np.newaxis] == cropped_points, axis=-1)
        #indices = np.where(mask)[0]

        return o3d.geometry.PointCloud(vis.get_cropped_geometry()), indices

    @staticmethod
    def center_radius_filter(pcd: o3d.geometry.PointCloud, percentage: float):
        points = np.asarray(pcd.points)
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)

        farthest = np.max(distances)
        threshold = percentage * farthest

        inliers = np.where(distances < threshold)[0]

        pcd = pcd.select_by_index(inliers)

        return pcd, inliers

    @staticmethod
    def filter_cluster(pcd: o3d.geometry.PointCloud):
        """
        Filter the point cloud by clustering it and keeping only the center cluster.
        The filtered point cloud and the indices of the points that are kept are returned
        """
        cluster_indices = np.array(pcd.cluster_dbscan(0.5, 100, print_progress=True))
        scene_center = pcd.get_center()

        closest_distance = None
        closest_cluster_id = None

        for cluster_id in set(cluster_indices):
            cluster_points = np.arange(len(pcd.points))[cluster_indices == cluster_id]
            cluster_pcd = pcd.select_by_index(cluster_points)
            cluster_center = cluster_pcd.get_center()
            distance = np.linalg.norm(scene_center - cluster_center)

            if closest_distance is None or distance < closest_distance:
                closest_distance = distance
                closest_cluster_id = cluster_id

        inliers = np.arange(len(pcd.points))[cluster_indices == closest_cluster_id]
        pcd = pcd.select_by_index(inliers)

        return pcd, inliers

    @staticmethod
    def show_model(pcd: o3d.geometry.PointCloud, extra_points: np.array = None, base_plane: Plane = None, other_geometries = None):
        geometries = [pcd]
        if other_geometries is not None:
            geometries.extend(other_geometries)

        if extra_points is not None:
            extra_pcd = o3d.geometry.PointCloud()
            extra_pcd.points = o3d.utility.Vector3dVector(extra_points)
            extra_pcd.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0, 0], (len(extra_points), 1)))

            geometries.append(extra_pcd)

        if base_plane is not None:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
            frame.translate(base_plane.center)
            frame.rotate(align_vectors(np.array([0, 0, 1]), base_plane.normal), center=base_plane.center)

            geometries.append(frame)

            o3d.visualization.draw_geometries(geometries, lookat=base_plane.center, up=base_plane.normal, front=np.array([1.0, 0.0, 0.0]), zoom=0.1)
        else:
            o3d.visualization.draw_geometries(geometries)
