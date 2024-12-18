import os
import glob
import pickle
import random
import time

import cv2
import numpy as np

from cut_and_splat.utils.folder import find_images_in_folder
from cut_and_splat.depth_detector import DepthDetector
from cut_and_splat.plane_finder import PlaneFinder, PlacementPlane
from cut_and_splat.utils.image import random_crop, center_crop


class Placement:
    """
    Class containing all information needed to render an object on a specific location on an image
    - 2D position on the image
    - Direction from point to camera in 3D space
    - Depth of point in original depth map
    - Up direction of plane object is standing on
    """
    def __init__(self, pos_2d: np.array, direction: np.array, up: np.array, depth: float):
        self.pos_2d = pos_2d
        self.direction = direction
        self.depth = depth
        self.up = up


class BackgroundLoader:
    """
    Class for loading background images from a folder and finding placement positions in those images.
    Images are found recursively in the folder
    """
    def __init__(self, folder: str, smart_placement: bool = True, min_resolution: int = None, target_size: tuple = None):
        self.folder = folder
        self.image_paths = find_images_in_folder(folder)
        self.min_resolution = min_resolution
        self.target_size = target_size

        self.smart_placement = smart_placement
        self.manual_normals = False     # Set to True in case you want to manually select normals for each scene
        self.scene_normals = {}
        self.depth_detector = DepthDetector()

        self.cache_planes = False
        self.plane_cache = {}

        if self.manual_normals:
            normal_cache_path = os.path.join(folder, 'normals.npz')
            if os.path.exists(normal_cache_path):
                loaded = np.load(normal_cache_path)
                self.scene_normals = {key: loaded[key] for key in loaded}
            else:
                self.scene_normals = self.compute_normals()
                np.savez(normal_cache_path, **self.scene_normals)

        if self.smart_placement:
            self.plane_finder = PlaneFinder(filter_top=True)

    def __len__(self):
        return len(self.image_paths)

    def find_or_load_depth(self, image: np.array, image_path: str):
        cached_depth_location = self.depth_detector.get_cache_location(image_path)

        # check if there is a cached depth, otherwise compute depth and save to file
        if os.path.exists(cached_depth_location):
            with open(cached_depth_location, 'rb') as f:
                depth = pickle.load(f)
        else:
            depth = self.depth_detector.detect(image)
            with open(cached_depth_location, 'wb') as f:
                pickle.dump(depth, f)

        return depth

    def compute_normals(self):
        normal_dict = {}
        for image_path in self.image_paths:
            _, filename = os.path.split(image_path)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            depth = self.find_or_load_depth(image, image_path)
            points = PlaneFinder.depth_to_points(depth)
            pcd = PlaneFinder.prepare_point_cloud(points)
            normal = PlaneFinder.get_manual_normal(pcd)
            normal_dict[filename] = normal
        return normal_dict


    @staticmethod
    def placement_on_plane(plane: PlacementPlane, depth: np.array) -> Placement:
        """
        Get a random placement position on the given plane
        """
        placement_index = random.randrange(0, len(plane.points_3d))
        pos_2d = plane.get_point_2d(placement_index)
        return Placement(
            pos_2d=pos_2d,
            direction=-plane.points_3d[placement_index],
            depth=depth[pos_2d[0], pos_2d[1]],
            up=plane.normal
        )

    def get(self, no_objects: int, image_index: int = None):
        """
        Loads an images from the target folder and finds placement positions on this image
        no_objects specifies the amount of placement positions that are needed
        if an image_index is specified that specific image is loaded, otherwise a random image is selected
        """
        image_path = random.choice(self.image_paths) if image_index is None else self.image_paths[image_index]
        _, filename = os.path.split(image_path)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not self.smart_placement:
            if self.target_size:
                image, _ = random_crop(image, None, self.target_size[0], self.target_size[1])
            return image, None

        if self.min_resolution is not None and min(image.shape[:-1]) < self.min_resolution:
            print(f"Image too small, reading new image")
            return self.get(no_objects)

        if self.cache_planes and os.path.basename(image_path) in self.plane_cache.keys():
            planes, scene_normal, depth = self.plane_cache[os.path.basename(image_path)]
            if self.target_size:
                image, _ = center_crop(image, None, self.target_size[0], self.target_size[1])
        else:

            depth = self.find_or_load_depth(image, image_path)

            if self.target_size:
                image, depth = center_crop(image, depth, self.target_size[0], self.target_size[1])

            # find planes in the depth map
            if self.manual_normals:
                override_normal = self.scene_normals[filename]
                planes, scene_normal = self.plane_finder.find_planes(depth, override_scene_normal=override_normal)

            else:
                planes, scene_normal = self.plane_finder.find_planes(depth)

            if self.cache_planes:
                self.plane_cache[os.path.basename(image_path)] = planes, scene_normal, depth

        if len(planes) == 0:
            return image, None

        # select the required amount of planes randomly, weighted by plane volume
        weights = [plane.volume for plane in planes]
        selected_planes = random.choices(planes, weights, k=no_objects)
        plane_indices = [planes.index(p) for p in selected_planes]
        placements = [self.placement_on_plane(plane, depth) for plane in selected_planes]

        return image, scene_normal, placements, depth
