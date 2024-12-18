import json
import os
import random
import numpy as np

from cut_and_splat.model_extractor import ModelExtractor
from cut_and_splat.utils.camera import camera_from_json
from cut_and_splat.utils.model import model_subset, get_model_base, adjust_ground_plane, merge_models
from scene import GaussianModel
from cut_and_splat.utils.geometry import Plane


class SimulationPipeline:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


class GaussianModelWrapper:
    def __init__(self, model_folder: str = None, use_cached=True):
        self.original_gaussians = GaussianModel(sh_degree=3)
        checkpoint_path = self.get_last_checkpoint(model_folder)
        print(f"loading {checkpoint_path}")
        self.original_gaussians.load_ply(checkpoint_path)

        model_extractor = ModelExtractor()

        with open(os.path.join(model_folder, "cameras.json"), 'r') as f:
            camera_data = json.load(f)[0]
            self.example_camera = camera_from_json(camera_data)

        foreground_cache = os.path.join(model_folder, "foreground.json")
        if use_cached and os.path.exists(foreground_cache):
            print("Loading cached foreground object")
            with open(foreground_cache, 'r') as f:
                foreground_data = json.load(f)
                indices = foreground_data["indices"]
                plane = Plane(
                    np.array(foreground_data["plane"]["normal"]),
                    np.array(foreground_data["plane"]["center"])
                )

            self.foreground_model = model_subset(self.original_gaussians, indices=indices)
            self.ground_plane = plane
        else:
            self.foreground_model, self.ground_plane, indices = model_extractor.extract(self.original_gaussians, manual_preprocess=True)

            if use_cached:
                cache_dump = {
                    'indices': list(map(int, indices)),
                    'plane': {
                        'normal': list(map(float, list(self.ground_plane.normal))),
                        'center': list(map(float, list(self.ground_plane.center)))
                    }
                }
                with open(foreground_cache, 'w') as f:
                    json.dump(cache_dump, f)

        self.ground_plane = adjust_ground_plane(self.foreground_model, self.ground_plane)
        self.foreground_base = get_model_base(self.foreground_model, self.ground_plane)

    @staticmethod
    def get_last_checkpoint(model_folder: str):
        folders = list(os.listdir(os.path.join(model_folder, "point_cloud")))
        folders = sorted(folders, key=lambda x : int(x.split("_")[-1]))

        checkpoint_path = os.path.join(model_folder, "point_cloud", folders[-1], "point_cloud.ply")

        return checkpoint_path


class Simulation:
    def __init__(self, file, max_objects=3, use_cache=True):

        self.class_objects = {}

        self.max_objects = max_objects

        with open(file, 'r') as f:
            for line in f:
                name, *locations = line.rstrip().split(",")

                if name.startswith("#"):
                    continue

                print(f"Importing {name}")

                self.class_objects[name] = []

                model_wrapper = GaussianModelWrapper(locations[0], use_cached=use_cache)

                for location in locations[1:]:
                    model_wrapper = merge_models(model_wrapper, GaussianModelWrapper(location, use_cached=use_cache))

                self.class_objects[name].append(model_wrapper)

    def spawn_objects(self):
        amount = random.randint(1, self.max_objects)
        classes = random.choices(list(self.class_objects.keys()), k=amount)

        models = []
        for c in classes:
            models.append(random.choice(self.class_objects[c]))

        return models, classes
