import cv2
import numpy as np
import torch

from cut_and_splat.simulation import Simulation, SimulationPipeline
from cut_and_splat.utils.camera import generate_random_camera, generate_directed_view
from cut_and_splat.utils.geometry import sample_sphere
from cut_and_splat.utils.image import augment_tensor
from cut_and_splat.utils.model import show_model
from gaussian_renderer import render


def show_rendering(rendering: torch.Tensor, title: str):
    rendering = rendering.cpu().numpy()
    rendering = np.transpose(rendering, (1, 2, 0))
    rendering = np.clip(rendering, 0.0, 1.0)
    rendering = cv2.cvtColor(rendering, cv2.COLOR_RGB2BGR)
    cv2.imshow(title, rendering)


def save_rendering(rendering: torch.Tensor, filename: str):
    rendering = rendering.cpu().numpy()
    rendering = np.transpose(rendering, (1, 2, 0))
    rendering = np.clip(rendering, 0.0, 1.0)
    rendering = cv2.cvtColor(rendering, cv2.COLOR_RGB2BGR)
    rendering = (rendering * 255).astype(np.uint8)
    cv2.imwrite(filename, rendering)



if __name__ == "__main__":
    mode = 'splats'
    pipeline = SimulationPipeline()
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    with torch.no_grad():
        sim = Simulation("output/hope/models.txt", use_cache=True)

        for class_name in sim.class_objects.keys():
            models = sim.class_objects[class_name]
            for model in models:
                if mode == 'foreground':
                    points = []
                    for _ in range(10000):
                        #cam = generate_random_camera(model.example_camera, model.foreground_base, model.ground_plane, 0)
                        cam = generate_directed_view(model.example_camera, model.foreground_base, model.ground_plane, np.array([1.0, -1.0, 0]), 0, random_rot=True, up=model.ground_plane.normal)
                        point = cam.camera_center.cpu().numpy()
                        points.append(point[:3])

                    points.append(model.foreground_base)

                    show_model(model, extra_points=np.array(points))
                else:
                    #view = model.example_camera
                    view = generate_random_camera(camera=model.example_camera, center=model.foreground_base, ground_plane=model.ground_plane, image_no=0)

                    for i in range(1):
                        #view = generate_random_camera(camera=model.example_camera, center=model.foreground_base, ground_plane=model.ground_plane, image_no=0)
                        campos = torch.Tensor(sample_sphere([0,0,0], 10.0)).cuda()
                        rendering = render(view, model.foreground_model, pipeline, background, override_campos=None)["render"]
                        show_rendering(rendering, f"{class_name}_foreground.png")

                        rendering = render(view, model.original_gaussians, pipeline, background, override_campos=None)["render"]
                        show_rendering(rendering, f"{class_name}_total.png")