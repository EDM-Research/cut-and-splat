import argparse
import json
import random

from tqdm import tqdm
import torch
import os

from cut_and_splat.background_loader import BackgroundLoader
from cut_and_splat.utils.folder import create_or_empty_folders
from gaussian_renderer import render
import torchvision
from torchvision.transforms import functional as F

from cut_and_splat.utils.camera import generate_directed_view, generate_random_view, filter_depth, generate_random_camera
from cut_and_splat.utils.geometry import align_vectors, sample_sphere
from cut_and_splat.utils.image import augment_tensor, resize_keep_aspect_ratio, correct_mask
from cut_and_splat.simulation import Simulation, SimulationPipeline
from scene.cameras import Camera
from utils.general_utils import set_seeds

BACKGROUND_FOLDER = r"D:\Datasets\COCO\test2017"
BACKGROUND_SIZES = None


def render_annotations(simulation: Simulation, output_path, pipeline, image_count: int = 5000, augment_foreground=False,
                       try_smart_placement=True, random_placement_prob=0.2, random_rotation: bool = False):
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    background_folder = BackgroundLoader(BACKGROUND_FOLDER, smart_placement=try_smart_placement, target_size=BACKGROUND_SIZES)

    render_path, mask_path = create_or_empty_folders(output_path)

    annotations = {
        "images": []
    }

    print(f"Rendering images\n")
    for image_no in tqdm(range(image_count)):
        smart_placement = try_smart_placement

        # Select foreground models and background scene
        models, names = simulation.spawn_objects()

        # Get background image and potential placement planes
        bg, *found_placements = background_folder.get(len(models))

        # Check if placements are found, if not use random placement
        if found_placements[0] is None:
            smart_placement = False
        else:
            # sometimes skip the smart placement to add more variation
            if random.uniform(0.0, 1.0) < random_placement_prob:
                smart_placement = False
            else:
                scene_normal, placements, depth = found_placements

        bg = torch.Tensor(bg).to("cuda")
        height = bg.size(0)
        width = bg.size(1)
        bg = bg.permute(-1, 0, 1) / 255.0

        object_annotations = []
        for j, model in enumerate(models):
            # Create camera parameters
            target_height = model.example_camera.image_height
            target_width = model.example_camera.image_width
            fovx, fovy = None, None

            if smart_placement:
                placement = placements[j]

                # Create camera viewpoint aligning object with plane
                R = align_vectors(placement.up, model.ground_plane.normal)
                dir = R @ placement.direction
                view = generate_directed_view(model.example_camera, model.foreground_base, model.ground_plane, dir, image_no, height=target_height, width=target_width, up=model.ground_plane.normal, fovx=fovx, fovy=fovy, random_rot=random_rotation)
            else:
                view = generate_random_camera(model.example_camera, model.foreground_base, model.ground_plane, image_no, height=target_height, width=target_width)

            # Render foreground opacity mask in white
            colors = torch.Tensor([1.0, 1.0, 1.0]).unsqueeze(0).repeat(len(model.foreground_model.get_xyz), 1).to("cuda")
            mask = render(view, model.foreground_model, pipeline, background, override_color=colors)["render"]

            # Render foreground colors
            campos = torch.Tensor(sample_sphere([0, 0, 0], 10.0)).cuda() if augment_foreground else None
            rendering = render(view, model.foreground_model, pipeline, background, override_campos=campos)["render"]
            rendering = torch.clamp(rendering, 0.0, 1.0)

            rendering = resize_keep_aspect_ratio(rendering, height, width)
            mask = resize_keep_aspect_ratio(mask, height, width)

            if smart_placement:
                # Shift foreground to target position
                translation = [placement.pos_2d[1] - rendering.size(2) // 2, placement.pos_2d[0] - rendering.size(1) // 2]
                angle = 0.0
            else:
                # Shift foreground to random position
                translation = [random.randint(-bg.size(2) // 2, bg.size(2) // 2), random.randint(-bg.size(1) // 2, bg.size(1) // 2)]
                angle = random.uniform(0.0, 360.0)

            # put the rendered object in place
            rendering = F.affine(rendering, translate=translation, angle=angle, scale=1.0, shear=[0.0, 0.0])
            mask = F.affine(mask, translate=translation, angle=angle, scale=1.0, shear=[0.0, 0.0])

            if smart_placement:
                # Check for occlusions with foreground
                rendering, mask = filter_depth(rendering, mask, depth, depth[placement.pos_2d[0]][placement.pos_2d[1]])

            if augment_foreground:
                rendering = augment_tensor(rendering)

            # Blend fore- and background
            bg = (1.0 - mask) * bg + mask * rendering

            # Threshold opacity map to create binary mask
            mask = (mask > 0.5).to(torch.float32)
            mask = correct_mask(mask)
            torchvision.utils.save_image(mask, os.path.join(mask_path, f"{image_no}_{j}.png"))

            object_annotations.append({
                "class": names[j],
                "mask": f"masks/{image_no}_{j}.png"
            })

        torchvision.utils.save_image(bg, os.path.join(render_path, f"{image_no}.png"))
        annotations["images"].append({
            "image": f"renders/{image_no}.png",
            "objects": object_annotations
        })

        image_no += 1

    with open(os.path.join(output_path, "annotations.json"), 'w') as f:
        json.dump(annotations, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--model-file", type=str, default='models.txt')
    parser.add_argument("--image-count", type=int, default=5000)
    parser.add_argument("--max-objects", type=int, default=3)
    parser.add_argument("--augment-foreground", action='store_true')
    parser.add_argument("--random-rotation", action='store_true')
    parser.add_argument("--no-smart-placement", action='store_true')

    args = parser.parse_args()

    os.environ['DEBUG'] = '0'

    with torch.no_grad():
        sim = Simulation(args.model_file, max_objects=args.max_objects)

        # Initialize system state (RNG)
        set_seeds()

        render_annotations(sim, f"{args.output_folder}", SimulationPipeline(), image_count=args.image_count,
                           augment_foreground=args.augment_foreground,
                           try_smart_placement=not args.no_smart_placement, random_rotation=args.random_rotation
                           )