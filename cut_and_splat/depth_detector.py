import os

import cv2
import numpy as np
import torch

from cut_and_splat.utils.camera import get_intrinsics
from cut_and_splat.zoedepth.data.data_mono import preprocessing_transforms
from cut_and_splat.zoedepth.models.builder import build_model
from cut_and_splat.zoedepth.utils.config import get_config


class DepthDetector:
    """
    Class for finding depth in RGB images
    Based on DepthAnything and zoedepth
    The estimated depth is metric depth and not relative depth
    """
    def __init__(self):
        self.target_size = [980, 980]
        overwrite = {"pretrained_resource": 'local::checkpoints/depth_anything_metric_depth_indoor.pt',
                     'img_size': self.target_size}
        config = get_config('zoedepth', "eval", 'nyu', **overwrite)
        self.model = build_model(config)
        self.model = self.model.cuda()
        self.model.eval()

        self.transform = preprocessing_transforms('test')

    def detect(self, image: np.array):
        """
        Computes the metric depth from the given RGB image in the [0, 255] range
        """
        assert len(image.shape) == 3 and image.shape[-1] == 3, "Input is not a single RGB image in (H, W, C) format"
        image_input = image / 255.0
        image_input = image_input.astype(np.float32)

        original_size = image_input.shape[:2]
        focal, *_ = get_intrinsics(*original_size)

        sample = self.transform({
            'image': image_input,
            'focal': focal
        })

        depth = self.infer(sample['image'].cuda(), focal=focal)

        output = np.squeeze(depth.cpu().numpy())
        output = cv2.resize(output, (original_size[1], original_size[0]))

        if os.environ.get("DEBUG", '0') == '1':
            self.show_debug_visualization(output)

        return output

    @torch.no_grad()
    def infer(self, images: torch.Tensor, focal: float):
        """Inference with flip augmentation"""
        # images.shape = N, C, H, W
        if len(images.shape) == 3:
            images = torch.unsqueeze(images, 0)

        def get_depth_from_prediction(pred):
            if isinstance(pred, torch.Tensor):
                pred = pred  # pass
            elif isinstance(pred, (list, tuple)):
                pred = pred[-1]
            elif isinstance(pred, dict):
                pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
            else:
                raise NotImplementedError(f"Unknown output type {type(pred)}")
            return pred

        pred1 = self.model(images, dataset='nyu', focal=focal)
        pred1 = get_depth_from_prediction(pred1)

        pred2 = self.model(torch.flip(images, [3]), dataset='nyu', focal=focal)
        pred2 = get_depth_from_prediction(pred2)
        pred2 = torch.flip(pred2, [3])

        mean_pred = 0.5 * (pred1 + pred2)

        return mean_pred

    @staticmethod
    def get_cache_location(image_path: str) -> str:
        """
        Get the full path of the file containing the chached positions for this image
        """
        return image_path.split(".")[0] + "_depth.npy"

    @staticmethod
    def show_debug_visualization(depth: np.array):
        """
        Show the detected depth map
        """
        formatted = ((depth / np.max(depth)) * 255).astype(np.uint8)
        cv2.imshow("depth", formatted)
        cv2.waitKey(1)