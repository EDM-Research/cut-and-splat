import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import albumentations as A


def augment_tensor(rendering):
    rendering_np = (rendering.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    augmented_rendering = augment_np(rendering_np) / 255.0

    rendering = torch.Tensor(augmented_rendering).to("cuda").permute(-1, 0, 1)

    return rendering

def correct_mask(mask):
    mask_np = (mask.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)

    mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)

    contour, hier = cv2.findContours(mask_np, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        cv2.drawContours(mask_np, [cnt], 0, 255, -1)

    mask_np = mask_np.reshape(mask_np.shape[0], mask_np.shape[1], 1)
    mask_np = np.repeat(mask_np, 3, axis=2)

    mask = torch.Tensor(mask_np).to("cuda").permute(-1, 0, 1).to(torch.float32)

    return mask

def augment_np(rendering):

    transforms = A.SomeOf([
        #A.GaussianBlur(blur_limit=(3, 5)),
        #A.MotionBlur(blur_limit=5),
        A.RandomToneCurve(scale=0.2),
        A.GaussNoise(),
        A.ISONoise(),
        A.MultiplicativeNoise(),
        A.RandomGamma(),
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4)
    ], n=2)

    augmented_rendering = transforms(image=rendering)["image"]

    return augmented_rendering


def resize_keep_aspect_ratio(image: torch.Tensor, width: int, height: int):
    """
    Resize an image Tensor to the specified width and height.
    The aspect ratio is kept consistent with the original image.
    Zero padding is added to the resulting image to keep the aspect ratio consistent.

    Args:
        image (torch.Tensor): Input image tensor of shape (C, W, H).
        width (int): Desired width of the resized image.
        height (int): Desired height of the resized image.

    Returns:
        torch.Tensor: Resized image tensor.
    """
    # Get original dimensions
    original_width, original_height = image.shape[1], image.shape[2]

    # Calculate aspect ratios
    aspect_ratio = original_width / original_height
    target_aspect_ratio = width / height

    # Resize based on aspect ratio
    if aspect_ratio < target_aspect_ratio:
        new_width = int(height * aspect_ratio)
        resized_image = F.interpolate(image.unsqueeze(0), size=(new_width, height), mode='bilinear',
                                      align_corners=False)
        diff = width - new_width
        padding_left = diff // 2
        padding_right = diff - padding_left
        resized_image = F.pad(resized_image, (0, 0, padding_left, padding_right), mode='constant', value=0)
    else:
        new_height = int(width / aspect_ratio)
        resized_image = F.interpolate(image.unsqueeze(0), size=(width, new_height), mode='bilinear',
                                      align_corners=False)
        diff = height - new_height
        padding_top = diff // 2
        padding_bottom = diff - padding_top
        resized_image = F.pad(resized_image, (padding_top, padding_bottom, 0, 0), mode='constant', value=0)

    return resized_image.squeeze(0)


def random_crop(image: np.array, depth: np.array, crop_width: int, crop_height: int) -> (np.array, np.array):
    """
    Take a random crop of the given size from both the image and depth map. The same crop is taken from both.
    The image has three channels and the depth map has one.
    """
    img_height, img_width, _ = image.shape

    #print(f"Image size: {img_width}, {img_height} \t crop size: {crop_width}, {crop_height}")

    smallest_scale = min(max(crop_height / img_height, crop_width / img_width), 0.98)
    scale_factor = random.uniform(smallest_scale, 0.99)
    image = cv2.resize(image, fx=scale_factor, fy=scale_factor, dsize=None)

    if depth is not None:
        depth_height, depth_width = depth.shape
        assert img_height == depth_height and img_width == depth_width, "Image and depth map must have the same dimensions"
        depth = cv2.resize(depth, fx=scale_factor, fy=scale_factor, dsize=None)

    img_height, img_width, _ = image.shape

    # Ensure the crop size is valid
    if crop_width > img_width or crop_height > img_height:
        raise ValueError("Crop size must be smaller than the dimensions of the image and depth map")

    # Randomly select the top-left corner of the crop
    x = np.random.randint(0, img_width - crop_width + 1)
    y = np.random.randint(0, img_height - crop_height + 1)

    # Take the crop from the image
    cropped_image = image[y:y + crop_height, x:x + crop_width]

    # Take the crop from the depth map if it is not None
    cropped_depth = None
    if depth is not None:
        cropped_depth = depth[y:y + crop_height, x:x + crop_width]

    return cropped_image, cropped_depth


def center_crop(rgb_image, depth_map, target_width, target_height):
    """
    Center crops the given RGB image and depth map (if provided) to the specified width and height.

    Parameters:
    rgb_image (numpy.ndarray): The RGB image to crop.
    depth_map (numpy.ndarray or None): The depth map to crop, or None if no depth map is provided.
    target_width (int): The width of the cropped image.
    target_height (int): The height of the cropped image.

    Returns:
    tuple: Cropped RGB image, and cropped depth map (or None if depth map was not provided).
    """

    def center_crop_image(image, target_width, target_height):
        """
        Center crops the given image to the specified width and height.

        Parameters:
        image (numpy.ndarray): The image to crop.
        target_width (int): The width of the cropped image.
        target_height (int): The height of the cropped image.

        Returns:
        numpy.ndarray: The cropped image.
        """
        h, w = image.shape[:2]
        start_x = (w - target_width) // 2
        start_y = (h - target_height) // 2
        return image[start_y:start_y + target_height, start_x:start_x + target_width]

    # Crop the RGB image
    rgb_image = cv2.resize(rgb_image, fx=0.7, fy=0.7, dsize=None)
    cropped_rgb = center_crop_image(rgb_image, target_width, target_height)

    # Crop the depth map if provided
    cropped_depth = None
    if depth_map is not None:
        depth_map = cv2.resize(depth_map, fx=0.7, fy=0.7, dsize=None)
        cropped_depth = center_crop_image(depth_map, target_width, target_height)

    return cropped_rgb, cropped_depth