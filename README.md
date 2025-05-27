# Cut-and-Splat: Leveraging Gaussian Splatting for Synthetic Data Generation

This is the code for our ROBOVIS paper Cut-and-Splat: Leveraging Gaussian Splatting for Synthetic Data Generation.

```
Bram Vanherle, Brent Zoomers, Jeroen Put, Frank Van Reeth and Nick Michiels. Cut-and-Splat: Leveraging Gaussian Splatting for Synthetic Data Generation. International Conference on Robotics, Computer Vision and Intelligent Systems ROBOVIS 2025, 2025.
```


## IBSYD Dataset

For evaluation in the paper, the IBSYD dataset was introduced. [The dataset can be found here](https://drive.google.com/file/d/18UWG0VSNdqiWYCC3QLtRcdbnAb6Y5ItC/view?usp=drive_link).

## Installation

For ease of use and optimal forward compatibility, this code is designed as a drop-in extension of the original Gaussian Splatting code.
To get started, clone the [original Gaussian Splatting repository](https://github.com/graphdeco-inria/gaussian-splatting.git) and follow their installation tutorial.

Next, download the Cut-and-Splat code and drop it in the folder where you installed Gaussian Splatting.
Specifically the file `render_annotations.py`, the folder `cut_and_splat` and `gaussian_renderer/__init__.py`.

Some extra requirements are needed to run the Cut-and-Splat code:

```
pip install -r requirements.txt
```

We use DepthAnything for finding depth maps in background images.
Download the following two trained models and place them in `checkpoints/`.

- [depth_anything_metric_depth_indoor.pt](https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints_metric_depth/depth_anything_metric_depth_indoor.pt)
- [depth_anything_vitl14.pth](https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitl14.pth)

Additionally, download the [COCO tes set](https://cocodataset.org/#download) for sourcing the background images.
Set the `BACKGROUND_FOLDER` variable in `render_annotations.py` to the path of that dataset.

## Usage

To generate a dataset for one or more objects, train a Gaussian splatting model for each of those objects following the tutorial on the original repository.
Make sure to place the objects on a large flat surface for good background subtraction.

After doing so, add the objects to the `model.txt` file to specify which objects should be rendered to the final dataset.
Your model file could look like this:

```
bottle,output/3a63c51c-e
vase,output/2eb21784-a
plant,output/2eca5280-3
```

To render a dataset, the following command can be used. Please make sure to run this in the same python environment as Gaussian splatting.

```
python render_annotations.py --output-folder output/dataset --model-file models.txt --image-count 5000
```

This program also accepts the following arguments `--augment-foreground`, `-random-rotation` and `--no-smart-placement` to recreate the ablations from the paper.

This will create a dataset in the output folder that can be used for object detection and instance segmentation.
The dataset will have the following structure.
```
/dataset
    annotations.json
    /renders
        0.png
        1.png
        ...
    /masks
        0_0.png
        0_1.png
        1_0.png
        ...
```
The `annotations.json` file has the following structure.
```
{
    "images" : [
        {
            "image": "renders/0.png",
            "objects": [
                {"class": "bottle", "mask": "masks/0_0.png"},
                {"class": "plant", "mask": "masks/0_1.png"}
            ]
        },
        {
            "image": "renders/1.png",
            "objects": [
                {"class": "vase", "mask": "masks/1_0.png"}
            ]
        },
        ...
    ]
}
```

## Used libraries

These are some of the libraries, packages and repositories we used. Many thanks to these authors.

- [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting.git)
- PyTorch
- OpenCV
- Numpy
- [ZoeDepth](https://github.com/isl-org/ZoeDepth)
- [Open3D](http://www.open3d.org/docs/release/index.html)
