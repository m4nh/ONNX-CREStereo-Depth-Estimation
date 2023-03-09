import cv2
import numpy as np
from crestereo import CREStereo
import time
from pipelime.sequences import SamplesSequence
from rich.progress import track
import typer
import albumentations as A

# class Timer with context
class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        print(
            "[{}] Elapsed: {:.2f} seconds".format(self.name, time.time() - self.tstart)
        )


def name(
    input_folder: str = typer.Option(
        ...,
        "--input-folder",
        "-i",
        help="Path to the input folder",
    ),
):

    # Model Selection options (not all options supported together)
    iters = 5  # Lower iterations are faster, but will lower detail.
    # Options: 2, 5, 10, 20

    shape = (240, 320)  # Input resolution.
    # Options: (240,320), (320,480), (380, 480), (360, 640), (480,640), (720, 1280)

    version = "init"  # The combined version does 2 passes, one to get an initial estimation and a second one to refine it.
    # Options: "init", "combined"

    # Initialize model
    model_path = f"models/crestereo_{version}_iter{iters}_{shape[0]}x{shape[1]}.onnx"
    depth_estimator = CREStereo(model_path)

    dataset = SamplesSequence.from_underfolder(input_folder)

    # Augmentations
    T = A.Compose(
        [
            A.SmallestMaxSize(max_size=shape[0]),
            A.CenterCrop(height=shape[0], width=shape[1]),
        ]
    )

    # Estimate the depth
    for sample in dataset:
        left = sample["left_rect_img"]()
        right = sample["right_rect_img"]()

        left = T(image=left)["image"]
        right = T(image=right)["image"]

        with Timer("Estimate disparity"):
            disparity_map = depth_estimator(left, right)

        color_disparity = depth_estimator.draw_disparity()

        stack = np.hstack((left, right, color_disparity))

        cv2.imshow("stack", stack)
        cv2.waitKey(0)

        print(disparity_map.shape)


if __name__ == "__main__":
    typer.run(name)
