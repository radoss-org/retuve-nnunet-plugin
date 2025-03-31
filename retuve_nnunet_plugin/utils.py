import os

import cv2
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from PIL import Image
from radstract.datasets.polygon_utils import segmentation_to_polygons
from retuve.classes.seg import SegFrameObjects, SegObject
from retuve.hip_us.classes.enums import HipLabelsUS

FILEDIR = os.path.abspath(f"{os.path.dirname(os.path.abspath(__file__))}/")


# Function to initialize and save the predictor if not already present
def get_or_create_predictor(model_location, name, config=None):

    if config is None:
        device = "cpu"
    else:
        device = config.device

    os.environ["nnUNet_results"] = f"{FILEDIR}/nnunet-processing/nnUNet_trained_models"

    # make dirs recursively
    os.makedirs(os.environ["nnUNet_results"], exist_ok=True)

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.model_sharing.model_export import export_pretrained_model
    from nnunetv2.model_sharing.model_import import install_model_from_zip_file

    needs_export = True
    if os.path.exists(model_location):
        install_model_from_zip_file(model_location)
        needs_export = False

    # Instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.2,
        use_gaussian=False,
        use_mirroring=False,
        perform_everything_on_device=False,
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )

    # Initialize the network architecture, load the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(
            f"{FILEDIR}/nnunet-processing/nnUNet_trained_models",
            f"{name}/nnUNetTrainer_50epochs__nnUNetPlans__2d",
        ),
        use_folds=(0,),
        checkpoint_name="checkpoint_final.pth",
    )

    if needs_export:
        export_pretrained_model(
            dataset_name_or_id=name,
            output_file=model_location,
            configurations=("2d",),
            trainer="nnUNetTrainer_50epochs",
            folds=(0,),
        )

    return predictor


def preprocess_image(img, standard_size):
    """
    Preprocesses an input image by resizing, converting to grayscale if necessary,
    and adding the appropriate batch and channel dimensions.

    Parameters:
    - img: PIL.Image object, the input image.
    - standard_size: Tuple (width, height) for resizing.

    Returns:
    - Preprocessed numpy array.
    """
    img = img.resize(standard_size, Image.NEAREST)
    img = np.array(img).squeeze()

    if img.ndim == 3 and img.shape[-1] > 1:
        img = img[..., 0]  # Convert to grayscale by taking the red channel

    img = np.expand_dims(img, axis=[0])  # Add channel dimension
    return img


def generate_segmentation_objects(img_original, ret, class_range, img_shape):
    """
    Generate segmentation objects from a prediction result.

    Parameters:
    - img_original: Original PIL.Image object.
    - ret: numpy array, the segmentation result.
    - class_range: Range of classes to process (e.g., range(1, 4)).
    - img_shape: Tuple (width, height) for resizing masks.

    Returns:
    - SegFrameObjects containing segmentation objects.
    """
    seg_frame_objects = SegFrameObjects(img=np.array(img_original))

    for clss in class_range:
        mask = (ret == clss).astype(np.uint8)  # Convert bools to 1's
        mask = cv2.resize(mask, img_shape, Image.NEAREST)

        mask_rgb_white = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * 255
        mask_rgb_red = mask_rgb_white.copy()  # Copy only if needed
        white_mask = (mask_rgb_white == [255, 255, 255]).all(axis=2)
        mask_rgb_red[white_mask] = [255, 0, 0]

        polygons = segmentation_to_polygons(mask_rgb_red)
        try:
            points = polygons[1][0]
            points_np = np.array(points, dtype=np.int32)
        except (IndexError, TypeError, KeyError):
            points = None  # or a default value

        box = None
        if points is not None and len(points) >= 3:
            x, y, w, h = cv2.boundingRect(points_np)
            box = np.array([x, y, x + w, y + h])

        if points is not None and len(points) >= 3:
            seg_obj = SegObject(points, clss - 1, mask_rgb_white, box=box)
            seg_obj.cls = HipLabelsUS(seg_obj.cls)
            seg_frame_objects.append(seg_obj)

    if len(seg_frame_objects) == 0:
        seg_frame_objects = SegFrameObjects.empty(img=np.array(img_original))

    return seg_frame_objects


def process_single_frame(args):
    """
    This function processes a single frame.
    It receives all necessary arguments in a tuple so it can run in a separate process.
    """
    i, img, predictor, props = args

    # --- Original per-frame logic ---
    img_original = img.copy()
    img_preprocessed = preprocess_image(img, (512, 448))

    ret = predictor.predict_single_npy_array(
        np.expand_dims(img_preprocessed, axis=0), props
    ).squeeze()

    seg_frame_objects = generate_segmentation_objects(
        img_original,
        ret,
        range(1, 4),
        (img_original.width, img_original.height),
    )

    found_in_this_result = [seg_obj.cls for seg_obj in seg_frame_objects]

    # Return everything needed to reconstruct order & final results
    return i, seg_frame_objects, found_in_this_result


def fit_triangle_to_mask(tri_1_points, tri_2_points):

    tri_1_points, _ = find_triangle_from_edges(tri_1_points)
    tri_2_points, _ = find_triangle_from_edges(tri_2_points)

    if tri_1_points is None or tri_2_points is None:
        return (None, None, None, None, None, None)

    most_left_point = 100000
    tri_left = None
    tri_right = None
    for point in tri_1_points + tri_2_points:
        if point[0] < most_left_point:
            most_left_point = point[0]
            tri_left = tri_1_points if point in tri_1_points else tri_2_points
            tri_right = tri_2_points if point in tri_1_points else tri_1_points

    # check that tri_left is the left triangle
    if tri_left[0][0] > tri_right[0][0]:
        tri_left, tri_right = tri_right, tri_left

    fem_l, pel_l_o, pel_l_i = define_points(tri_left)
    fem_r, pel_r_o, pel_r_i = define_points(tri_right)

    return fem_l, pel_l_o, pel_l_i, fem_r, pel_r_o, pel_r_i


def find_triangle_from_edges(points):
    # Means we are already passing in a processed triangle
    if len(points) == 3:
        triangle = np.array(points)
        return triangle, cv2.contourArea(triangle)

    contours = np.array([points], dtype=np.int32)

    if len(contours) == 0:
        return None, 0  # No contours found

    # Approximate the largest contour to a polygon
    epsilon = 0.1 * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)

    # Check if the approximated contour has 3 vertices (triangle)
    if len(approx) == 3:
        triangle = np.array([approx[0][0], approx[1][0], approx[2][0]])
        return triangle, cv2.contourArea(triangle)
    else:
        return None, 0  # No triangle found


def define_points(triangle):
    # Convert all points to tuples
    triangle = [(int(point[0]), int(point[1])) for point in triangle]

    # Find the lowest point in the triangle
    lowest_point = max(triangle, key=lambda point: point[1])
    triangle.remove(lowest_point)

    # Find the leftmost point in the triangle
    highest_point = min(triangle, key=lambda point: point[1])
    triangle.remove(highest_point)

    # The last point is the one not picked
    remaining_point = triangle[0]

    return lowest_point, highest_point, remaining_point
