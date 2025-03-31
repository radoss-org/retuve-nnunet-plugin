import os

import numpy as np
from retuve.classes.seg import SegFrameObjects
from retuve.hip_xray.classes import LandmarksXRay
from retuve.keyphrases.config import Config

from .utils import (
    FILEDIR,
    fit_triangle_to_mask,
    generate_segmentation_objects,
    get_or_create_predictor,
    preprocess_image,
)

MODEL_LOCATION_XRAY = f"{FILEDIR}/weights/hip-nnunet-xray.zip"


def get_nnunet_model_xray(config=None):
    return get_or_create_predictor(
        MODEL_LOCATION_XRAY, "Dataset657_PublicHipXray", config
    )


def nnunet_predict_xray(images, keyphrase, model=None, stream=False):
    config = Config.get_config(keyphrase)
    if model is None:
        model = get_nnunet_model_xray(config)

    os.environ["nnUNet_results"] = f"{FILEDIR}/nnunet-processing/nnUNet_trained_models"

    props = {"spacing": [1, 1, 1]}
    final_images = [preprocess_image(img, (384, 384)) for img in images]
    final_images = np.array(final_images)

    rets = [
        model.predict_single_npy_array(np.expand_dims(image, axis=0), props).squeeze()
        for image in final_images
    ]

    seg_results = []
    for ret, img in zip(rets, images):
        seg_frame_objects = SegFrameObjects(img=np.array(img))

        ret_2 = ret.copy()
        ret_2[:, ret_2.shape[1] // 2 :] = 0
        ret[:, : ret.shape[1] // 2] = 0

        for ret_x in [ret, ret_2]:

            seg_frame_objects_sub = generate_segmentation_objects(
                img, ret_x, [1], (img.width, img.height)
            )

            seg_frame_objects.append(seg_frame_objects_sub[0])

        seg_results.append(seg_frame_objects)

    landmark_results = []
    for seg_frame_objects in seg_results:
        landmarks = LandmarksXRay()
        if len(seg_frame_objects) != 2:
            landmark_results.append(landmarks)
            continue

        tri_1, tri_2 = seg_frame_objects[0], seg_frame_objects[1]
        fem_l, pel_l_o, pel_l_i, fem_r, pel_r_o, pel_r_i = fit_triangle_to_mask(
            tri_1.points, tri_2.points
        )

        if fem_l is not None:
            landmarks.fem_l, landmarks.pel_l_o, landmarks.pel_l_i = (
                pel_l_i,
                pel_l_o,
                pel_l_i,
            )
            landmarks.fem_r, landmarks.pel_r_o, landmarks.pel_r_i = (
                pel_r_i,
                pel_r_o,
                pel_r_i,
            )

        landmark_results.append(landmarks)

    return landmark_results, seg_results
