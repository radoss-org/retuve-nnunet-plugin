import os

import numpy as np
import torch
from radstract.data.dicom import convert_dicom_to_images
from retuve.keyphrases.config import Config

from .utils import (
    FILEDIR,
    generate_segmentation_objects,
    get_or_create_predictor,
    preprocess_image,
)

MODEL_LOCATION_ULTRASOUND = f"{FILEDIR}/weights/hip-nnunet-us.zip"


torch.serialization.add_safe_globals(
    [
        np.core.multiarray.scalar,
        np.dtypes.Float64DType,
        np.dtype,
        np.dtypes.Float32DType,
    ]
)


def get_nnunet_model_us(config=None):
    return get_or_create_predictor(
        MODEL_LOCATION_ULTRASOUND, "Dataset656_PublicHipUS", config
    )


def nnunet_predict_dcm_us(dcm, keyphrase, model=None):
    config = Config.get_config(keyphrase)

    dicom_images = convert_dicom_to_images(
        dcm,
        crop_coordinates=config.crop_coordinates,
        dicom_type=config.dicom_type,
    )

    return nnunet_predict_us(dicom_images, config, model)


def nnunet_predict_us(imgs, keyphrase, model=None):
    return nnunet_predict_us_shared(imgs, keyphrase, model, MODEL_LOCATION_ULTRASOUND)


def nnunet_predict_us_shared(imgs, keyphrase, predictor=None, model_loc=None):
    config = Config.get_config(keyphrase)
    if predictor is None:
        predictor = get_or_create_predictor(model_loc, "Dataset656_PublicHipUS", config)

    os.environ["nnUNet_results"] = f"{FILEDIR}/nnunet-processing/nnUNet_trained_models"

    props = {"spacing": [1, 1, 1]}
    seg_results = []

    for i, img in enumerate(imgs):
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
        print(f"Found in frame {i+1}/{len(imgs)}: {found_in_this_result}")

        seg_results.append(seg_frame_objects)

    return seg_results
