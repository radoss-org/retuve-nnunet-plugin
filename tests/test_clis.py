from retuve.batch import run_single
from retuve.defaults.hip_configs import default_US
from retuve.keyphrases.enums import HipMode
from retuve.testdata import Cases, download_case

from retuve_nnunet_plugin.ultrasound import get_nnunet_model_us, nnunet_predict_dcm_us

default_US.batch.hip_mode = HipMode.US3D
default_US.batch.mode_func = nnunet_predict_dcm_us
default_US.device = "cpu"
default_US.batch.mode_func_args = {"model": get_nnunet_model_us(default_US)}
default_US.api.api_token = "password"


def test_single():
    dcm_file = download_case(Cases.ULTRASOUND_DICOM)[0]

    run_single(
        default_US,
        dcm_file,
    )


if __name__ == "__main__":
    test_single()
