import pydicom
from PIL import Image
from retuve.defaults.hip_configs import default_US, default_xray
from retuve.funcs import analyse_hip_3DUS, analyse_hip_xray_2D
from retuve.testdata import Cases, download_case

from retuve_nnunet_plugin.ultrasound import nnunet_predict_dcm_us
from retuve_nnunet_plugin.xray import nnunet_predict_xray


def test_ultrasound():
    dcm_file = download_case(Cases.ULTRASOUND_DICOM)[0]

    default_US.device = "cpu"

    dcm = pydicom.dcmread(dcm_file)

    hip_datas, *_ = analyse_hip_3DUS(
        dcm,
        keyphrase=default_US,
        modes_func=nnunet_predict_dcm_us,
        modes_func_kwargs_dict={},
    )

    hip_datas.grafs_hip.metrics[0].value > 0


def test_xray():

    jpg_file = download_case(Cases.XRAY_JPG)[0]

    default_xray.device = "cpu"

    img = Image.open(jpg_file)

    hip, *_ = analyse_hip_xray_2D(
        img,
        keyphrase=default_xray,
        modes_func=nnunet_predict_xray,
        modes_func_kwargs_dict={},
    )

    assert hip.metrics[0].value > 0
