# Retuve nnUNet Segmentation AI Plugin

![tests](https://github.com/radoss-org/retuve-nnunet-plugin/actions/workflows/test.yml/badge.svg)

__For more information on Retuve, see https://github.com/radoss-org/retuve__

This codebase has the nnUNet AI Plugin for Retuve, which uses Radiopedia data from [The Open Hip Dataset](https://github.com/radoss-org/open-hip-dysplasia) to train.

The model weights are strictly under the **terms of the CC BY-NC-SA 3.0 license**. This is because the model is trained on Radiopedia Data, which is under the CC BY-NC-SA 3.0 license.

This means that you cannot use this codebase for any commercial purposes and you must attribute Radiopedia for the data used to train the model.

The codes for the licence can be found in the  [LICENSE](LICENSE) file.

## Installation

To install the plugin, you can use the following command:

```bash
pip install git+https://github.com/radoss-org/retuve-nnunet-plugin.git
```

## Example Usage

Please see https://github.com/radoss-org/retuve/tree/main/examples for more examples. This is purely meant to illustrate how to use the plugin.

```python
import pydicom
from retuve.defaults.hip_configs import default_US
from retuve.funcs import analyse_hip_3DUS
from retuve.testdata import Cases, download_case

from retuve_nnunet_plugin.ultrasound import nnunet_predict_dcm_us

# Get an example case
dcm_file = download_case(Cases.ULTRASOUND_DICOM)[0]

default_US.device = "cpu"

dcm = pydicom.dcmread(dcm_file)

hip_datas, *_ = analyse_hip_3DUS(
    dcm,
    keyphrase=default_US,
    modes_func=nnunet_predict_dcm_us,
    modes_func_kwargs_dict={},
)

print(hip_datas)
```

## Attribution

We give full attribution to the authors that made this effort possible on Radiopedia. The list of these authors can be found [here](https://github.com/radoss-org/open-hip-dysplasia/tree/main/radiopedia_ultrasound_2d#attribution).

## License

The codes for the licence can be found in the  [LICENSE](LICENSE) file.

If you are interested in a less-restritive licence, the first step is to [contact Radiopedia](https://radiopaedia.org/licence?lang=gb#obtaining_a_license) for a special licence to use all the data this model is trained on. That list can be found [here](https://github.com/radoss-org/open-hip-dysplasia/tree/main/radiopedia_ultrasound_2d#attribution).

RadOSS will then consider providing you a commercial licence for this plugin at no charge. Please contact us at info@radoss.org when you have obtained the licence from Radiopedia.

## Citation

If you use this plugin, please cite the following:

```
@article{Isensee2021,
    author = {Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. and Petersen, Jens and Maier-Hein, Klaus H.},
    doi = {10.1038/s41592-020-01008-z},
    journal = {Nature methods},
    number = {2},
    pages = {203--211},
    publisher = {Nature Publishing Group},
    title = {nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
    volume = {18},
    year = {2021}
}

@misc{radiopaedia_ddh_cases,
  author = {Sheikh, Yusra and Thibodeau, Ryan and Ranchod, Ashesh Ishwarlal and
            Hisham},
  title = {Radiopaedia cases of Developmental Dysplasia of the Hip},
  year = {2023-2024},
  howpublished = {\url{https://radiopaedia.org/}},
  note = {Cases: 72628 (Yusra Sheikh), 172535-172536, 172658, 172534, 171555-171556, 172533, 171551, 171553-171554 (Ryan Thibodeau), 167854-167855, 167857 (Ashesh Ishwarlal Ranchod), 56568 (Hisham Alwakkaa); Accessed: [Date of access]}
}
```