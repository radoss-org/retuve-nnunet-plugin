import os
import warnings

import pytest
from retuve.defaults.hip_configs import default_US

default_US.register("ultrasound", live=True)


@pytest.fixture(scope="session", autouse=True)
def set_env():
    os.environ["RETUVE_DISABLE_WARNING"] = "True"


# Suppress specific DeprecationWarnings during tests
@pytest.fixture(autouse=True)
def suppress_warnings():
    warnings.filterwarnings("ignore")
