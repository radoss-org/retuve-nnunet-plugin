import contextlib
import threading
import time

import requests
import uvicorn
from retuve.app import app
from retuve.app.helpers import app_init
from retuve.defaults.hip_configs import default_US
from retuve.keyphrases.enums import HipMode
from retuve.testdata import Cases, download_case

from retuve_nnunet_plugin.ultrasound import get_nnunet_model_us, nnunet_predict_dcm_us

default_US.batch.hip_mode = HipMode.US3D
default_US.batch.mode_func = nnunet_predict_dcm_us
default_US.device = "cpu"
default_US.batch.mode_func_args = {"model": get_nnunet_model_us(default_US)}
default_US.api.api_token = "password"


class Server(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()


def send_request(url, file_path, data):
    with open(file_path, "rb") as f:
        file_content = f.read()

    files = {
        "file": (
            file_path.split("/")[-1],
            file_content,
            "application/octet-stream",
        )
    }

    # Add additional form data
    form_data = {key: value for key, value in data.items()}

    response = requests.post(url, files=files, data=form_data)
    print(f"Status Code: {response.status_code}")

    try:
        response_body = response.json()
        print("Response Body:", response_body)
    except requests.exceptions.JSONDecodeError:
        print("Response Body is not JSON")

    assert response_body["metrics_3d"][0]["name"] == "alpha"
    assert response_body["metrics_3d"][0]["graf"] > 30

    assert response_body["metrics_3d"][1]["name"] == "coverage"
    assert response_body["metrics_3d"][1]["graf"] > 0.3


# Allows for with server.run_in_thread():
config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
server = Server(config=config)

app_init(app)


def test_data_creation():
    dcm_file = download_case(Cases.ULTRASOUND_DICOM)[0]

    tests = [
        {
            "file_path": dcm_file,
            "data": {
                "keyphrase": "ultrasound",
                "api_token": "password",
            },
        },
    ]

    with server.run_in_thread():
        url = "http://localhost:8000/api/model/"
        for test in tests:
            send_request(url, test["file_path"], test["data"])
