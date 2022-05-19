# Capture

Camera capture and fisheye undistort python bindings of c++ code via pybind11.

### Usage

* Clone this repository into your local folder.
* Run commands below to initialize pybind11 submodule.

```shell
git submodule init
git submodule update
```

* Run commands below to generate python library.

```shell
mkdir build && cd build
cmake .. && make
```

* Copy generated 'Capture.*.so' file which can be found in the 'test' folder into the same folder where your '.py' file
  locates.
* Import Capture library into your project and now you are able to use it. An example is shown below:

```python
import cv2
import time
import Capture
from threading import Thread


def run():
    Capture.run()


if __name__ == '__main__':
    p = Thread(target=run)
    p.daemon = True
    p.start()

    while True:
        frame = Capture.get()
        if frame.shape[0] == 0:
            time.sleep(0.005)
            continue

        inference_result = Capture.getInferenceResult()
        print(inference_result)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
```

* Inference results will be returned in format like

```python
[label1, x1, y1, w1, h1, prob1, label2, x2, y2, w2, h2, prob2, ...]
```

* Indexes and corresponding labels are listed in table below

| index |        label        |
|:-----:|:-------------------:|
|   0   |   colorful circle   |
|   1   |      turn left      |
|   2   |     speed limit     |
|   3   |     turn right      |
|   4   | remove speed limit  |

