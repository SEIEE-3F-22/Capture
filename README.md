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
* Copy generated 'Capture.*.so' file which can be found in the 'test' folder into the same folder where your '.py' file locates.
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

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
```