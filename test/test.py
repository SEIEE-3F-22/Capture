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

    timeStamp = time.time()
    while True:
        frame = Capture.get()
        if frame.shape[0] == 0:
            time.sleep(0.005)
            continue

        lastTimeStamp = timeStamp
        timeStamp = time.time()
        print(1 / (timeStamp - lastTimeStamp))

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
