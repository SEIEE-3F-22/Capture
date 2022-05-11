import cv2
import time
from Capture import *
from multiprocessing import Process

def run(capture):
    capture.run()

if __name__ == '__main__':
    capture = Capture(0)
    p = Process(target=run, args=(capture,))
    p.daemon = True
    p.start()
    time.sleep(1)

    while True:
        frame = capture.get()
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break