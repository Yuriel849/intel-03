# !/usr/bin/env python3
'''
"Python code for operating smart factory"
'''

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np
# from openvino.inference_engine import IECore
import openvino as ov

from iotdemo import ColorDetector, FactoryController, MotionDetector

FORCE_STOP = False


def thread_cam1(q):  # thread 변환
    # MotionDetector
    det = MotionDetector()
    det.load_preset("./resources/motion.cfg", "default")

    # Load and initialize OpenVINO
    core = ov.Core()
    model = core.read_model("./resources.openvino.xml")

    # HW2 Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture('resources/conveyor.mp4')
    start_flag = True
    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # Enqueue "VIDEO:Cam1 live", frame info
        q.put(("VIDEO:Cam1 live", frame))

        # Motion detect
        detected = det.detect(frame)
        if detected is None:
            continue

        # Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(("VIDEO:Cam1 detected", detected))

        # Inference OpenVINO
        input_tensor = np.expand_dims(detected, 0)

        if start_flag is True:
            ppp = ov.preprocess.PrePostProcessor(model)
            ppp.input().tensor() \
                .set_shape(input_tensor.shape) \
                .set_element_type(ov.Type.u8) \
                .set_layout(ov.Layout('NHWC'))
            ppp.input().preprocess() \
                .resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
            ppp.input().model().set_layout(ov.Layout('NCHW'))
            ppp.output().tensor().set_element_type(ov.Type.f32)

            model = ppp.build()
            compiled_model = core.compile_model(model, "CPU")
            start_flag = False

        results = compiled_model.infer_new_request({0: input_tensor})
        predictions = next(iter(results.values()))
        probs = predictions.reshape(-1)

        if probs[0] > 0.0:
            print("Bad Item")
            # in queue for moving the actuator 1
            q.put(("PUSH", 1))
        else:
            print("Good Item")

        # Calculate ratios
        # print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")
    cap.release()
    q.put(('Finish', None))
    exit()


def thread_cam2(q):
    # MotionDetector
    det = MotionDetector()
    det.load_preset("./resources/motion.cfg", "default")

    # ColorDetector
    color = ColorDetector()
    color.load_preset("./resources/color.cfg", "default")

    # HW2 Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture('resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put(("VIDEO:Cam2 live", frame))

        # Detect motion
        detected = det.detect(frame)
        if detected is None:
            continue

        # Enqueue "VIDEO:Cam2 detected", detected info.
        q.put(("VIDEO:Cam2 detected", detected))

        # Detect color
        predict = color.detect(detected)

        # Compute ratio
        name, ratio = predict[0]
        ratio *= 100
        print(f"{name}: {ratio:.2f}%")

        # Enqueue to handle actuator 2
        if name == "blue" and ratio > .5:
            q.put(("PUSH", 2))

    cap.release()
    q.put(('DONE', None))
    exit()


def imshow(title, frame, pos=None):  # 화면에 출력
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


def main():
    force_stop = False

    parser = ArgumentParser(prog='python3 factory.py',
                            description="Factory tool")

    parser.add_argument("-d",
                        "--device",
                        default=None,
                        type=str,
                        help="Arduino port")
    args = parser.parse_args()

    # HW2 Create a Queue
    que = Queue()

    # HW2 Create thread_cam1 and thread_cam2 threads and start them.
    thread1 = threading.Thread(target=thread_cam1, args=(que, ), daemon=True)
    thread2 = threading.Thread(target=thread_cam2, args=(que, ), daemon=True)

    thread1.start()
    thread2.start()

    # main, join 콜을 하여 thread 1과 thread 2가 잘 끝났는지 기다렸다가 종료.
    # Q. 왜 굳이 main thread를 두어야 할까? 그냥 cam1 실행, cam2 실행하면 안 되는 걸까?
    # 화면 업데이트 때문이다.
    # 화면이라는 리소스는 하나인데 여러 thread에서 업데이트를 하려고 하면 에러가 발생할 수 있다.

    with FactoryController(args.device) as ctrl:
        while not force_stop:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # HW2 get an item from the queue.
            # You might need to properly handle exceptions.
            # de-queue name and data
            try:
                event = que.get_normal()
            except Empty:
                continue

            # HW2 show videos with titles of
            # 'Cam1 live' and 'Cam2 live' respectively.
            name, data = event

            if name.startswith("VIDEO:"):
                imshow(name[6:], data)
            elif name == "PUSH":
                # Control actuator, name == 'PUSH'
                ctrl.push_actuator(data)
            elif name == "DONE":
                force_stop = True
    thread1.join()
    thread2.join()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("EXCEPTION : KeyboardInterrupt")
    finally:
        os._exit()
