from imageai.Detection import VideoObjectDetection
import os
from matplotlib import pyplot as plt
import cv2
from imageai.Detection import ObjectDetection
import os

# execution_path = os.getcwd()


def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth


execution_path = os.getcwd()


cap = cv2.VideoCapture(0)
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "yolov3.pt"))
custom = detector.CustomObjects(
    person=True, dog=True,   bicycle=True,   car=True, bus=True,  truck=True)


focal_len = 3.1
known_width = 10
speed_us = 0.01

last_distance = 0


while True:
    print("capturing")
    success, img = cap.read()
#     print(success)
#     cv2.imshow("image",img)
    detections = detector.detectObjectsFromImage(input_image=img,
                                                 output_image_path=os.path.join(
                                                     execution_path, "imagenew.jpg"),
                                                 minimum_percentage_probability=30)
    for eachObject in detections:
        box_pints = eachObject["box_points"]
        width = box_pints[2]-box_pints[0]

        inches = distance_to_camera(known_width, focal_len, width)
        if(inches < 0.06):
            print("please slowdown ")
            # while True:
            #     print("please slowdown ")
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break

        if(eachObject["name"] == "person"):
            speed = (inches - last_distance)/5
            print("person speed :", speed)

            if(speed - speed_us < -0.0001):
                print("alert you may  it object ")
                # while True:
                #     print("please slowdown ")
                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                #         break
            last_distance = inches
        print(eachObject["name"], " : ", eachObject["percentage_probability"],
              " : ", eachObject["box_points"], "width :", width, " distance : ", inches)

        print("--------------------------------")
    cv2.imshow("image", img)
    cv2.waitKey(5)

cap.release()

# video_detector = VideoObjectDetection()
# video_detector.setModelTypeAsYOLOv3()
# video_detector.setModelPath(os.path.join(execution_path, "yolov3.pt"))
# video_detector.loadModel()

# plt.show()
# print("detecting")
# video_results = video_detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "test_3.mp4"), output_file_path=os.path.join(execution_path, "video_second_analysis") ,
#  frames_per_second=5, per_second_function=forSecond,  minimum_percentage_probability=30, return_detected_frame=True, log_progress=True)
