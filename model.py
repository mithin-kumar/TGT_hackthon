from imageai.Detection import VideoObjectDetection
import os
from matplotlib import pyplot as plt
import cv2
from imageai.Detection import ObjectDetection
import os
import winsound


def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth


execution_path = os.getcwd()


cap = cv2.VideoCapture("sample1.mp4")

# creating the already trained model using imageai library
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "yolov3.pt"))

# chooseing objects  to detect
custom = detector.CustomObjects(
    car=True)

# mentioning our camera focal length
focal_len = 5

# to calculate the distance of object from camera we need to know the actual width of image here we have taken an approximation
known_width = 3

# to track the relative velocity initialise
last_distance = 0
safe_distance = 0.06
driver_speed = 0.04
cv2.namedWindow("image", cv2.WINDOW_NORMAL)

while True:
    # capturing webcam
    print("c")
    success, img = cap.read()

    # detecting objects in the image
    if success:
        print("sucess")
    else:
        break
    detections = detector.detectObjectsFromImage(input_image=img,
                                                 output_image_path=os.path.join(
                                                     execution_path, "imagenew.jpg"),
                                                 minimum_percentage_probability=30)
    for eachObject in detections:
        box_pints = eachObject["box_points"]
        width = box_pints[2]-box_pints[0]

        inches = distance_to_camera(known_width, focal_len, width)
        if(inches < safe_distance):  # if the object is mush nearer  alert the driver about slowdown
            print("please slowdown ")

        # here we are taking only person as object for prototyping but we on ground  we have take it as car, bus etc.
        if(eachObject["name"] == "car"):
            # speed of the object is calculated as difference in distance and last calculated distance
            speed = ((inches - last_distance)/5)*100
            print("person speed :", speed)
            cv2.rectangle(
                img, (box_pints[0], box_pints[1]), (box_pints[2], box_pints[3]), (0, 255, 0), 2)
            if(speed < 0):
                cv2.putText(img, "relative speed :"+str("negative")+"Kmph", (box_pints[0], box_pints[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                winsound.Beep(2000, 1500)
                cv2.putText(img, "alert!your speed morethan object speed! pls slowdown ", (0, 33),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                print("alert! object is slowing down")
            elif(speed == 0):
                cv2.putText(img, "relative speed :"+str("zero")+"Kmph", (box_pints[0], box_pints[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                cv2.putText(img, "relative speed :"+str("positive")+"Kmph", (box_pints[0], box_pints[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            if(speed <= -driver_speed):  # case when object is slowing down or in rest position
                cv2.putText(img, "alert! object is rest", (0, 33),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                print("alert! object is rest or  it is moving opposite to you ")
            # elif(speed < 0):
                cv2.putText(img, "alert!your speed morethan object speed! pls slowdown ", (0, 33),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                print("alert! object is slowing down")

            last_distance = inches
        print(eachObject["name"], " : ", eachObject["percentage_probability"],
              " : ", eachObject["box_points"], "width :", width, " distance : ", inches)

        print("--------------------------------")
    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # cv2.waitKey(0.5)

cap.release()
