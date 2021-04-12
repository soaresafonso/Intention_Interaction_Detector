# Intention_Interaction_Detector
### Code and datasets for a intention of interaction detector

This repository contains the ROS packages needed to run an intent of interaction detector, as well as to detect the gestures that a user performs if he intends to interact. The physical data of the user as well as his movements are collected through a camera and then it is presented in real time, on a command line. If the user wants to interact or not, the program is able to recognize the gestures that the user does (handshake with any hand, a wave with any hand, a bow and a prayer position with the hands).

The packages are:

The openface2_ros package that can be seen in detail in this link https://github.com/ditoec/openface2_ros

The openpose_ros package that can be seen in detail in this link https://github.com/firephinx/openpose_ros

The vision_opencv package needed to run the above packages, as mentioned in the two links.

The data_receiver package that contains a folder of datasets used to build the different models and classifiers used in this application, the .sav files that represent these models and classifiers and finally the .py files that were used to obtain the data from the different datasets and train the classifiers.
The file data_receiver.py is the final application that receives the data subscribed from the openface2_ros and the openpose_ros and then processes this data to know if the user wants to interact and what gestures is the user doing, using the above classifiers to decide.

### To run the the Intention_Interaction_Detector using a computer camera as the input signal follow the next steps

On a server run (the packages must be run in their directory):
```
roscore
roslaunch openface2_ros openface2_ros.launch 
roslaunch openpose_ros openpose_ros.launch
```

On a computer connected to the same network as the server above (the packages must be run in their directory):

```
rosrun usb_cam usb_cam_node 
rosrun data_receiver data_receiver.py
```
