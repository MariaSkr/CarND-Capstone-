
### Udacity Team

- Maria Skryagina (maria.skriaghina@gmail.com)
- Amit Marathe (amit.marathe@gmail.com)
- Leo Mao (leo.cl.mao@gmail.com)
- Junxiao Zhu (junxiaozhu@gmail.com)
- Linsen Chong (linsen.chong@gmail.com)

This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

Please use **one** of the two installation options, either native **or** docker installation.


### Introduction

In this project, our team designed an autonomous vehicle system. We tested on a simulator. The project consists of several parts: 1) Traffic light detection, 2) waypoint generation and 3) manuever and control.

### Architecture Diagram

![image](imgs/final-project-ros-graph-v2.png)
 

### Traffic Light Detection
In this part, we detect the closest traffic light ahead and classifying the color of this light from images.

Getting the color of traffic light from images can include two tasks: detecting where traffic lights are and then determining the color of traffic light. In this project, we use [Tensorflow Object Detection API]( https://github.com/tensorflow/models/tree/master/research/object_detection). We followed the API guidances to install the API, prepare training data  train our model. Here, we briefly discribe the key steps.

#### Training data preparation
Tensorflow API requires images and the corresponding labels (to denote where the objects are and the class label) as raw data. In order to train the model with API, we need to convert the data into TFRecord format. The training images we use are the Udacity traffic light data (simulator and real data) from [here](https://drive.google.com/file/d/0B-Eiyn-CUQtxdUZWMkFfQzdObUE/view). The data include images (simulator and real data) and their labels. We would like to thank [Anthony Sarkis](https://medium.com/@anthony_sarkis) for making this data set available). From this data, we create the TFRecord files.



#### Train the model
We then follow the TensorFlow API [guideline](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md) to train the model. The most importance thing is to choose a model and setup the [objective detection pipeline configuration](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md) for the model. Luckily, TensorFlow API provides a rich set of object detection models, see [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). We choose model ssd_inception_v2_coco model since it runs fast. This gives us to use a the onfiguration file template for ssd_inception model, see [ssd_inception_v2.config](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_inception_v2_coco.config).

Our configuration files are [ssd_inception_v2_sim.config](https://github.com/amitmarathe/majl/blob/master/image_classification_model/config/ssd_inception_v2_sim.config), for simulator data; and [ssd_inception_v2_real.config](https://github.com/amitmarathe/majl/blob/master/image_classification_model/config/ssd_inception_v2_real.config) for real data.

In addition, we need a provide a [label_map](https://github.com/amitmarathe/majl/blob/master/image_classification_model/label_map.pbtxt) file to associate class ids to colors. This file is needed in configuration.

The key points to set up include
num_classes: 4
max_detections_per_class:50
max_total_detections:50. 
num_steps: 5000.
fine_tune_checkpoint: "ssd_inception_v2_coco_2018_01_28/model.ckpt"

#### Generate protobuf
Once training is finished, the model and metamodel as saved as .ckpt files. The next step is freeze the model and save them as protobuf file. We use TensorFlow API's [exporter](https://github.com/tensorflow/models/blob/master/research/object_detection/export_inference_graph.py) to generate protobuf file from the .ckpt files. 
The protobuf files can be found in folder [protobuf](https://github.com/amitmarathe/majl/tree/master/image_classification_model/protobuf). 

For the trained model on simulator data, we freeze the trained model at checkpoint 2841 (model.ckpt-2841). For real data, we use model at checkpoint1096.

#### Note
Udacity requires tensorflow version 1.3.0. to run the traffic light detector. However, the tensorflow API requires Tensorflow (>=1.9.0) to train the model. Therefore, our workflow involves 1) using tensorflow 1.9.0 to generate model checkpoints (i.e.,model.ckpt files), and 2) using the exporter of tensorflow 1.3.0. to create protobuf files. 

### Waypoint Loader Node

- From Autoware

```xml
<?xml version="1.0"?>
<launch>
    <node pkg="waypoint_loader" type="waypoint_loader.py" name="waypoint_loader">
        <param name="path" value="$(find styx)../../../data/wp_yaw_const.csv" />
        <param name="velocity" value="40" />
    </node>
</launch>
```

**Parameters:**
- load a csv file corresponding to a path to follow + a configurable maximum velocity

**Publisher:**
- /base_waypoints: set of waypoints and associated target velocity corresonding to a planned path (note that we stop at the end of the planned path)


### Waypoint Updater Node


**Subscribers:**
- /current_pose: ego (x, y) position. Populated by Autoware locatization module (GPS+LIDAR based).
- /base_waypoints: path planned as a discrete set of (x, y) positions
- /traffic_waypoint: -1 or a number > 0 corresponding to a waypoint where we should stop (RED light match)


If 1st detection of a RED Traffic Light: compute a deceleration path ( in SQRT; Not a linear decrease: the faster the decrease the closer to the stop position )
```python
    def is_stop_close(self, base_waypoint_idx):
        """ Checks whether it is time to start slowing down
        """
        stop_is_close = False
        if self.stop_wp > 0:
            # stop is ahead
            d_stop = self.distance(
            self.base_waypoints, base_waypoint_idx, self.stop_wp) - self.stop_m
            current_wp = self.base_waypoints[base_waypoint_idx]
            stop_is_close = d_stop < current_wp.twist.twist.linear.x ** SLOWDOWN
        return stop_is_close

    def brake(self, i):
        """ Decreases waypoint velocity
        """
        wp = self.base_waypoints[i]
        wp_speed = wp.twist.twist.linear.x
        d_stop = self.distance(self.base_waypoints, i, self.stop_wp) - self.stop_m
        speed = 0.
        if d_stop > 0:
           speed = d_stop * (wp_speed ** (1. - SLOWDOWN))
        if speed < 1:
            speed = 0.
        return speed
```
- If end of red light: we restore the original planned path and its associated velocities.


**Publisher:**
- /final_waypoints: a set of waypoints and their associated velocity (based on object/traffic light detection information) that we should follow

**Loop: 10 HZ**
- every 100 ms: 
     - Find the closest waypoints (with base_waypoint)
     - Extract LOOKAHEAD_WPS waypoints (typically 40 points). Per waypoint velocity has been already updated by /traffic_waypoint callback
     - Publish /final_waypoints

![image](imgs/waypoint-updater.png )

### Waypoint Follower Node

- Pure Pursuit from Autoware
  
For more details cf:   
https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf  
https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf  


**Parameters:**
- /linear_interpolate_mode:  

**Subscribers:**
- /final_waypoints: 
- /current_pose:
- /current_velocity:



### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
