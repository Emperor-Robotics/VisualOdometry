### Kadyn Martinez
### Wataru Oshima
### CSCI 585
### 10/18/22
Project: Proposal
===

## 1. Suggestions for projects
<!-- three projects from list three simple bullet points -->
- Robot club TB3 [Track and push ball to goal]
  - Tracks a ball using machine vision
  - Robot is tasked to move ball into the goal
  - Pretty cool, reminiscent of Robo-Soccer
- Turtlebot 3 Maze Explorer
  - Seems to use a bug type algorithm
  - Smooth navigation of really confined rooms
  - Code explaination though
- Multi Robot Exploration
  - Really neat for swarm robotics
  - Multiple robots mapping an environment
  - multi robot collaboration (Broadcast and request help)
<!-- three bullet points from your own project idea -->

Our Project Idea:
- 

## 2. Research code and demonstrate processing

## 3. Exploratory Program
### Team and expected contributions
- Kadyn Martinez
  - Contributions
- Wataru Oshima
  - Contributions

### Existing Applications and Resources

### Possible now using starter code or examples using class resources

### Describe Goals to extend what you are leveraging

### Describe what you will deliver and demonstrate

## 4. Paragraph Proposal
Visual Odometry
<!-- 90-100% -->
### Optimal
Implement it and combine indoor naviation.

<!-- 80-90% -->
### Target
Implement it ourselves or extend package

<!-- 70-80% -->
### Minimum
Visual Odometry Demo from package



# Resources
Visual Odometry Survey
https://github.com/klintan/vo-survey

Learning VI-ORB repo
https://github.com/jingpang/LearnVIORB

Open VINS # Visual Inertial Navigation
https://github.com/rpng/open_vins


Simple Visual Monocular Odometry
https://github.com/avisingh599/mono-vo

Ros Simple visual Monocular Odometry (Based on Previous)
https://github.com/atomoclast/ros_mono_vo


# Usage - VO Standalone

Run the following command to test out optical flow and trajectory estimation.
The calibration file is the one generated from ros image pipeline. For more information see `CALIB/README.md`
Depending on whats entered for the second argument,
-  If the value is an integer it will be treated as a device index and open that camera for live optical flow.
-  If the value is a filepath, it will open the file and run the optical flow on the file.

```
python prototype/optiflow_w_essential.py [CALIBRATION.YAML] [DEVICE INDEX or FILEPATH]
```


# Usage - ROS PKG (UNDER CONSTRUCTION)

Run the following or your camera's native ros package to get video stream.
```
ros2 run usb_cam usb_cam_node_exe --ros-args --params-file camera_params.yaml 
```

NOTE: camera_params should be copied from 
`/opt/ros/humble/share/usb_cam/config/params.yaml` 
and be modified accordingly (camera dimensions, and device path are most common changes)

