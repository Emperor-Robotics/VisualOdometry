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

### Our Project Idea (Visual Interial Odometry):
- Track multiple spots in 3d space dynamically
- Get odometry of the camera
- Useful in many applications, since cameras are easily obtainable

## 2. Research code and demonstrate processing
http://www.cs.toronto.edu/~urtasun/courses/CSC2541/03_odometry.pdf
```
Capture new frame Ik
Extract and match features between Ik-1 and Ik
Decompose essential matrix into Rk and tk, and form Tk
Compute relative scale and rescale tk accordingly
concatenate transform by computing Ck=Ck-1 * Tk
Repeat
```
Visual Odometry survey
https://github.com/klintan/vo-survey

## 3. Exploratory Program
### Project Github:
  We're keeping our project code localized in the following repository:
  https://github.com/Emperor-Robotics/VisualOdometry
  
### Team and expected contributions
- Kadyn Martinez
  - Research
  - Writing Custom package code
  - Taking test data
  - Testing on multiple robots
  - Test on different Cameras
- Wataru Oshima
  - Research
  - Writing Custom package code
  - Taking test data
  - Testing on multiple robots
  - Test on different Cameras

### Existing Applications and Resources
Note, these are just a few of the resources found.

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

## Proposal Table
| Task    | Metrics                                                                                               | Demonstrate                                                                                   | Test                                                                                   |
| ------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Optimal | <ul><li>Within 10% of other established odometry sources (Established meaning OEM odometry)</li></ul> | <ul><li>Run Custom VIO package on live robot</li><li>Custom VIO in navigation stack</li></ul> | <ul><li>Compare Custom VIO pkg Vs Lidar or Wheel Odom</li></ul>                        |
| Target  | <ul><li>Accurate to 1 meter</li></ul>                                                                 | <ul><li>Run Custom VO package</li><li>Add Inertia into the loop for VIO</li></ul>             | <ul><li>Compare Custom Odometry Vs empiricaly measured movement</li></ul>              |
| Minumum | <ul><li>Accurate to within 10 meters on test file</li></ul>                                           | <ul><li>Run off the shelf Visual Odometry code</li></ul>                                      | <ul><li>Run OTS Visual Odometry code to get odometry from a common test file</li></ul> |