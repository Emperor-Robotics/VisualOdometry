# Camera Calibration


### Camera calibration resource page for nav2
https://navigation.ros.org/tutorials/docs/camera_calibration.html

NOTE: The pdf provided is the one generated from following the above tutorial, maybe we just all use that one?


Run the Following in two separate terminals.


Run camera node, usb_cam seems to work for most USB cameras. If you have a specific camera that has its own package, run that to get image stream.
```
ros2 run usb_cam usb_cam_node_exe --ros-args --params-file camera_params.yaml 
```
NOTE: camera_params should be copied from 
`/opt/ros/humble/share/usb_cam/config/params.yaml` 
and be modified accordingly (camera dimensions, and device path are most common changes)



The below node remaps assume v4l2 node above.
```
ros2 run camera_calibration cameracalibrator --size 7x9 --square 0.02 --ros-args -r image:=/image_raw -p camera:=/camera_info
```

