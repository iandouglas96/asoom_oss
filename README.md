# ASOOM: Aerial Semantic Online Ortho-Mapping

~[Demo Image](demo_img.png)

## Dependencies
* OpenCV (Tested with 3.4)
* Eigen (Tested with 3.3)
* grid_map (Forked version) https://github.com/iandouglas96/grid_map
* GTSAM (Tested with 4.0.3)
* ROS1 (Tested with Melodic)

## Test Data
Some data can be found [here](https://drive.google.com/file/d/1XzIDCkFKaATIreknPa39f1PhiEZI8kid/view?usp=sharing).
To run:
```
roscore
rosparam set /use_sim_time true
roslaunch asoom asoom.launch
rosbag play --clock asoom_demo.bag
```
Then open rviz and load `launch/viz.rviz`.
If you would like to map semantics as well, you will need to also run [ErfNet](https://github.com/iandouglas96/erfnet_pytorch_ros), as well as set the `use_semantics` parameter.
We use ORBSLAM3 for odometry with a wrapper found [here](https://github.com/iandouglas96/orbslam3_ros/).
However, odometry data is included in the provided bag file.
The settings used in our experiments are in `asoom_iros_params.launch`.
