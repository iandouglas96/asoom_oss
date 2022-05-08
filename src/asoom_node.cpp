#include <ros/ros.h>
#include "asoom/asoom_wrapper.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "asoom");
  ros::NodeHandle nh("~");

  try {
    ASOOMWrapper node(nh);
    node.initialize();
    ros::spin();
  } catch (const std::exception& e) {
    ROS_ERROR("%s: %s", nh.getNamespace().c_str(), e.what());
  }
  return 0;
}
