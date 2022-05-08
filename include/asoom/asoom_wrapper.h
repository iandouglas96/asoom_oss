#pragma once

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/NavSatFix.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/subscriber.h>

#include "asoom/asoom.h"

/*!
 * Wrap up ASOOM with ROS stuff.
 * Contains pubs and subs as well as parameter management and type conversions
 * to ROS-land.
 */
class ASOOMWrapper {
  public:
    //! Setup variables, init stuff
    ASOOMWrapper(ros::NodeHandle& nh);

    //! Setup ROS pubs/subs
    void initialize();

  private:
    /***********************************************************
     * LOCAL FUNCTIONS
     ***********************************************************/

    //! Helper function to create ASOOM object
    ASOOM createASOOM(ros::NodeHandle& nh);

    //! Called to publish output
    void outputCallback(const ros::TimerEvent& event);

    void publishPoseGraphViz(const ros::Time& time);
    void publishRecentPointCloud(const ros::Time& time);
    void publishRecentPose(const ros::Time& time);
    void publishUTMTransform(const ros::Time& time);
    void publishMap(const ros::Time& time);
    void publishKeyframeImgs();

    //! Callback for VO.  Pass empty image pointer if only tracking pose
    void poseImgCallback(const geometry_msgs::PoseStamped::ConstPtr& pose_msg,
        const sensor_msgs::Image::ConstPtr& img_msg);

    //! Callback for GPS data
    void gpsCallback(const sensor_msgs::NavSatFix::ConstPtr& gps_msg);
    
    //! Callback for Semantic data
    void semCallback(const sensor_msgs::Image::ConstPtr& sem_msg);
    
    //! Convert Eigen position to ROS point
    static geometry_msgs::Point Eigen2ROS(const Eigen::Vector3d& pos);

    //! Convert Eigen pose to ROS pose
    static geometry_msgs::Pose Eigen2ROS(const Eigen::Isometry3d& pose);

    //! Convert ROS Pose to Eigen pose
    static Eigen::Isometry3d ROS2Eigen(const geometry_msgs::PoseStamped& pose_msg);

    /*!
     * Convert ROS GPS msg to Eigen
     * This also includes moving from WGS84 to UTM
     */
    static Eigen::Vector3d ROS2Eigen(const sensor_msgs::NavSatFix& gps_msg);

    /***********************************************************
     * LOCAL VARIABLES
     ***********************************************************/

    ros::NodeHandle nh_;

    ASOOM asoom_;

    //! Period with which to publish stuff to ROS
    float ros_pub_period_ms_;

    //! If true, sub to synchronized images and poses
    bool require_imgs_;

    //! If true, sub to semantic image topic
    bool use_semantics_;

    //! If true, use stamp from GPS.  If false, restamp with ROS time
    bool use_gps_stamp_;

    Eigen::Vector2d utm_origin_;

    //! ROS Pubs and subs
    std::unique_ptr<message_filters::Subscriber<geometry_msgs::PoseStamped>> pose_sync_sub_;
    std::unique_ptr<message_filters::Subscriber<sensor_msgs::Image>> img_sync_sub_;
    std::unique_ptr<message_filters::TimeSynchronizer<geometry_msgs::PoseStamped, 
      sensor_msgs::Image>> pose_img_sync_sub_;
    ros::Subscriber gps_sub_, pose_sub_, sem_sub_;
    ros::Publisher trajectory_viz_pub_, recent_cloud_pub_, recent_key_pose_pub_, map_pub_, 
      keyframe_img_pub_, map_sem_img_pub_, map_sem_img_viz_pub_, map_sem_img_center_pub_;

    //! Timer to loop and publish visualizations and the map
    ros::Timer output_timer_;

    //! Vector of points specifying camera frustum viz
    const std::vector<Eigen::Vector3d> frustum_pts_;
    static std::vector<Eigen::Vector3d> initFrustumPts(float scale=0.5);
};
