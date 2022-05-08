#pragma once

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <memory>
#include <map>
#include <iostream>

namespace Eigen {
  using Array5Xd = Array<double, 5, Dynamic>;
  using Array5Xf = Array<float, 5, Dynamic>;
  using Array6Xd = Array<double, 6, Dynamic>;
}

using DepthCloudArray = Eigen::Array5Xf;

/*!
 * Manage a single keyframe in the pose graph
 */
class Keyframe {
  public:
    Keyframe(long stamp, cv::Mat img, const Eigen::Isometry3d& pose) 
      : stamp_(stamp), img_(img), pose_(pose), odom_pose_(pose) {}

    // Setters and getters
    long getStamp() const {
      return stamp_;
    }

    float getScale() const {
      return scale_;
    }

    Eigen::Isometry3d getPose() const {
      return pose_;
    }

    Eigen::Isometry3d getRectPose() const {
      return pose_ * rect_dpose_;
    }

    Eigen::Isometry3d getOdomPose() const {
      return odom_pose_;
    }

    const cv::Mat& getImg() const {
      return img_;
    }

    bool hasDepth() const {
      return have_depth_;
    }

    bool hasSem() const {
      return have_sem_;
    }

    const cv::Mat& getSem() const {
      return sem_img_;
    }

    bool hasRepublished() const {
      return republished_;
    }

    void republish() {
      republished_ = true;
    }

    void setOptimized() {
      is_optimized_ = true;
    }

    bool isOptimized() {
      return is_optimized_;
    }

    void setScale(float s) {
      scale_ = s;
    }

    void setPose(const Eigen::Isometry3d& p) {
      pose_ = p;
    }

    void setDepth(const Eigen::Isometry3d& dp, const cv::Mat& rect_img,
        const std::shared_ptr<Eigen::Array3Xd>& depth) {
      rect_dpose_ = dp;
      rect_img_ = rect_img;
      depth_ = depth;
      if (depth_) {
        have_depth_ = true;
      }
      on_disk_ = false;
    }

    void setDepth(const Keyframe& key) {
      setDepth(key.rect_dpose_, key.rect_img_, key.depth_);
    }

    void setSem(const cv::Mat& sem) {
      if (sem.type() == CV_8UC1) {
        sem_img_ = sem;
        have_sem_ = true;
        on_disk_ = false;
      }
    }

    DepthCloudArray getDepthCloud() const;

    bool needsMapUpdate(float delta_d = 1, float delta_theta = 5*M_PI/180) const;

    bool inMap() const {
      return !map_pose_.matrix().isIdentity(1e-5);
    }

    void updateMapPose() {
      map_pose_ = pose_;
    }

    //! Saves data to disk, then wipes data
    void saveToDisk();

    //! Returns true if load successful
    bool loadFromDisk();

  private:
    /***********************************************************
     * LOCAL VARIABLES
     ***********************************************************/

    //! Timestamp in nsec from epoch
    long stamp_;

    //! Current global pose estimation
    Eigen::Isometry3d pose_;
    // Pose estimation before optimization.  Keep track for stereo purposes, since
    // when PGO permutes the image alignment for stereo is not as good
    Eigen::Isometry3d odom_pose_;
    float scale_;

    /*! 
     * The pose used by the mapper.  If pose_ changes from this significantly, we
     * want to trigger a map rebuild
     */
    Eigen::Isometry3d map_pose_{Eigen::Isometry3d::Identity()};

    //! Corrective rotation for rectification
    Eigen::Isometry3d rect_dpose_{Eigen::Isometry3d::Identity()};

    // These cv::Mats and shared_ptr allow Keyframes to be copied quickly, at the cost 
    // of not being a deep copy.  However, this is ok, since these are never modified
    // in keyframes_ directly once they are set.

    //! Image associated with keyframe
    cv::Mat img_;
    cv::Mat rect_img_;

    //! Semantic image, later can be overwritten with rect image
    cv::Mat sem_img_;

    // These are separate flags so when we cache to disk, still true or false
    bool have_depth_ = false;
    bool have_sem_ = false;
    //! Depth cloud associated with keyframe for the same image, stored row-major
    std::shared_ptr<Eigen::Array3Xd> depth_;

    //! True if the keyframe image has been republished
    bool republished_ = false;

    //! True if data on disk is up to date
    bool on_disk_ = false;

    //! True if data currently in memory
    bool in_mem_ = true;

    //! True if the pose has been optimized by PGO
    bool is_optimized_ = false;

    /***********************************************************
     * LOCAL STATIC FUNCTIONS
     ***********************************************************/

    static void saveDataBinary(const cv::Mat& img, std::ofstream& outfile);
    static void saveDataBinary(const std::shared_ptr<Eigen::Array3Xd>& arr, 
        std::ofstream& outfile);

    static void readDataBinary(std::ifstream& infile, cv::Mat& img);
    static void readDataBinary(std::ifstream& infile, 
        std::shared_ptr<Eigen::Array3Xd>& arr);
};

// Using pointers here should make sort faster
using Keyframes = std::map<long, std::unique_ptr<Keyframe>>;
