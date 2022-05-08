#pragma once

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include "asoom/keyframe.h"

using gtsam::symbol_shorthand::P;
using gtsam::symbol_shorthand::S;

namespace Eigen {
  using Vector6d = Matrix<double, 6, 1>;
}

/*!
 * Wrapper for GTSAM Pose Graph
 */
class PoseGraph {
  public:
    struct Params {
      //! Vector of diagonal sigmas for between factors from VO
      Eigen::Vector6d between_sigmas;

      //! Vector of diagonal sigmas for GPS factors
      Eigen::Vector3d gps_sigmas;

      /*!
       * Vector of diagonal sigmas for GPS multiplied with time
       * Related to quality of interpolation
       */
      Eigen::Vector3d gps_sigma_per_sec;

      //! Whether or not to fix scale (e.g. VIO or stereo) or allow GTSAM to optimize (mono VO)
      bool fix_scale;

      //! Number of frames of GPS and VO needed to initialize
      int num_frames_init;

      //! Number of new frames before optimizing
      int num_frames_opt;

      //! Full constructor
      Params(const Eigen::Vector6d& bs, const Eigen::Vector3d& gs, 
          const Eigen::Vector3d& gsps = Eigen::Vector3d::Zero(), bool fs = false, int nfi = 5,
          int nfo = 0)
        : between_sigmas(bs), gps_sigmas(gs), gps_sigma_per_sec(gsps), fix_scale(fs),
          num_frames_init(nfi), num_frames_opt(nfo) {}

      //! The "Everything is independent and isotropic" constructor
      Params(double bs_p, double bs_r, double gs, 
          double gsps = 0, bool fs = false, int nfi = 5, int nfo = 0) 
        : between_sigmas((Eigen::Vector6d() << bs_r, bs_r, bs_r, bs_p, bs_p, bs_p).finished()),
          gps_sigmas(Eigen::Vector3d::Constant(gs)), 
          gps_sigma_per_sec(Eigen::Vector3d::Constant(gsps)), fix_scale(fs), 
          num_frames_init(nfi), num_frames_opt(nfo) {}
    };

    /*!
     * Constructor for PoseGraph
     *
     * @param params Parameters for the pose graph
     */
    PoseGraph(const Params& params, 
        const Eigen::Isometry3d& initial_pose = Eigen::Isometry3d::Identity());

    /*!
     * Create new image frame in the pose graph
     *
     * @param stamp Timestamp in nsec
     * @param pose Pose of image as computed by VO up to scale
     * @param sigmas Sigmas for diagonal covariance
     * @return ID number of the new frame in the graph
     */
    size_t addFrame(long stamp, const Eigen::Isometry3d& pose, 
        const Eigen::Vector6d& sigmas);

    //! Overloaded version with default sigmas
    size_t addFrame(long stamp, const Eigen::Isometry3d& pose);
    
    //! Overloaded version that takes a Keyframe 
    size_t addFrame(const Keyframe& frame);

    /*!
     * Add GPS factor to graph.  Automatically linearly interpolates
     *
     * @param stamp Timestamp in nsec
     * @param gps_pose GPS pos in UTM coordinates
     */
    void addGPS(long stamp, const Eigen::Vector3d& utm_pos);

    //! Run gtsam optimization
    void update();

    /*!
     * Get pose of particular node in graph
     *
     * @param stamp Timestamp in nsec to get pose of
     * @return Pose of node.  If no node at timestamp, return nothing
     */
    std::optional<Eigen::Isometry3d> getPoseAtTime(long stamp) const;

    /*!
     * Get pose of particular node in graph
     *
     * @param ind Index of node to get pose of
     * @return Pose of node.  If no node at index, return nothing
     */
    Eigen::Isometry3d getPoseAtIndex(size_t ind) const;

    /*!
     * @return The overall scale of the graph
     */
    double getScale() const;

    /*!
     * @return Number of nodes in graph
     */
    size_t size() const;

    /*!
     * @return Current error in graph
     */
    double getError() const;

    /*!
     * @return True if there are enough gps factors to converge the scale
     */
    bool isInitialized() const;
  private:
    /***********************************************************
     * LOCAL FUNCTIONS
     ***********************************************************/

    /*!
     * Add GPS Factors off of the buffer
     */
    void processGPSBuffer();

    /*!
     * Internal method to handle adding factor, including factor counts
     *
     * @param key Key of Pose node to attach GPS factor to
     * @param utm_pos GPS postition in UTM coord
     * @param sigma Position uncertainty
     */
    void addGPSFactor(const gtsam::Key& key, const Eigen::Vector3d& utm_pos, 
        const Eigen::Vector3d& sigma);

    /*!
     * Convert GTSAM pose to Eigen
     */
    inline static gtsam::Pose3 Eigen2GTSAM(const Eigen::Isometry3d& eigen_pose) {
      return gtsam::Pose3(eigen_pose.matrix());
    }

    /*!
     * Convert Eigen pose to GTSAM
     */
    inline static Eigen::Isometry3d GTSAM2Eigen(const gtsam::Pose3& gtsam_pose) {
      return Eigen::Isometry3d(gtsam_pose.matrix());
    }

    /***********************************************************
     * LOCAL CONSTANTS
     ***********************************************************/

    const Params params_;

    const Eigen::Isometry3d initial_pose_;
    
    /***********************************************************
     * LOCAL VARIABLES
     ***********************************************************/

    //! GTSAM factor graph
    gtsam::NonlinearFactorGraph graph_;

    //! Keep track of the current best estimates for nodes
    gtsam::Values current_opt_;

    struct OriginalPose {
      OriginalPose(const Eigen::Isometry3d& p, const gtsam::Key &k) : pose(p), key(k) {}
      Eigen::Isometry3d pose;
      gtsam::Key key;
    };

    //! Map to keep track of timestamp to original poses
    std::map<long, std::shared_ptr<OriginalPose>> pose_history_;

    //! Map to keep track of timestamp to GPS locations
    std::map<long, std::shared_ptr<Eigen::Vector3d>> gps_buffer_;

    //! Number of frames in graph
    size_t size_;

    //! Number of frames in graph at time of last opt
    size_t last_opt_size_;

    //! Number of GPS factors in graph
    size_t gps_factor_count_;
    
    //! Index of initial origin factor before GPS
    int initial_pose_factor_id_;

    //! Index of initial scale factor before GPS
    int initial_scale_factor_id_;
};

