#pragma once

#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>
#include <Eigen/Dense>
#include "asoom/keyframe.h"
#include "asoom/semantic_color_lut.h"

/*!
 * Wraps a GridMap as the underlying data structure, manages integrated map
 */
class Map {
  public: 
    struct Params {
      float resolution; // In cell/m (consistent with top_down_render)
      float buffer_size_m;
      float req_point_density; // In pts/m^2

      float dist_for_rebuild; // In m
      float ang_for_rebuild; // In rad

      Params(float r, float bsm, float rqd, float dfr, float afr) : 
        resolution(r), buffer_size_m(bsm), req_point_density(rqd), dist_for_rebuild(dfr),
        ang_for_rebuild(afr) {}

      Params() : Params(2, 50, 500, 1, 5*M_PI/180) {}
    };

    Map(const Params& params, const SemanticColorLut& lut);

    /*!
     * Add a new pointcloud to the map.
     *
     * @param cloud Pointcloud to add, should already be in the world frame
     * @param camera_pose Camera pose from which cloud projected.
     * @param stamp Timestamp in nsec of cloud
     */
    void addCloud(const DepthCloudArray& cloud, const Eigen::Isometry3d& camera_pose, 
        long stamp);

    /*!
     * Reset all map layers
     */
    void clear();

    /*!
     * Resize so min and max are in bounds up to buffer
     */
    void resizeToBounds(const Eigen::Vector2d& min, const Eigen::Vector2d& max);

    grid_map_msgs::GridMap exportROSMsg();

    //! This is really just for test purposes
    inline const grid_map::GridMap& getMap() const {
      return map_;
    }

    //! Simple getter for parameters
    inline const Params& getParams() const {
      return params_;
    }

    /*!
     * Export map images
     * @param sem Image of semantic layer (grayscale class indices)
     * @param sem_viz Visualization of semantic layer (BGR color)
     * @return Map center
     */
    Eigen::Vector2f getMapSemImg(cv::Mat& sem, cv::Mat& sem_viz) const;

  private:
    const Params params_;

    SemanticColorLut semantic_color_lut_;

    grid_map::GridMap map_;

    //! Keep track of most recent timestamp for image in map
    long most_recent_stamp_;
};
