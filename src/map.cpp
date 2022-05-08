#include "asoom/map.h"

Map::Map(const Params& params, const SemanticColorLut& lut) : 
  params_(params), semantic_color_lut_(lut)
{
  map_ = grid_map::GridMap({
      "elevation", 
      "color", 
      "semantics",
      "semantics_viz",
      "view_angle",
      "num_points"});
  map_.setBasicLayers({"elevation", "color"});
  // Reset layers to the appropriate values
  map_.setFrameId("map");
  // Init with buffer on all sides of (0, 0)
  map_.setGeometry(grid_map::Length(params_.buffer_size_m, params_.buffer_size_m)*2, 
      1/params_.resolution, grid_map::Position(0, 0));
  clear();
}

void Map::addCloud(const DepthCloudArray& cloud, const Eigen::Isometry3d& camera_pose, 
    long stamp) 
{
  if (stamp > most_recent_stamp_) {
    most_recent_stamp_ = stamp;
  }
  grid_map::Position corner_pos;
  map_.getPosition(grid_map::Index(0, 0), corner_pos);
  // This is the array with the indices each point falls into
  Eigen::Array2Xi inds = (((-cloud).topRows<2>().colwise() + 
      corner_pos.array().cast<float>()) / map_.getResolution()).round().cast<int>();
  Eigen::Isometry3f camera_pose_inv = camera_pose.inverse().cast<float>();

  // Get references to map layers we need, speeds up access so we only do key lookup once
  grid_map::Matrix &elevation_layer = map_["elevation"];
  grid_map::Matrix &color_layer = map_["color"];
  grid_map::Matrix &semantics_layer = map_["semantics"];
  grid_map::Matrix &semantics_viz_layer = map_["semantics_viz"];
  grid_map::Matrix &view_angle_layer = map_["view_angle"];
  grid_map::Matrix &num_points_layer = map_["num_points"];

  // Iterate through points
  grid_map::Index ind;
  Eigen::Vector3f pt_camera_frame;
  double view_angle;
  uint8_t sem_ind;
  uint32_t sem_color_packed;
  for (size_t col=0; col<inds.cols(); col++) {
    ind = inds.col(col);
    if ((ind < 0).any() || (ind >= map_.getSize()).any()) {
      // Point is outside map bounds
      continue;
    }
    // Cumulative mean
    if (std::isnan(elevation_layer(ind[0], ind[1]))) {
      elevation_layer(ind[0], ind[1]) = cloud(2, col);
    } else {
      elevation_layer(ind[0], ind[1]) += (cloud(2, col) - elevation_layer(ind[0], ind[1])) / 
        (num_points_layer(ind[0], ind[1]) + 1);
    }
    num_points_layer(ind[0], ind[1])++;

    // Use color from closest to image center
    pt_camera_frame = camera_pose_inv * cloud.col(col).head<3>();
    view_angle = std::abs(std::atan2(pt_camera_frame.head<2>().norm(), pt_camera_frame[2]));
    if (view_angle < view_angle_layer(ind[0], ind[1]) && 
        std::abs(elevation_layer(ind[0], ind[1]) - cloud(2, col)) < 1 &&
        num_points_layer(ind[0], ind[1]) > params_.req_point_density / std::pow(params_.resolution, 2)) {
      // Update color if depth is consistent with map and if closer to image center
      view_angle_layer(ind[0], ind[1]) = view_angle;
      color_layer(ind[0], ind[1]) = cloud(3, col);

      sem_ind = cloud(4, col);
      // Don't overwrite with unknown
      if (sem_ind != 255) {
        semantics_layer(ind[0], ind[1]) = sem_ind;
        sem_color_packed = semantic_color_lut_.ind2Color(sem_ind);
        semantics_viz_layer(ind[0], ind[1]) = *reinterpret_cast<float*>(&sem_color_packed);
      }
    }
  }
}

void Map::clear() {
  most_recent_stamp_ = 0;
  map_.setConstant("elevation", NAN);
  map_.setConstant("color", NAN);
  map_.setConstant("semantics", NAN);
  map_.setConstant("semantics_viz", NAN);
  map_.setConstant("view_angle", M_PI/2);
  map_.setConstant("num_points", 0);
}

void Map::resizeToBounds(const Eigen::Vector2d& min, const Eigen::Vector2d& max) {
  grid_map::Position center_pos = map_.getPosition();
  grid_map::Length size = map_.getLength();

  grid_map::Length diff = center_pos.array() - size/2 - min.array() + params_.buffer_size_m;
  if ((diff > 0).any()) {
    size += diff.cwiseMax(0);
    map_.grow(size, grid_map::GridMap::SW);
  }

  diff = -center_pos.array() - size/2 + max.array() + params_.buffer_size_m;
  if ((diff > 0).any()) {
    size += diff.cwiseMax(0);
    map_.grow(size, grid_map::GridMap::NE);
  }
}

grid_map_msgs::GridMap Map::exportROSMsg() {
  grid_map_msgs::GridMap msg;
  map_.setTimestamp(most_recent_stamp_);
  grid_map::GridMapRosConverter::toMessage(map_, msg);
  return msg;
}

Eigen::Vector2f Map::getMapSemImg(cv::Mat &sem, cv::Mat &sem_viz) const {
  sem = cv::Mat(map_.getSize()(0), map_.getSize()(1), CV_8UC1, cv::Scalar(255));
  sem_viz = cv::Mat(map_.getSize()(0), map_.getSize()(1), CV_8UC3, cv::Scalar(255,255,255));
  // The default toImage does not properly handle color, so we loop manually
  // This code is modified from that
  const grid_map::Matrix& sem_layer_viz = map_["semantics_viz"];
  const grid_map::Matrix& sem_layer = map_["semantics"];

  std::array<uint8_t, 3> color_vec;
  for (grid_map::GridMapIterator it(map_); !it.isPastEnd(); ++it) {
    const grid_map::Index index(*it);
    const float& value = sem_layer_viz(index(0), index(1));

    if (std::isfinite(value)) {
      const grid_map::Index image_index(it.getUnwrappedIndex());
      sem.at<uint8_t>(image_index(0), image_index(1)) = sem_layer(index(0), index(1));

      color_vec = SemanticColorLut::unpackColor(*reinterpret_cast<const uint32_t*>(&value));
      sem_viz.at<cv::Vec<uint8_t, 3>>(image_index(0), image_index(1)) = 
        cv::Vec<uint8_t, 3>(color_vec[0], color_vec[1], color_vec[2]);
    }
  }

  // grid_map::Position is Eigen::Vector2d
  return map_.getPosition().cast<float>();
}
