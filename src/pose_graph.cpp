#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include "asoom/pose_graph.h"
#include "asoom/between_pose_scale_factor.h"

PoseGraph::PoseGraph(const Params& params, const Eigen::Isometry3d& initial_pose)
      : size_(0), last_opt_size_(0), initial_pose_factor_id_(-1), gps_factor_count_(0), 
        initial_pose_(initial_pose), params_(params), graph_()
{
  current_opt_.insert(S(0), 1.0);

  // Create prior on scale initially because otherwise unconstrained
  initial_scale_factor_id_ = graph_.size();
  graph_.emplace_shared<gtsam::PriorFactor<double>>(S(0), 1.0,
      gtsam::noiseModel::Constrained::All(1));
}

size_t PoseGraph::addFrame(long stamp, const Eigen::Isometry3d& pose, 
    const Eigen::Vector6d& sigmas) 
{
  if (size_ > 0) {
    // Get different from most recent
    auto most_recent_pose = pose_history_.rbegin()->second;
    Eigen::Isometry3d diff = most_recent_pose->pose.inverse() * pose;
    graph_.emplace_shared<gtsam::BetweenPoseScaleFactor>(most_recent_pose->key, 
        P(size_), S(0), Eigen2GTSAM(diff), gtsam::noiseModel::Diagonal::Sigmas(sigmas));

    // Use current optimization estimates to improve initial guess
    diff.translation() *= getScale();
    auto most_recent_pose_opt = GTSAM2Eigen(current_opt_.at<gtsam::Pose3>(most_recent_pose->key));
    current_opt_.insert(P(size_), Eigen2GTSAM(most_recent_pose_opt * diff));
  } else {
    // Create prior on first pose to remove free degree of freedom until GPS installed
    // Start downward-facing
    initial_pose_factor_id_ = graph_.size();
    graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(P(size_), 
        Eigen2GTSAM(initial_pose_),
        gtsam::noiseModel::Constrained::All(6));
    current_opt_.insert(P(size_), Eigen2GTSAM(Eigen::Isometry3d::Identity()));
  }
  pose_history_.emplace(stamp, std::make_shared<OriginalPose>(pose, P(size_)));
  size_++;
  processGPSBuffer();
  return size_ - 1;
}

size_t PoseGraph::addFrame(long stamp, const Eigen::Isometry3d& pose) {
  return addFrame(stamp, pose, params_.between_sigmas);
}

size_t PoseGraph::addFrame(const Keyframe& frame) {
  return addFrame(frame.getStamp(), frame.getPose(), params_.between_sigmas);
}

void PoseGraph::addGPS(long stamp, const Eigen::Vector3d& utm_pos) {
  // We buffer GPS measurements because we might get several GPS messages
  // and not get the frame to go between them for a while
  gps_buffer_.emplace(stamp, std::make_shared<Eigen::Vector3d>(utm_pos));
  processGPSBuffer();
}

void PoseGraph::processGPSBuffer() {
  for (auto gps_it = gps_buffer_.begin(); gps_it != gps_buffer_.end();) {
    // First pose after or at same time as GPS
    auto target_frame_it = pose_history_.lower_bound(gps_it->first);

    if (pose_history_.size() > 0) {
      if (std::abs(gps_it->first - pose_history_.rbegin()->first) > 100e9) {
        std::cout << "\033[31m" << "[WARNING] GPS timestamp " << 
          std::abs(gps_it->first - pose_history_.rbegin()->first)/1e9 << 
          " sec off most recent pose" << "\033[0m" << std::endl;
      }
    }

    // All poses are older than GPS
    if (target_frame_it == pose_history_.end()) return;

    if (target_frame_it->first == gps_it->first) {
      // GPS stamp aligns perfectly, great!
      addGPSFactor(target_frame_it->second->key, *(gps_it)->second, params_.gps_sigmas);
    } else {
      // GPS from before pose, next GPS from after
      auto gps_next_it = std::next(gps_it);
      if (gps_next_it != gps_buffer_.end()) {
        double after_t_diff = gps_next_it->first - target_frame_it->first;
        double before_t_diff = target_frame_it->first - gps_it->first;

        // before_t_diff has to be >= 0, if after is <= 0 then we want to check next pair
        if (after_t_diff > 0) {
          // bracket, assemble
          auto after_pose = gps_next_it->second;
          auto before_pose = gps_it->second;

          double diff_sum = after_t_diff + before_t_diff;
          Eigen::Vector3d pos_interp = 
              ((*after_pose * before_t_diff) + (*before_pose * after_t_diff)) / diff_sum;

          double interp_factor_sec = std::min(after_t_diff, before_t_diff)/1e9;
          addGPSFactor(target_frame_it->second->key, pos_interp, 
              params_.gps_sigmas + params_.gps_sigma_per_sec * interp_factor_sec);
        }
      } else {
        // We are on the last element, return so we don't erase
        return;
      }
    }

    // Wipe old history, don't make duplicate factors
    // Also serves to advance the loop
    gps_it = gps_buffer_.erase(gps_it);
  }
}

void PoseGraph::addGPSFactor(const gtsam::Key& key,
    const Eigen::Vector3d& utm_pos, const Eigen::Vector3d& sigma) 
{
  if (graph_.exists(initial_pose_factor_id_)) {
    graph_.remove(initial_pose_factor_id_);
  }
  if (!params_.fix_scale) {
    if (graph_.exists(initial_scale_factor_id_) && gps_factor_count_ > 0) {
      // Takes 2 GPS constraints to determine scale
      graph_.remove(initial_scale_factor_id_);
    }
  }

  graph_.emplace_shared<gtsam::GPSFactor>(key, utm_pos,
      gtsam::noiseModel::Diagonal::Sigmas(sigma));
  gps_factor_count_++;
}

void PoseGraph::update() {
  if (last_opt_size_ + params_.num_frames_opt > size_ && getScale() > 0) {
    // Haven't gotten enough new frames to bother running new opt
    return;
  }
  last_opt_size_ = size_;

  gtsam::LevenbergMarquardtParams opt_params;
  gtsam::LevenbergMarquardtOptimizer opt(graph_, current_opt_, opt_params);
  current_opt_ = opt.optimize();

  if (getScale() < 0) {
    // With motion in the x-y plane, can end up with false local min
    current_opt_.update<double>(S(0), getScale()*-1);
    for (size_t ind=0; ind<size(); ind++) {
      auto cur_pose = getPoseAtIndex(ind);
      cur_pose.rotate(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()));
      current_opt_.update<gtsam::Pose3>(P(ind), Eigen2GTSAM(cur_pose));
    }

    // Run optimizer again
    // We could recurse, but want to not get infinite loop
    gtsam::LevenbergMarquardtOptimizer opt2(graph_, current_opt_, opt_params);
    current_opt_ = opt2.optimize();
  }
}

std::optional<Eigen::Isometry3d> PoseGraph::getPoseAtTime(long stamp) const {
  auto element = pose_history_.find(stamp);
  if (element == pose_history_.end()) return {};

  return GTSAM2Eigen(current_opt_.at<gtsam::Pose3>(element->second->key));
}

Eigen::Isometry3d PoseGraph::getPoseAtIndex(size_t ind) const {
  if (ind >= size_) {
    throw std::out_of_range("Index out of range");
  }
  return GTSAM2Eigen(current_opt_.at<gtsam::Pose3>(P(ind)));
}

double PoseGraph::getScale() const {
  return current_opt_.at<double>(S(0));
}

size_t PoseGraph::size() const {
  return size_;
}

double PoseGraph::getError() const {
  return graph_.error(current_opt_);
}

bool PoseGraph::isInitialized() const {
  return ((size() >= params_.num_frames_init && gps_factor_count_ >= params_.num_frames_init) || 
         params_.fix_scale) && getScale() > 0;
}
