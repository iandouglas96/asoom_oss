#include <iostream>
#include <fstream>
#include "asoom/keyframe.h"
#include "asoom/semantic_color_lut.h"

DepthCloudArray Keyframe::getDepthCloud() const {
  if (!hasDepth()) {
    return DepthCloudArray::Zero(5, 0);
  }

  // transform to global frame
  Eigen::Isometry3d trans = getRectPose();

  DepthCloudArray cloud(5, depth_->cols());
  size_t dense_ind;
  size_t sparse_ind = 0;
  cv::Vec3b bgr;
  for (size_t x=0; x<rect_img_.size().width; x++) {
    for (size_t y=0; y<rect_img_.size().height; y++) {
      // Get row major index
      dense_ind = y*rect_img_.size().width + x;

      // Undefined depth
      if ((*depth_)(2, dense_ind) > 150 || (*depth_)(2, dense_ind) < 1) {
        continue;
      }
      
      // The weird PCL color format
      bgr = rect_img_.at<cv::Vec3b>(y, x);
      uint32_t color_packed = SemanticColorLut::packColor(bgr[0], bgr[1], bgr[2]);
      cloud.block<3,1>(0, sparse_ind) = (trans * depth_->col(dense_ind)).cast<float>();
      cloud(3, sparse_ind) = *reinterpret_cast<float*>(&color_packed);

      // We pack in the class integer in the same way
      uint8_t sem_class = 255;
      if (hasSem() && sem_img_.size() == rect_img_.size()) {
        sem_class = sem_img_.at<uint8_t>(y, x);
      }
      cloud(4, sparse_ind) = sem_class; 

      sparse_ind++;
    }
  }

  cloud.conservativeResize(Eigen::NoChange_t(), sparse_ind);
  return cloud;
}

bool Keyframe::needsMapUpdate(float delta_d, float delta_theta) const {
  Eigen::Isometry3d diff = pose_.inverse() * map_pose_;
  if (diff.translation().norm() > delta_d || 
      Eigen::AngleAxisd(diff.rotation()).angle() > delta_theta) {
    return true;
  }
  return false;
}

void Keyframe::saveDataBinary(const cv::Mat& img, std::ofstream& outfile) {
  // Write header
  int rows = img.size().height;
  outfile.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  int cols = img.size().width;
  outfile.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
  int type = img.type();
  outfile.write(reinterpret_cast<const char *>(&type), sizeof(type));

  size_t size = img.total() * img.elemSize();
  outfile.write(reinterpret_cast<const char *>(img.data), size);
}

void Keyframe::saveDataBinary(const std::shared_ptr<Eigen::Array3Xd>& arr, 
    std::ofstream& outfile) 
{
  // Write header
  int rows = 0;
  int cols = 0;
  if (arr) {
    rows = arr->rows();
    cols = arr->cols();
  }
  int type = 1; // somewhat arbitrary to indicate double

  outfile.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  outfile.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
  outfile.write(reinterpret_cast<const char *>(&type), sizeof(type));

  if (arr) {
    size_t size = arr->size() * sizeof(double);
    outfile.write(reinterpret_cast<const char *>(arr->data()), size);
  }
}

void Keyframe::readDataBinary(std::ifstream& infile, cv::Mat& img) {
  // Read header
  int rows, cols, type;
  infile.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  infile.read(reinterpret_cast<char *>(&cols), sizeof(cols));
  infile.read(reinterpret_cast<char *>(&type), sizeof(type));

  img = cv::Mat(rows, cols, type);
  size_t size = img.total() * img.elemSize();
  infile.read(reinterpret_cast<char *>(img.data), size);
}

void Keyframe::readDataBinary(std::ifstream& infile, 
    std::shared_ptr<Eigen::Array3Xd>& arr) 
{
  // Read header
  int rows, cols, type;
  infile.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  infile.read(reinterpret_cast<char *>(&cols), sizeof(cols));
  infile.read(reinterpret_cast<char *>(&type), sizeof(type));

  if (rows > 0 && cols > 0) {
    arr = std::make_shared<Eigen::Array3Xd>(rows, cols);
    size_t size = arr->size() * sizeof(double);
    infile.read(reinterpret_cast<char *>(arr->data()), size);
  }
}

void Keyframe::saveToDisk() {
  // This is pretty ugly.  We are going to literally stream the data to disk.
  // This is arguably rather unsafe, but it's very fast, because we don't have
  // to deal with formatting the data when we are reading/writing.
  
  if (!on_disk_) {
    std::ostringstream name; 
    name << getenv("HOME") << "/.ros/asoom_cache/frame_" << stamp_ << ".bin";
    std::ofstream outfile(name.str(), std::ios::binary | std::ios::trunc);

    saveDataBinary(img_, outfile);
    saveDataBinary(rect_img_, outfile);
    saveDataBinary(sem_img_, outfile);
    saveDataBinary(depth_, outfile);

    outfile.close();
  }

  img_.release();
  rect_img_.release();
  sem_img_.release();
  depth_.reset();
  on_disk_ = true;
  in_mem_ = false;
}

bool Keyframe::loadFromDisk() {
  if (in_mem_) {
    return true;
  }
  if (!on_disk_) {
    return false;
  }

  std::ostringstream name; 
  name << getenv("HOME") << "/.ros/asoom_cache/frame_" << stamp_ << ".bin";
  std::ifstream infile(name.str(), std::ios::binary);

  if (!infile.is_open()) {
    return false;
  }
  
  readDataBinary(infile, img_);
  readDataBinary(infile, rect_img_);
  readDataBinary(infile, sem_img_);
  readDataBinary(infile, depth_);

  in_mem_ = true;
  return on_disk_;
}
