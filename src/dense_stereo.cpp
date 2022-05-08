#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include "asoom/dense_stereo.h"

DenseStereo::DenseStereo(const Params& params) : params_(params) {
  if (params.use_sgbm) {
    int P1 = params.P1_coeff*3*params.block_size*params.block_size;
    int P2 = params.P2_coeff*3*params.block_size*params.block_size;

    stereo_ = cv::StereoSGBM::create(params.min_disparity, params.num_disparities,
        params.block_size, P1, P2, params.disp_12_map_diff, params.pre_filter_cap,
        params.uniqueness_ratio, params.speckle_window_size, params.speckle_range);
  } else {
    stereo_ = cv::StereoBM::create(params.num_disparities, params.block_size);
  }
}

void DenseStereo::computeDisp(const cv::Mat& im1, const cv::Mat& im2, cv::Mat& disp) {
  if (params_.use_sgbm) {
    stereo_->compute(im1, im2, disp);
  } else {
    // BM does not support color images
    cv::Mat im1_grey, im2_grey;
    cv::cvtColor(im1, im1_grey, cv::COLOR_BGR2GRAY);
    cv::cvtColor(im2, im2_grey, cv::COLOR_BGR2GRAY);
    stereo_->compute(im1_grey, im2_grey, disp);
  }

  // We use 64F so when converting to Eigen everything is double
  disp.convertTo(disp, CV_64F);
  disp /= std::pow(2, 4);

  // Mask out where one or the other images is not there
  cv::Mat imgcomb, imgbuf, mask;
  cv::cvtColor(im1, imgbuf, cv::COLOR_BGR2GRAY);
  cv::cvtColor(im2, imgcomb, cv::COLOR_BGR2GRAY);
  cv::bitwise_and(imgcomb, 0, imgcomb, imgbuf<1);
  // Borders tend to be weird
  cv::dilate(imgcomb<1, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(30, 30)));
  cv::bitwise_and(disp, 0, disp, mask);
}

void DenseStereo::setIntrinsics(const Eigen::Matrix3d& K, const cv::Size& size) {
  K_ = K;
  img_plane_pts_ = Eigen::Array3Xd::Ones(3, size.height * size.width);
  // Note: These indices are assuming unrolled row-major a-la OpenCV
  // Index for col (x)
  // 0,1,2,0,1,2,...
  img_plane_pts_.row(0) = Eigen::RowVectorXd::LinSpaced(size.width, 0, size.width-1).replicate(
      1, size.height);
  // Index for row (y)
  // 0,0,0,1,1,1,2,2,2,...
  img_plane_pts_.row(1) = Eigen::RowVectorXi::LinSpaced(size.width*size.height, 
      0, size.height-1).cast<double>();

  img_plane_pts_ = K_.inverse() * img_plane_pts_.matrix();
}

std::shared_ptr<Eigen::Array3Xd> DenseStereo::projectDepth(
    const cv::Mat& disp, double baseline) 
{
  if (disp.size().height*disp.size().width != img_plane_pts_.cols()) {
    throw intrinsic_mismatch_exception();
  }

  // Convert to Eigen and flatten while we are at it
  // Flattens to row-major, since OpenCV is managing storage here
  Eigen::Map<const Eigen::RowVectorXd> disp_eig(disp.ptr<double>(), disp.rows*disp.cols);
  // All the invalid zeros will now be infs due to inversion
  return std::make_shared<Eigen::Array3Xd>(
      img_plane_pts_.rowwise() * (baseline * K_(0, 0) / disp_eig.array()));
}
