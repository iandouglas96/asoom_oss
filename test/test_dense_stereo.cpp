#include <gtest/gtest.h>
#include <ros/package.h>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include "asoom/dense_stereo.h"
#include "asoom/rectifier.h"

TEST(ASOOM_dense_stereo_test, test_stereo) {
  cv::Mat im1 = cv::imread(ros::package::getPath("asoom") + 
                           "/test/test_imgs/img1.jpg");
  Eigen::Isometry3d pose1 = Eigen::Isometry3d::Identity();
  pose1.translate(90*Eigen::Vector3d(
       -0.0371524,
       -0.0682094,
       -0.00340335));
  pose1.rotate(Eigen::Quaterniond(
        0.999921,
       -0.00550298,
       -0.0108288,
       -0.00308136)); //wxyz
  Keyframe k1(1635642164881558848, im1, pose1);
  
  cv::Mat im2 = cv::imread(ros::package::getPath("asoom") + 
                           "/test/test_imgs/img2.jpg");
  Eigen::Isometry3d pose2 = Eigen::Isometry3d::Identity();
  pose2.translate(90*Eigen::Vector3d(
       -0.026517,
       -0.0470873,
       -0.00254835));
  pose2.rotate(Eigen::Quaterniond(
        0.999965,
       -0.00367989,
       -0.00742567,
       -0.00131107));
  Keyframe k2(1635642165797544512, im2, pose2);

  Rectifier rect(Rectifier::Params(
        ros::package::getPath("asoom") + "/config/titan_wide.yaml", 0.5));
  cv::Mat i1m1, i1m2, i2m1, i2m2;
  auto transforms = rect.genRectifyMaps(k1, k2, i1m1, i1m2, i2m1, i2m2);

  cv::Mat rect1, rect2;
  rect.rectifyImage(im1, i1m1, i1m2, rect1);
  rect.rectifyImage(im2, i2m1, i2m2, rect2);

  DenseStereo stereo(DenseStereo::Params{});
  cv::Mat disp;
  stereo.computeDisp(rect1, rect2, disp);
  cv::imwrite("asoom_disp_viz.png", disp*255/80);
  std::cout << "Wrote asoom_disp_viz.png to test disparity" << std::endl << std::flush;

  // Approximate depth at center of image
  // d = (fx * baseline) / disp
  double depth_center = ((pose1.translation() - pose2.translation()).norm() * 
      rect.getOutputK()(0, 0)) / disp.at<double>(disp.size().height/2, disp.size().width/2);
  EXPECT_NEAR(depth_center, 80, 20);
  // Should be 0 everywhere where depth undefined
  EXPECT_FLOAT_EQ(disp.at<double>(0, 0), 0);

  // Have not set intrinsics yet
  EXPECT_THROW({
      stereo.projectDepth(disp, 1);
    }, DenseStereo::intrinsic_mismatch_exception
  );
  stereo.setIntrinsics(rect.getOutputK(), rect.getOutputSize());

  std::shared_ptr<Eigen::Array3Xd> depth_pc = stereo.projectDepth(disp, 
      (pose1.translation() - pose2.translation()).norm());
  EXPECT_FLOAT_EQ((*depth_pc)(2, disp.size().height*disp.size().width/2 + disp.size().width/2), 
      depth_center);
  EXPECT_TRUE(std::isinf((*depth_pc)(2, 0)));
}
