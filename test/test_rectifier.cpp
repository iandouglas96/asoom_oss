#include <gtest/gtest.h>
#include <ros/package.h>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "asoom/rectifier.h"

TEST(ASOOM_rectifier_test, test_init) {
  EXPECT_THROW({
      Rectifier rect(Rectifier::Params("/no/config/here", 1));
    }, YAML::BadFile
  );

  Rectifier rect(Rectifier::Params(ros::package::getPath("asoom") + "/config/grace_quarters.yaml", 1));
  Eigen::Matrix3d K = rect.getOutputK();

  // Sanity check the matrix
  EXPECT_FLOAT_EQ(K(1, 0), 0);
  EXPECT_FLOAT_EQ(K(0, 1), 0);
  EXPECT_FLOAT_EQ(K(2, 0), 0);
  EXPECT_FLOAT_EQ(K(2, 1), 0);
  EXPECT_FLOAT_EQ(K(2, 2), 1);
  EXPECT_TRUE(K(0, 0) > 100);
  EXPECT_TRUE(K(1, 1) > 100);
  EXPECT_TRUE(K(0, 2) > 100);
  EXPECT_TRUE(K(1, 2) > 100);
}

TEST(ASOOM_rectifier_test, test_fisheye_init) {
  Rectifier rect(Rectifier::Params(ros::package::getPath("asoom") + "/config/aerial_pennov.yaml", 1));
  Eigen::Matrix3d K = rect.getOutputK();

  // Sanity check the matrix
  EXPECT_FLOAT_EQ(K(1, 0), 0);
  EXPECT_FLOAT_EQ(K(0, 1), 0);
  EXPECT_FLOAT_EQ(K(2, 0), 0);
  EXPECT_FLOAT_EQ(K(2, 1), 0);
  EXPECT_FLOAT_EQ(K(2, 2), 1);
  EXPECT_TRUE(K(0, 0) > 100);
  EXPECT_TRUE(K(1, 1) > 100);
  EXPECT_TRUE(K(0, 2) > 100);
  EXPECT_TRUE(K(1, 2) > 100);
}

TEST(ASOOM_rectifier_test, test_rect) {
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

  Rectifier rect(Rectifier::Params(ros::package::getPath("asoom") + "/config/titan_wide.yaml", 0.5));
  cv::Mat i1m1, i1m2, i2m1, i2m2;
  auto transforms = rect.genRectifyMaps(k1, k2, i1m1, i1m2, i2m1, i2m2);

  // Each rotation should take each pose to a common frame
  EXPECT_NEAR(Eigen::Quaterniond((pose1 * transforms.first).rotation()).angularDistance(
              Eigen::Quaterniond((pose2 * transforms.second).rotation())), 0, 0.01);

  cv::Mat rect1, rect2;
  rect.rectifyImage(im1, i1m1, i1m2, rect1);
  rect.rectifyImage(im2, i2m1, i2m2, rect2);

  EXPECT_EQ(rect1.size().height*2, im1.size().height);
  EXPECT_EQ(rect1.size().width*2, im1.size().width);

  cv::Mat rect_viz;
  cv::hconcat(rect1, rect2, rect_viz);
  // Draw horizontal lines for checking alignment
  for (int y=10; y<rect_viz.size().height; y+=50) {
    cv::line(rect_viz, cv::Point(0, y), cv::Point(rect_viz.size().width, y), 
        cv::Scalar(255, 255, 255));
  }
  cv::imwrite("asoom_rectification_viz.png", rect_viz);
  std::cout << "Wrote asoom_rectification_viz.png to test rect" << std::endl << std::flush;
}

TEST(ASOOM_rectifier_test, test_fisheye_rect) {
  // These are the same images, not the same calib as aerial_pennov
  // However, the lenses are similar, so should be reasonably close
  // Good enough for a sanity check, though not for stereo
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

  Rectifier rect(Rectifier::Params(ros::package::getPath("asoom") + "/config/aerial_pennov.yaml", 0.5));
  cv::Mat i1m1, i1m2, i2m1, i2m2;
  auto transforms = rect.genRectifyMaps(k1, k2, i1m1, i1m2, i2m1, i2m2);

  // Each rotation should take each pose to a common frame
  EXPECT_NEAR(Eigen::Quaterniond((pose1 * transforms.first).rotation()).angularDistance(
              Eigen::Quaterniond((pose2 * transforms.second).rotation())), 0, 0.01);

  cv::Mat rect1, rect2;
  rect.rectifyImage(im1, i1m1, i1m2, rect1);
  rect.rectifyImage(im2, i2m1, i2m2, rect2);

  EXPECT_EQ(rect1.size().height*2, im1.size().height);
  EXPECT_EQ(rect1.size().width*2, im1.size().width);

  cv::Mat rect_viz;
  cv::hconcat(rect1, rect2, rect_viz);
  // Draw horizontal lines for checking alignment
  for (int y=10; y<rect_viz.size().height; y+=50) {
    cv::line(rect_viz, cv::Point(0, y), cv::Point(rect_viz.size().width, y), 
        cv::Scalar(255, 255, 255));
  }
  cv::imwrite("asoom_fisheye_rectification_viz.png", rect_viz);
  std::cout << "Wrote asoom_fisheye_rectification_viz.png to test rect" << std::endl << std::flush;
}
