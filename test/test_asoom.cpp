#include <gtest/gtest.h>
#include <ros/package.h>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include "asoom/asoom.h"

TEST(ASOOM_asoom_test, test_pgo_thread) {
  ASOOM a(ASOOM::Params(100, 100, 100, 1.5), PoseGraph::Params(0.1, 0.1, 0.1, 0, true),
      Rectifier::Params(), DenseStereo::Params(), Map::Params());

  // Sanity check
  EXPECT_EQ(a.getGraph().size(), 0);

  // Build simple graph
  cv::Mat img; // Just empty image since we aren't doing anything with it
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translate(Eigen::Vector3d(1,0,0)); 
  a.addFrame(10, img, pose);
  pose.translate(Eigen::Vector3d(1,0,0)); 
  pose.rotate(Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitZ()));
  a.addFrame(20, img, pose);
  pose.translate(Eigen::Vector3d(0,-1,0)); 
  pose.rotate(Eigen::AngleAxisd(M_PI/4, Eigen::Vector3d::UnitX()));
  a.addFrame(30, img, pose);

  // Wait for optimizer to run
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  auto poses = a.getGraph();

  // Verify that nothing has really changed, except translation to 0
  // We don't expect the middle point, since didn't move enough
  ASSERT_EQ(poses.size(), 2);
  EXPECT_FLOAT_EQ(poses[0].translation()[0], 0);
  EXPECT_FLOAT_EQ(poses[1].translation()[0], -2);
}

TEST(ASOOM_asoom_test, test_pgo_gps_thread) {
  ASOOM a(ASOOM::Params(100, 100, 100, 0), PoseGraph::Params(0.1, 0.1, 0.1, 0, false, 2),
      Rectifier::Params(), DenseStereo::Params(), Map::Params());

  // Sanity check
  EXPECT_EQ(a.getGraph().size(), 0);

  // Build simple graph
  cv::Mat img; // Just empty image since we aren't doing anything with it
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translate(Eigen::Vector3d(1,0,0)); 
  a.addFrame(10, img, pose);
  pose.translate(Eigen::Vector3d(1,0,0)); 
  pose.rotate(Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitZ()));
  a.addFrame(20, img, pose);

  a.addGPS(10, Eigen::Vector3d(10,0,0));
  a.addGPS(20, Eigen::Vector3d(20,0,0));

  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  // Haven't initialized yet
  EXPECT_EQ(a.getGraph().size(), 0);

  pose.translate(Eigen::Vector3d(0,-1,0)); 
  pose.rotate(Eigen::AngleAxisd(M_PI/4, Eigen::Vector3d::UnitX()));
  a.addFrame(30, img, pose);
  a.addGPS(30, Eigen::Vector3d(30,0,0));

  // Wait for optimizer to run
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  auto poses = a.getGraph();

  // Now we have initialized, make sure next pose is in the right place
  ASSERT_EQ(poses.size(), 1);
  EXPECT_FLOAT_EQ(poses[0].translation()[0], 30);
}

TEST(ASOOM_asoom_test, test_stereo_thread) {
  ASOOM a(ASOOM::Params(100, 100, 100, 0.1), PoseGraph::Params(0.1, 0.1, 0.1, 0, true),
    Rectifier::Params(ros::package::getPath("asoom") + "/config/grace_quarters.yaml", 0.5), 
    DenseStereo::Params(), Map::Params());

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
  a.addFrame(0, im1, pose1);

  // Wait to make sure pose graph thread has run
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(a.getNewKeyframes().size(), 1);
  EXPECT_EQ(a.getNewKeyframes().size(), 0);
  
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
  a.addFrame(100, im2, pose2);

  // Wait long enough that stereo has completed
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  EXPECT_EQ(a.getNewKeyframes().size(), 1);
  EXPECT_EQ(a.getNewKeyframes().size(), 0);

  DepthCloudArray pc = a.getDepthCloud(0);
  EXPECT_EQ(pc.rows(), 5);
  EXPECT_EQ(pc.cols(), 0);

  pc = a.getDepthCloud(100);
  EXPECT_EQ(pc.rows(), 5);
  EXPECT_TRUE(pc.cols() > 10000);
}
