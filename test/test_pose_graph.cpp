#include <gtest/gtest.h>
#include "asoom/pose_graph.h"

TEST(ASOOM_pose_graph_test, test_two_nodes) {
  Eigen::Isometry3d init_pose = Eigen::Isometry3d::Identity();
  init_pose.rotate(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY()));

  auto pg = std::make_unique<PoseGraph>(PoseGraph::Params(0.1, 0.1, 0.1), init_pose);
  ASSERT_TRUE(pg);

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  // Make starting point not at 0,0,0 so optimizer does something
  pose.translate(Eigen::Vector3d(1,0,0)); 
  size_t ind = pg->addFrame(10, pose);
  EXPECT_EQ(ind, 0);

  pose.translate(Eigen::Vector3d(1,0,0)); 
  ind = pg->addFrame(20, pose);
  EXPECT_EQ(ind, 1);

  // Test optimization
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(0).translation()[0], 0);
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(1).translation()[0], 1);
  pg->update();
  // We rotate around y for init without gps, so x gets flipped
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(0).translation()[0], 0);
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(1).translation()[0], -1);

  // Test Time getter
  EXPECT_FALSE(pg->getPoseAtTime(100));
  EXPECT_TRUE(pg->getPoseAtTime(10));
  EXPECT_FLOAT_EQ(pg->getPoseAtTime(20)->translation()[0], -1);

  EXPECT_FLOAT_EQ(pg->getScale(), 1);
  EXPECT_NEAR(pg->getError(), 0, 0.00001);
}

TEST(ASOOM_pose_graph_test, test_gps) {
  auto pg = std::make_unique<PoseGraph>(PoseGraph::Params(0.1, 0.1, 0.1));
  ASSERT_TRUE(pg);

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  // Make starting point not at 0,0,0 so optimizer does something
  pose.translate(Eigen::Vector3d(1,0,0)); 
  size_t ind = pg->addFrame(10, pose);
  EXPECT_EQ(ind, 0);

  pose.translate(Eigen::Vector3d(1,0,0)); 
  ind = pg->addFrame(20, pose);
  EXPECT_EQ(ind, 1);

  pg->addGPS(10, Eigen::Vector3d(10,0,0));
  pg->addGPS(20, Eigen::Vector3d(20,0,0));

  // Test optimization
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(0).translation()[0], 0);
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(1).translation()[0], 1);
  pg->update();
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(0).translation()[0], 10);
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(1).translation()[0], 20);

  EXPECT_FLOAT_EQ(pg->getScale(), 10);
  EXPECT_NEAR(pg->getError(), 0, 0.00001);
}

TEST(ASOOM_pose_graph_test, test_gps_rot) {
  auto pg = std::make_unique<PoseGraph>(PoseGraph::Params(0.1, 0.1, 0.1));
  ASSERT_TRUE(pg);

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  // Make starting point not at 0,0,0 so optimizer does something
  pose.translate(Eigen::Vector3d(1,0,0)); 
  size_t ind = pg->addFrame(10, pose);
  EXPECT_EQ(ind, 0);

  pose.translate(Eigen::Vector3d(1,0,0)); 
  pose.rotate(Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitZ()));
  ind = pg->addFrame(20, pose);
  EXPECT_EQ(ind, 1);

  pose.translate(Eigen::Vector3d(0,-1,0)); 
  pose.rotate(Eigen::AngleAxisd(M_PI/4, Eigen::Vector3d::UnitX()));
  ind = pg->addFrame(30, pose);
  EXPECT_EQ(ind, 2);

  pg->addGPS(10, Eigen::Vector3d(10,0,0));
  pg->addGPS(20, Eigen::Vector3d(20,0,0));
  pg->addGPS(30, Eigen::Vector3d(30,0,0));

  // Test optimization
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(0).translation()[0], 0);
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(1).translation()[0], 1);
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(2).translation()[0], 2);
  pg->update();
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(0).translation()[0], 10);
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(1).translation()[0], 20);
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(2).translation()[0], 30);

  EXPECT_FLOAT_EQ(pg->getScale(), 10);
  EXPECT_NEAR(pg->getError(), 0, 0.00001);
}

TEST(ASOOM_pose_graph_test, test_init) {
  auto pg = std::make_unique<PoseGraph>(PoseGraph::Params(0.1, 0.1, 0.1));
  ASSERT_TRUE(pg);

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  // Make starting point not at 0,0,0 so optimizer does something
  pose.translate(Eigen::Vector3d(1,0,0)); 
  size_t ind = pg->addFrame(10, pose);
  EXPECT_EQ(ind, 0);

  pose.translate(Eigen::Vector3d(1,0,0)); 
  ind = pg->addFrame(20, pose);
  EXPECT_EQ(ind, 1);

  // GPS priors rotate entire pose set
  pg->addGPS(10, Eigen::Vector3d(0,10,0));
  pg->addGPS(20, Eigen::Vector3d(0,20,0));

  // Test optimization
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(0).translation()[0], 0);
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(1).translation()[0], 1);
  pg->update();
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(0).translation()[1], 10);
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(1).translation()[1], 20);
  EXPECT_NEAR(Eigen::Quaterniond(pg->getPoseAtIndex(0).rotation()).w(), 0.7071, 0.001);
  EXPECT_NEAR(Eigen::Quaterniond(pg->getPoseAtIndex(0).rotation()).z(), 0.7071, 0.001);
  EXPECT_NEAR(Eigen::Quaterniond(pg->getPoseAtIndex(1).rotation()).w(), 0.7071, 0.001);
  EXPECT_NEAR(Eigen::Quaterniond(pg->getPoseAtIndex(1).rotation()).z(), 0.7071, 0.001);

  EXPECT_FLOAT_EQ(pg->getScale(), 10);
  EXPECT_NEAR(pg->getError(), 0, 0.00001);

  // Add new pose, take into account current estimates
  pose.translate(Eigen::Vector3d(1,0,0)); 
  ind = pg->addFrame(30, pose);
  EXPECT_EQ(ind, 2);
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(2).translation()[1], 30);
  EXPECT_NEAR(Eigen::Quaterniond(pg->getPoseAtIndex(2).rotation()).w(), 0.7071, 0.001);
  EXPECT_NEAR(Eigen::Quaterniond(pg->getPoseAtIndex(2).rotation()).z(), 0.7071, 0.001);
  EXPECT_NEAR(pg->getError(), 0, 0.00001);

  pose.translate(Eigen::Vector3d(0,1,0)); 
  pose.rotate(Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitZ()));
  ind = pg->addFrame(30, pose);
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(3).translation()[1], 30);
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(3).translation()[0], -10);
  EXPECT_NEAR(pg->getError(), 0, 0.00001);
}

TEST(ASOOM_pose_graph_test, test_gps_bracket) {
  auto pg = std::make_unique<PoseGraph>(PoseGraph::Params(0.1, 0.1, 0.1));
  ASSERT_TRUE(pg);

  pg->addGPS(4, Eigen::Vector3d(100,0,0));
  pg->addGPS(8, Eigen::Vector3d(8,0,0));
  pg->addGPS(15, Eigen::Vector3d(15,0,0));

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  // Make starting point not at 0,0,0 so optimizer does something
  pose.translate(Eigen::Vector3d(1,0,0)); 
  size_t ind = pg->addFrame(10, pose);
  EXPECT_EQ(ind, 0);

  pose.translate(Eigen::Vector3d(1,0,0)); 
  ind = pg->addFrame(20, pose);
  EXPECT_EQ(ind, 1);

  // Test optimization
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(0).translation()[0], 0);
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(1).translation()[0], 1);
  pg->update();
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(0).translation()[0], 10);
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(1).translation()[0], 11);
  EXPECT_FLOAT_EQ(pg->getScale(), 1);

  pg->addGPS(21, Eigen::Vector3d(21,0,0));
  pg->update();
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(0).translation()[0], 10);
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(1).translation()[0], 20);

  EXPECT_FLOAT_EQ(pg->getScale(), 10);
  EXPECT_NEAR(pg->getError(), 0, 0.00001);
}

TEST(ASOOM_pose_graph_test, test_cov) {
  auto pg = std::make_unique<PoseGraph>(PoseGraph::Params(0.1, 0.1, 0.0, 0.1, true));
  ASSERT_TRUE(pg);

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();

  pose.translate(Eigen::Vector3d(1,0,0)); 
  size_t ind = pg->addFrame(10e9, pose);
  pg->addGPS(10e9, Eigen::Vector3d(1,0,0));
  EXPECT_EQ(ind, 0);

  pg->addGPS(19e9, Eigen::Vector3d(2,0,0));
  pose.translate(Eigen::Vector3d(1,0,0)); 
  ind = pg->addFrame(20e9, pose);
  EXPECT_EQ(ind, 1);
  pg->addGPS(25e9, Eigen::Vector3d(3,0,0));

  // Test optimization
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(0).translation()[0], 0);
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(1).translation()[0], 1);
  pg->update();
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(0).translation()[0], 1);
  EXPECT_NEAR(pg->getPoseAtIndex(1).translation()[0], 2.1, 0.05);
  EXPECT_FLOAT_EQ(pg->getScale(), 1);
}

TEST(ASOOM_pose_graph_test, test_neg_scale) {
  auto pg = std::make_unique<PoseGraph>(PoseGraph::Params(0.1, 0.1, 0.1));
  ASSERT_TRUE(pg);

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  // Make starting point not at 0,0,0 so optimizer does something
  pose.translate(Eigen::Vector3d(1,0,0)); 
  size_t ind = pg->addFrame(10, pose);
  EXPECT_EQ(ind, 0);

  pose.translate(Eigen::Vector3d(1,0,0)); 
  ind = pg->addFrame(20, pose);
  EXPECT_EQ(ind, 1);

  pg->addGPS(10, Eigen::Vector3d(-10,0,0));
  pg->addGPS(20, Eigen::Vector3d(-20,0,0));

  // Test optimization
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(0).translation()[0], 0);
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(1).translation()[0], 1);
  pg->update();
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(0).translation()[0], -10);
  EXPECT_FLOAT_EQ(pg->getPoseAtIndex(1).translation()[0], -20);

  EXPECT_FLOAT_EQ(pg->getScale(), 10);
  EXPECT_NEAR(pg->getError(), 0, 0.00001);
}
