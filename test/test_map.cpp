#include <gtest/gtest.h>
#include "asoom/map.h"

TEST(ASOOM_map_test, test_map) {
  Map map(Map::Params(10, 25, 0, 1, 5*M_PI/180), SemanticColorLut());

  // Check initialization
  EXPECT_FLOAT_EQ(map.getMap().atPosition("view_angle", grid_map::Position(0, 0)), M_PI/2);

  grid_map::Position pos;
  map.getMap().getPosition(grid_map::Index(0, 0), pos);
  EXPECT_FLOAT_EQ(pos[0], 50./2 - 0.1/2);
  EXPECT_FLOAT_EQ(pos[1], 50./2 - 0.1/2);
  map.getMap().getPosition(map.getMap().getSize() - 1, pos);
  EXPECT_FLOAT_EQ(pos[0], -50./2 + 0.1/2);
  EXPECT_FLOAT_EQ(pos[1], -50./2 + 0.1/2);

  DepthCloudArray cloud(5, 2);
  grid_map::Position pos1(0.05, 0.05);
  grid_map::Position pos2(5.05, 0.05);
  cloud.col(0) << pos1.cast<float>(), 1, 1, 1;
  cloud.col(1) << pos2.cast<float>(), 2, 2, 2;
  map.addCloud(cloud, Eigen::Isometry3d::Identity(), 0);
  EXPECT_FLOAT_EQ(map.getMap().atPosition("elevation", pos1), 1);
  EXPECT_EQ(map.getMap().atPosition("num_points", pos1), 1);
  EXPECT_FLOAT_EQ(map.getMap().atPosition("elevation", pos2), 2);
  EXPECT_EQ(map.getMap().atPosition("num_points", pos2), 1);
  EXPECT_EQ(map.getMap().atPosition("color", pos1), 1);

  pos1 = grid_map::Position(0.01, 0.09);
  pos2 = grid_map::Position(5.01, 0.09);
  cloud.col(0) << pos1.cast<float>(), 1, 2, 2;
  cloud.col(1) << pos2.cast<float>(), 3, 3, 3;
  map.addCloud(cloud, Eigen::Isometry3d::Identity(), 0);
  EXPECT_FLOAT_EQ(map.getMap().atPosition("elevation", pos1), 1);
  EXPECT_EQ(map.getMap().atPosition("num_points", pos1), 2);
  EXPECT_FLOAT_EQ(map.getMap().atPosition("elevation", pos2), 2.5);
  EXPECT_EQ(map.getMap().atPosition("num_points", pos2), 2);
  EXPECT_EQ(map.getMap().atPosition("color", pos1), 1);

  pos1 = grid_map::Position(0.0, 0.0);
  cloud.col(0) << pos1.cast<float>(), 1, 2, 2;
  map.addCloud(cloud, Eigen::Isometry3d::Identity(), 0);
  EXPECT_EQ(map.getMap().atPosition("color", pos1), 2);

  cv::Mat sem, sem_viz;
  map.getMapSemImg(sem, sem_viz);
  grid_map::Index p;
  map.getMap().getIndex(pos1, p);
  EXPECT_EQ(sem.at<uint8_t>(p.x(), p.y()), 2);
}
