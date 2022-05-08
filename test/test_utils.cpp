#include <gtest/gtest.h>
#include <ros/package.h>
#include "asoom/utils.h"
#include "asoom/semantic_color_lut.h"

TEST(ASOOM_utils, test_utm) {
  // Compare with https://www.latlong.net/lat-long-utm.html
  int zone;
  auto utm = LatLong2UTM(Eigen::Vector2d(0,0), zone);
  EXPECT_NEAR(utm[0], 166021.44, 0.01);
  EXPECT_NEAR(utm[1], 0, 0.01);
  EXPECT_EQ(zone, 31);

  utm = LatLong2UTM(Eigen::Vector2d(39.941676, -75.199431), zone);
  EXPECT_NEAR(utm[0], 482962.15, 0.01);
  EXPECT_NEAR(utm[1], 4421302.89, 0.01);
  EXPECT_EQ(zone, 18);

  utm = LatLong2UTM(Eigen::Vector2d(-34.358175l, 18.498917), zone);
  EXPECT_NEAR(utm[0], 269977.60, 0.01);
  EXPECT_NEAR(utm[1], 6195294.67, 0.01);
  EXPECT_EQ(zone, -34);

  utm = LatLong2UTM(Eigen::Vector2d(-39.261566, 177.865155), zone);
  EXPECT_NEAR(utm[0], 574639.25, 0.01);
  EXPECT_NEAR(utm[1], 5653839.86, 0.01);
  EXPECT_EQ(zone, -60);

  utm = LatLong2UTM(Eigen::Vector2d(28.608336, -80.604200), zone);
  EXPECT_NEAR(utm[0], 538695.49, 0.01);
  EXPECT_NEAR(utm[1], 3164657.86, 0.01);
  EXPECT_EQ(zone, 17);
}

TEST(ASOOM_utils, test_sem_color_lut) {
  SemanticColorLut lut(ros::package::getPath("asoom") + "/config/semantic_lut.yaml");  
  // Some unknown class
  auto color = SemanticColorLut::unpackColor(lut.ind2Color(34));
  EXPECT_EQ(color[0], 0);
  EXPECT_EQ(color[1], 0);
  EXPECT_EQ(color[2], 0);
  EXPECT_EQ(lut.color2Ind(SemanticColorLut::packColor(color[0], color[1], color[2])), 255);
  color = SemanticColorLut::unpackColor(lut.ind2Color(0));
  EXPECT_EQ(color[0], 255);
  EXPECT_EQ(color[1], 0);
  EXPECT_EQ(color[2], 0);
  EXPECT_EQ(lut.color2Ind(SemanticColorLut::packColor(color[0], color[1], color[2])), 0);

  cv::Mat img = cv::Mat::zeros(1000, 1000, CV_8UC1);
  for (uint8_t i=0; i<6; i++) {
    img.at<uint8_t>(i, 0) = i;
  }
  cv::Mat color_img;
  lut.ind2Color(img, color_img);
  EXPECT_EQ(color_img.type(), CV_8UC3);
  // These are BGR because OpenCV
  EXPECT_EQ(color_img.at<cv::Vec3b>(0, 0), cv::Vec3b(255, 0, 0));
  EXPECT_EQ(color_img.at<cv::Vec3b>(1, 0), cv::Vec3b(0, 255, 0));
  EXPECT_EQ(color_img.at<cv::Vec3b>(3, 0), cv::Vec3b(0, 100, 0));
  EXPECT_EQ(color_img.at<cv::Vec3b>(4, 0), cv::Vec3b(255, 255, 0));
  EXPECT_EQ(color_img.at<cv::Vec3b>(5, 0), cv::Vec3b(0, 0, 0));

  // Go backwards
  lut.color2Ind(color_img, img);
  EXPECT_EQ(img.type(), CV_8UC1);
  for (uint8_t i=0; i<5; i++) {
    EXPECT_EQ(img.at<uint8_t>(i, 0), i);
  }
  EXPECT_EQ(img.at<uint8_t>(5, 0), 255);
}
