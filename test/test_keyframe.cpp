#include <gtest/gtest.h>
#include <chrono>
#include <opencv2/imgproc/imgproc.hpp>
#include "asoom/keyframe.h"

TEST(ASOOM_keyframe_test, test_save_load_img) {
  cv::Mat img(100, 100, CV_8UC3);
  Keyframe k(100, img.clone(), Eigen::Isometry3d::Identity());
  EXPECT_TRUE(!k.hasDepth());
  EXPECT_TRUE(!k.hasSem());
  ASSERT_EQ(img.size(), k.getImg().size());
  EXPECT_EQ(cv::sum(cv::abs(img - k.getImg()))[0], 0);

  k.saveToDisk();
  ASSERT_EQ(k.getImg().size(), cv::Size(0, 0));

  ASSERT_TRUE(k.loadFromDisk());
  ASSERT_EQ(img.size(), k.getImg().size());
  EXPECT_EQ(cv::sum(cv::abs(img - k.getImg()))[0], 0);
}

TEST(ASOOM_keyframe_test, test_save_load_full) {
  using namespace std::chrono;

  cv::Mat img(1280, 800, CV_8UC3);
  Keyframe k(100, img.clone(), Eigen::Isometry3d::Identity());
  cv::Mat sem(1280, 800, CV_8UC1);
  k.setSem(sem);
  auto depth = std::make_shared<Eigen::Array3Xd>(3, 640*400);
  cv::Mat img_rect;
  cv::resize(img, img_rect, img.size()/2);
  k.setDepth(Eigen::Isometry3d::Identity(), img_rect, depth);

  // Sanity check
  ASSERT_EQ(cv::sum(cv::abs(img - k.getImg()))[0], 0);
  ASSERT_EQ(cv::sum(cv::abs(img - k.getImg()))[1], 0);
  ASSERT_EQ(cv::sum(cv::abs(img - k.getImg()))[2], 0);
  int initial_size = k.getDepthCloud().size();

  auto start_t = high_resolution_clock::now();
  k.saveToDisk();
  auto end_t = high_resolution_clock::now();
  std::cout << "Saving: " << duration_cast<microseconds>(end_t - start_t).count() << "us" << std::endl;

  ASSERT_EQ(k.getImg().size(), cv::Size(0, 0));

  // Load, check same data
  start_t = high_resolution_clock::now();
  ASSERT_TRUE(k.loadFromDisk());
  end_t = high_resolution_clock::now();
  std::cout << "Loading: " << duration_cast<microseconds>(end_t - start_t).count() << "us" << std::endl;

  ASSERT_EQ(cv::sum(cv::abs(img - k.getImg()))[0], 0);
  ASSERT_EQ(cv::sum(cv::abs(img - k.getImg()))[1], 0);
  ASSERT_EQ(cv::sum(cv::abs(img - k.getImg()))[2], 0);
  ASSERT_EQ(initial_size, k.getDepthCloud().size());
 
  // Check that loading if already loaded basically instant
  start_t = high_resolution_clock::now();
  ASSERT_TRUE(k.loadFromDisk());
  end_t = high_resolution_clock::now();
  std::cout << "Loading already loaded: " << duration_cast<microseconds>(end_t - start_t).count() << "us" << std::endl;
  ASSERT_TRUE(duration_cast<microseconds>(end_t - start_t).count() < 10);

  // Check that saving if already saved basically instant
  start_t = high_resolution_clock::now();
  // Already saved, so time here is time releasing memory
  k.saveToDisk();
  end_t = high_resolution_clock::now();
  std::cout << "Saving already saved: " << duration_cast<microseconds>(end_t - start_t).count() << "us" << std::endl;
  ASSERT_TRUE(duration_cast<microseconds>(end_t - start_t).count() < 1000);
}
