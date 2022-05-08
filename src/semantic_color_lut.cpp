#include <yaml-cpp/yaml.h>
#include "asoom/semantic_color_lut.h"

SemanticColorLut::SemanticColorLut(const std::string& lut_path) {
  if (lut_path != NO_SEM) {
    const YAML::Node lut = YAML::LoadFile(lut_path);
    for (const auto& color : lut) {
      if (color.size() != 3) {
        throw invalid_lut_file_exception();
      }

      uint32_t rgb_packed = packColor(color[0].as<uint32_t>(),
                                      color[1].as<uint32_t>(),
                                      color[2].as<uint32_t>());

      color_to_ind_.insert({rgb_packed, ind_to_color_.size()});
      ind_to_color_.push_back(rgb_packed);
    }
  }
}

void SemanticColorLut::ind2Color(const cv::Mat& ind_mat, cv::Mat& color_mat) {
  cv::cvtColor(ind_mat, color_mat, cv::COLOR_GRAY2BGR);

  using Pixel = cv::Point3_<uint8_t>;
  color_mat.forEach<Pixel>([&](Pixel& pixel, const int position[]) -> void {
    auto color = unpackColor(ind2Color(pixel.x));
    pixel.x = color[0]; // b
    pixel.y = color[1]; // g
    pixel.z = color[2]; // r
  });
}

void SemanticColorLut::color2Ind(const cv::Mat& color_mat, cv::Mat& ind_mat) {
  cv::Mat color_mat_deep = color_mat.clone();

  using Pixel = cv::Point3_<uint8_t>;
  color_mat_deep.forEach<Pixel>([&](Pixel& pixel, const int position[]) -> void {
    uint8_t ind = color2Ind(SemanticColorLut::packColor(pixel.x, pixel.y, pixel.z));
    pixel = {ind, ind, ind};
  });

  cv::cvtColor(color_mat_deep, ind_mat, cv::COLOR_BGR2GRAY);
}
