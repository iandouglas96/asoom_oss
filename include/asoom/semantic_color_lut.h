#pragma once

#include <string>
#include <map>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class SemanticColorLut {
  public:
    /*!
     * Exception class to indicate lut file we can't handle
     */
    class invalid_lut_file_exception: public std::exception
    {
      virtual const char* what() const throw()
      {
        return "Invalid color lut file";
      }
    };

    inline static const std::string NO_SEM = "NO_SEM_MODE";
    SemanticColorLut(const std::string& lut_path = NO_SEM);

    static inline uint32_t packColor(uint8_t b, uint8_t g, uint8_t r) {
      return (static_cast<uint32_t>(r) << 16 | 
              static_cast<uint32_t>(g) << 8 | 
              static_cast<uint32_t>(b));
    }

    static inline std::array<uint8_t, 3> unpackColor(uint32_t color) {
      return {static_cast<uint8_t>(color & 0x0000ff),
              static_cast<uint8_t>((color >> 8) & 0x0000ff),
              static_cast<uint8_t>((color >> 16) & 0x0000ff)};
    }
  
    inline uint32_t ind2Color(uint8_t ind) {
      uint32_t color = 0;
      if (ind < ind_to_color_.size() && ind >= 0) {
        color = ind_to_color_[ind];
      }
      return color;
    }
    void ind2Color(const cv::Mat& ind_mat, cv::Mat& color_mat);

    inline uint8_t color2Ind(uint32_t color) {
      uint8_t ind = 255;
      auto color_ptr = color_to_ind_.find(color);
      if (color_ptr != color_to_ind_.end()) {
        ind = color_ptr->second;
      }
      return ind;
    }
    void color2Ind(const cv::Mat& color_mat, cv::Mat& ind_mat);

  private:
    // Internally represent colors as packed 32 bit values
    std::vector<uint32_t> ind_to_color_;
    std::map<uint32_t, uint8_t> color_to_ind_;
};
