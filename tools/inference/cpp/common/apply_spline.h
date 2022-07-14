#pragma once

#include <opencv2/core.hpp>

cv::Mat apply_spline(const cv::Mat &img_bgr_map, const cv::Mat &img_bgr,
                     const std::vector<float> &r_tensor_values, 
                     const std::vector<float> & g_tensor_values, 
                     const std::vector<float> & b_tensor_values);

cv::Mat apply_spline(const cv::Mat &img_bgr,
                     const std::vector<float> &r_tensor_values, 
                     const std::vector<float> &g_tensor_values, 
                     const std::vector<float> &b_tensor_values);     