#pragma once
#include <string>
#include <opencv2/core.hpp>

/*
    model_path: onnx模型全路径
    down_scale: 缩小倍数，大于等于1， 影响计算时间，数字越大计算越快，但精度有一定影响
    map_point_wise: 
                1. true 在原图上计算曲线映射同时应用到原图上，这种方式精度更好但速度更慢，这种方式中 down_scale 仅仅控制计算曲线参数的图像大小； 
                2. false 在小图上计算曲线映射，再应用到原图上，这种方式中 down_scale 不仅控制计算曲线参数的图像大小还控制曲线映射的图像大小
*/
void enahnce_init_model(const std::string model_path, size_t down_scale=16, bool map_point_wise=false);

/*
    img_bgr: 3通道归一化的原始图像，通道顺序是[b g r],图像值[0, 1.0]
    return 
        3通道[b g r]图像，图像值[0, 1.0]
*/
cv::Mat enahnce_run(const cv::Mat &img_bgr);

void enahnce_release_model();

