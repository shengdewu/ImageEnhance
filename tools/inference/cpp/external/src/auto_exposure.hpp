#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "onnxruntime.hpp"


class AutoExposure{
public:
	/// \brief 初始化
	/// \param onnx_path
	/// \param num_threads
	explicit AutoExposure(std::string_view onnx_path)
	{
		ort_ = std::make_shared<ONNXRuntime>(onnx_path, "AutoExposure");
        assert(ort_->input_count_ == 1);
	}
	~AutoExposure() = default;

public:
	/// \brief 执行自动曝光
	/// \param input BGR HWC 8UC3
	/// \return 增强图像  BGR HWC 8UC1
	cv::Mat exposure(const cv::Mat &input);

private:
	/// \brief 应用曲线到图像, 先在小图（img_bgr_map）上计算映射矩阵，再把映射矩阵放大到图像大小，再和原图计算
	/// \param img_bgr_map 推理中使用的小图 BGR HWC CV_32FC3 
    /// \param img_bgr 自动曝光的原图像 BGR HWC  CV_32FC3
    /// \param r_tensor_values r通道对应的参数
    /// \param g_tensor_values g通道对应的参数
    /// \param b_tensor_values b通道对应的参数   
	/// \return 效果图 CV_32FC3
    cv::Mat apply_spline(const cv::Mat &img_bgr_map, const cv::Mat &img_bgr,
                         const std::vector<float> &r_tensor_values, 
                         const std::vector<float> &g_tensor_values, 
                         const std::vector<float> &b_tensor_values);


    cv::Size scale_longe_edge(cv::Size size, size_t ref_size);                        

private:
	/// \brief ONNXRuntime
	std::shared_ptr<ONNXRuntime> ort_;
};
