#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "mnnruntime.hpp"


class AutoExposure{
public:
	/// \brief 初始化
	/// \param mnn_path
	explicit AutoExposure(std::string_view mnn_path)
	{
		ort_ = std::make_shared<MNNRuntime>(mnn_path);
        assert(ort_->input_channel == 1);
		assert(ort_->num_outputs == 1);
	}
	~AutoExposure() = default;

public:
	/// \brief 执行自动曝光
	/// \param input BGR HWC 8UC3
	/// \param ref_size 推理图像大小
	/// \return 增强图像  BGR HWC 8UC1
	cv::Mat exposure(const cv::Mat &input, size_t ref_size=256);

private:
	/// \brief 应用曲线到图像, 先在小图（img_bgr_map）上计算映射矩阵，再把映射矩阵放大到图像大小，再和原图计算
	/// \param img_bgr 自动曝光的原图像 BGR HWC  CV_32FC3
	/// \param img_bgr_map 小图 BGR HWC CV_32FC3 
    /// \param img_gray_map 推理中使用的小图的灰度图 hw CV_32FC1
    /// \param r_tensor_values 亮度曲线
	/// \return 效果图 CV_32FC3
	cv::Mat apply_spline_luma(const cv::Mat &img_bgr, 
							  const cv::Mat &img_bgr_map, 
							  const cv::Mat &img_gray_map, 
							  const std::vector<float> &r_tensor_values);	

	MNN::Session *create_session();

	std::vector<float> fetch_cure_param(MNN::Tensor* tensor);

    cv::Size scale_longe_edge(cv::Size size, size_t ref_size);                        

private:
	/// \brief MNNRuntime
	std::shared_ptr<MNNRuntime> ort_;
};
