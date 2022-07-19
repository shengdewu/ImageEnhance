#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "mnnruntime.hpp"


class CurveMNN{
public:
	/// \brief 初始化
	/// \param mnn_path
	explicit CurveMNN(std::string_view mnn_path)
	{
		ort_ = std::make_shared<MNNRuntime>(mnn_path);
        assert(ort_->input_channel == 1);
		assert(ort_->num_outputs == 1);
	}
	~CurveMNN() = default;

public:
	/// \brief 执行自动曝光
	/// \param input RGB HWC 8UC3
	/// \param ref_size 推理图像大小
	/// \return 增强图像  RGB HWC 8UC3
	cv::Mat exposure(const cv::Mat &input, size_t ref_size=256);

private:
	/// \brief 应用曲线到图像, 先在小图（img_bgr_map）上计算映射矩阵，再把映射矩阵放大到图像大小，再和原图计算
	/// \param img 自动曝光的原图像 RGB HWC  CV_32FC3
    /// \param gray 推理中使用的小图的灰度图 hw CV_32FC1
    /// \param L 亮度曲线
	/// \return 效果图 RGB CV_32FC3
	static cv::Mat apply_spline_luma(const cv::Mat &img,
							         const cv::Mat &gray,
							         const std::vector<float> &L);

    void fill_input_data(const cv::Mat& in_img);

	static std::vector<float> fetch_cure_param(MNN::Tensor* tensor);

    cv::Size scale_long_edge(cv::Size size, size_t ref_size);


private:
	/// \brief MNNRuntime
	std::shared_ptr<MNNRuntime> ort_;
    /// \brief the name of the inference output
    static const std::string L_NAME;
};
