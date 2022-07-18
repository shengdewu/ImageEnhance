//
// Created by ts on 2022/1/21.
//

#pragma once

#include <iostream>
#include <string>
#include <cstring>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#define SHOW_LOG


template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}


/// \brief onnx runtime 初始工具类
class ONNXRuntime
{
public:
	Ort::Env env_;
	std::unique_ptr<Ort::Session> session_;
	// cpu memory
	Ort::AllocatorWithDefaultOptions allocator_;
	Ort::MemoryInfo memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	// model weights info
	size_t input_count_, output_count_;
	std::vector<const char*> input_names_, output_names_;
	std::vector<ONNXTensorElementDataType> input_type_, output_type_;
	std::vector<std::vector<int64_t>> input_dims_, output_dims_;

public:
	ONNXRuntime(std::string_view onnx_path, std::string_view log_id):_show_log(true)
	{
		// env
		int num_threads = 8;
		env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, log_id.data());
		// session options
		Ort::SessionOptions session_options;
		session_options.SetIntraOpNumThreads(num_threads);
		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
		session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING);
		// session
		session_ = std::make_unique<Ort::Session>(env_, onnx_path.data(), session_options);

		// info
		input_count_ = session_->GetInputCount();
		for (size_t i = 0; i < input_count_; ++i) {
			input_names_.emplace_back(session_->GetInputName(i, allocator_));
			Ort::TypeInfo inputTypeInfo = session_->GetInputTypeInfo(i);
			auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
			input_type_.emplace_back(inputTensorInfo.GetElementType());
			input_dims_.emplace_back(inputTensorInfo.GetShape());
		}

		output_count_ = session_->GetOutputCount();
		for (size_t i = 0; i < output_count_; ++i) {
			output_names_.emplace_back(session_->GetOutputName(i, allocator_));
			Ort::TypeInfo outputTypeInfo = session_->GetOutputTypeInfo(0);
			auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
			output_type_.emplace_back(outputTensorInfo.GetElementType());
			output_dims_.emplace_back(outputTensorInfo.GetShape());
		}

		if (_show_log) {
			std::cout << log_id << " init with threads count " << num_threads << std::endl;
			std::cout << "input names: " << input_names_ << " dims: " << input_dims_ << std::endl;
			std::cout << "output names: " << output_names_ << " dims: " << output_dims_ << std::endl;
		}

	}
	~ONNXRuntime() = default;

private:
    bool _show_log;
};
