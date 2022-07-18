//
// Created by ts on 2022/7/6.
//

#pragma once

#include <iostream>
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>


/// \brief MNN runtime 初始工具类
class MNNRuntime
{
public:
	std::unique_ptr<MNN::Interpreter> mnn_interpreter;
	MNN::Session *mnn_session = nullptr;
	MNN::Tensor *input_tensor = nullptr; // assume single input.
	MNN::ScheduleConfig schedule_config;
	std::unique_ptr<MNN::CV::ImageProcess> pretreat; // init at subclass
	int input_batch;
	int input_channel;
	int input_height;
	int input_width;
	int dimension_type;
	int num_outputs;

public:
	explicit MNNRuntime(std::string_view mnn_path)
	{
		// 1. init interpreter
		mnn_interpreter = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.data()));
		// 2. init schedule_config
		int num_threads = 8;
		schedule_config.numThread = num_threads;
		schedule_config.type = MNN_FORWARD_AUTO;
		MNN::BackendConfig backend_config;
		backend_config.precision = MNN::BackendConfig::Precision_High; // default Precision_High
		schedule_config.backendConfig = &backend_config;
		// 3. create session
		mnn_session = mnn_interpreter->createSession(schedule_config);
		// 4. init input tensor
		input_tensor = mnn_interpreter->getSessionInput(mnn_session, nullptr);
		// 5. init input dims
		input_batch = input_tensor->batch();
		input_channel = input_tensor->channel();
		input_height = input_tensor->height();
		input_width = input_tensor->width();
		dimension_type = input_tensor->getDimensionType();
		// 6. resize tensor & session needed ???
		if (dimension_type == MNN::Tensor::CAFFE) {
			// NCHW
			mnn_interpreter->resizeTensor(input_tensor, {input_batch, input_channel, input_height, input_width});
			mnn_interpreter->resizeSession(mnn_session);
		} // NHWC
		else if (dimension_type == MNN::Tensor::TENSORFLOW) {
			mnn_interpreter->resizeTensor(input_tensor, {input_batch, input_height, input_width, input_channel});
			mnn_interpreter->resizeSession(mnn_session);
		}
		// output count
		num_outputs = (int) mnn_interpreter->getSessionOutputAll(mnn_session).size();

		if (true) {
			std::cout << "=============== Input-Dims ==============" << std::endl;
			if (input_tensor) input_tensor->printShape();
			if (dimension_type == MNN::Tensor::CAFFE)
				std::cout << "Dimension Type: (CAFFE/PyTorch/ONNX)NCHW" << std::endl;
			else if (dimension_type == MNN::Tensor::TENSORFLOW)
				std::cout << "Dimension Type: (TENSORFLOW)NHWC" << std::endl;
			else if (dimension_type == MNN::Tensor::CAFFE_C4)
				std::cout << "Dimension Type: (CAFFE_C4)NC4HW4" << std::endl;
			std::cout << "=============== Output-Dims ==============" << std::endl;
			auto tmp_output_map = mnn_interpreter->getSessionOutputAll(mnn_session);
			for (const auto &it : tmp_output_map) {
				std::cout << it.first << ": ";
				it.second->printShape();
			}
			std::cout << "========================================\n";
		}

	}

	~MNNRuntime()
	{
		mnn_interpreter->releaseModel();
		if (mnn_session) mnn_interpreter->releaseSession(mnn_session);
	}
};
