#include <iostream>
#include <vector>
#include <string>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <numeric>
#include <time.h>
#include <chrono>
#include <opencv2/core/simd_intrinsics.hpp>
#include "enhance.h"
#include "common/common.h"
#include "common/apply_spline.h"


const std::vector<std::string> ImgEnhance::_OUTPUT_NAMES {"r"};

ImgEnhance::ImgEnhance(const std::string mnn_path, size_t num_threads)
            :_mnn_path(mnn_path),
            _num_threads(num_threads){

    create_mnn_env();

  _pretreat = std::shared_ptr<MNN::CV::ImageProcess>(
      MNN::CV::ImageProcess::create(
          MNN::CV::BGR,
          MNN::CV::RGB,
          _mean_vals, 3,
          _normal_vals, 3
      )
  );

}

ImgEnhance::~ImgEnhance(){
    if(_mnn_interpreter != nullptr){
        _mnn_interpreter->releaseModel();
        if(_mnn_session != nullptr){
            _mnn_interpreter->releaseSession(_mnn_session);
        }
    }
}

cv::Mat ImgEnhance::run(const cv::Mat &input_img_bgr, size_t ref_size){

    auto init_mat_time = std::chrono::system_clock::now();
    cv::Mat input_img_bgr_normal, input_img_gray_normal;
    input_img_bgr.convertTo(input_img_bgr_normal, CV_32FC3, 1.0/255.0);

    cv::Size target_size = scale_longe_edge(cv::Size(input_img_bgr_normal.cols, input_img_bgr_normal.rows), ref_size);
	cv::Mat map_img, in_img;
	cv::resize(input_img_bgr_normal, map_img, target_size, 0, 0, cv::INTER_AREA);
    cv::cvtColor(map_img, in_img, cv::ColorConversionCodes::COLOR_BGR2GRAY);
    cv::Mat nchw_img = cv::dnn::blobFromImage(in_img, 1.0, in_img.size(), cv::Scalar(), false);
    std::cout << "input_img_bgr_normal: "<< input_img_bgr_normal.rows << "," << input_img_bgr_normal.cols << "," << input_img_bgr_normal.channels() << std::endl;
    std::cout << "map_img: "<< map_img.rows << "," << map_img.cols << "," << map_img.channels() << std::endl;
    std::cout << "in_img: " << in_img.rows << "," << in_img.cols << "," << in_img.channels() << std::endl;
    std::cout << "nchw_img: " << nchw_img.rows << "," << nchw_img.cols << "," << nchw_img.channels() << "," << nchw_img.dims << std::endl;

    spend_time("init_mat_time", init_mat_time);

    auto init_calc_time = std::chrono::system_clock::now();

    int d_w = in_img.size().width;
    int d_h = in_img.size().height;
    int d_c = in_img.channels();
    assert(d_c == 1);

    std::cout << in_img.at<float>(0, 0) << "," << in_img.at<float>(d_w, d_h) << std::endl;

    MNN::Session* mnn_session = create_session();

    auto input_tensor = _mnn_interpreter->getSessionInput(mnn_session, nullptr);
    int input_batch = input_tensor->batch();
    int input_channel = input_tensor->channel();
    int input_height = input_tensor->height();
    int input_width = input_tensor->width();

    // print_input_info("resize before");
    std::vector<int> target_dims{1, d_c, d_h, d_w};
    std::vector<int> input_dims{input_batch, input_channel, input_height, input_width};
    if(input_dims != target_dims){
        _mnn_interpreter->resizeTensor(input_tensor, target_dims);
        _mnn_interpreter->resizeSession(mnn_session);
    }

    input_tensor = _mnn_interpreter->getSessionInput(mnn_session, nullptr);
    // _pretreat->convert(const_cast<unsigned char*>(in_img.data), d_w, d_h, in_img.step[0], input_tensor);
    auto nchw_tensor = new MNN::Tensor(input_tensor, input_tensor->getDimensionType());
    std::cout << nchw_tensor->height() * nchw_tensor->width() * nchw_tensor->batch() * nchw_tensor->channel() << "," << d_c*d_h*d_w << std::endl;
    std::cout << nchw_tensor->host<float>()[0] << "," << nchw_tensor->host<float>()[d_h*d_w] << std::endl;
    memccpy(nchw_tensor->host<float>(), reinterpret_cast<float*>(const_cast<unsigned char*>(nchw_img.data)), 0, 1*d_c*d_h*d_w);
    input_tensor->copyFromHostTensor(nchw_tensor);
    delete nchw_tensor;

    // input_tensor = _mnn_interpreter->getSessionInput(mnn_session, nullptr);
    std::cout << input_tensor->host<float>()[0] << "," << input_tensor->host<float>()[d_h*d_w] << std::endl;
    // print_input_info("resize after");


    _mnn_interpreter->runSession(mnn_session);
    auto output_tensors = _mnn_interpreter->getSessionOutputAll(mnn_session);

    std::vector<float> r_tensor_values = fetch_cure_param(output_tensors.at("r"));
    if(r_tensor_values.size() == 0){
        return cv::Mat();
    }

    std::cout << r_tensor_values << std::endl;

    spend_time("init_calc_time", init_calc_time);

    cv::Mat enhance_img = apply_spline_luma(input_img_bgr_normal, map_img, in_img, r_tensor_values);

    auto convert_time = std::chrono::system_clock::now();
    enhance_img.convertTo(enhance_img, CV_8UC3);
    spend_time("convert_time", convert_time);

    return enhance_img;
}


MNN::Session* ImgEnhance::create_session(){
    // 2 init schedule configt
    MNN::ScheduleConfig schedule_config;
    schedule_config.numThread = _num_threads;
    MNN::BackendConfig backend_config;
    backend_config.precision = MNN::BackendConfig::Precision_High;
    schedule_config.backendConfig = &backend_config;

    //3 create session
    return _mnn_interpreter->createSession(schedule_config);
}

void ImgEnhance::create_mnn_env(){
    // 1. init interpreter
    _mnn_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(_mnn_path.c_str()));

    // 2 init schedule configt
    MNN::ScheduleConfig schedule_config;
    schedule_config.numThread = _num_threads;
    MNN::BackendConfig backend_config;
    backend_config.precision = MNN::BackendConfig::Precision_High;
    schedule_config.backendConfig = &backend_config;

    //3 create session
    _mnn_session = _mnn_interpreter->createSession(schedule_config);

    // //4 init input tensor
    auto tmp_input_map = _mnn_interpreter->getSessionInputAll(_mnn_session);
    assert(1 == tmp_input_map.size());

    auto input_tensor = _mnn_interpreter->getSessionInput(_mnn_session, nullptr);
    // // 5. init input dims
    int input_batch = input_tensor->batch();
    int input_channel = input_tensor->channel();
    int input_height = input_tensor->height();
    int input_width = input_tensor->width();
    assert(input_channel == 1);
    int dimension_type = input_tensor->getDimensionType();
    std::cout << "the input tensor has " << input_tensor->size() << std::endl;
    if(dimension_type == MNN::Tensor::CAFFE){
        //NCHW
        // _mnn_interpreter->resizeTensor(input_tensor, {input_batch, input_channel, input_height, input_width});
        // _mnn_interpreter->resizeSession(_mnn_session);
        std::cout << "Dimension Type is CAFFE, NCHW!\n";
    }
    else if(dimension_type == MNN::Tensor::TENSORFLOW){
        //NHWC
        // _mnn_interpreter->resizeTensor(input_tensor, {input_batch, input_height, input_width, input_channel});
        // _mnn_interpreter->resizeSession(_mnn_session);
        std::cout << "Dimension Type is TENSORFLOW, NHWC!\n";
    }
    else if(dimension_type == MNN::Tensor::CAFFE_C4){
        std::cout << "Dimension Type is CAFFE_C4, skip resizeTensor & resizeSession!\n";
    }

    auto tmp_output_map = _mnn_interpreter->getSessionOutputAll(_mnn_session);

    for (auto it = tmp_output_map.cbegin(); it != tmp_output_map.cend(); ++it){
        std::cout << "Output: " << it->first << ": ";
        it->second->printShape();
        assert(std::find(_OUTPUT_NAMES.begin(), _OUTPUT_NAMES.end(), it->first) != _OUTPUT_NAMES.end());
    }
}


std::vector<float> ImgEnhance::fetch_cure_param(MNN::Tensor* tensor){
    MNN::Tensor host_tensor(tensor, tensor->getDimensionType());
    tensor->copyToHostTensor(&host_tensor);

    std::vector<float> cure_param;
    auto output_shape = host_tensor.shape();
    const float *f_data = host_tensor.host<float>();
    for(size_t i=0; i<output_shape.at(1); i++){
        cure_param.push_back(f_data[i]);
    }

    return cure_param;
}

void ImgEnhance::print_input_info(std::string flag)
{
    std::cout << "===============" << flag <<" ==============\n";
    std::map<std::string, MNN::Tensor*> input_tensors = _mnn_interpreter->getSessionInputAll(_mnn_session);
    std::cout << "the input tensor has " << input_tensors.size() << std::endl;
    for(auto it=input_tensors.begin(); it!=input_tensors.end(); it++){
        std::cout << "input name " << it->first << ": ";
        it->second->printShape();
        int dimension_type = it->second->getDimensionType();
        if (dimension_type == MNN::Tensor::CAFFE)
            std::cout << "Dimension Type: (CAFFE/PyTorch/ONNX)NCHW" << "\n";
        else if (dimension_type == MNN::Tensor::TENSORFLOW)
            std::cout << "Dimension Type: (TENSORFLOW)NHWC" << "\n";
        else if (dimension_type == MNN::Tensor::CAFFE_C4)
            std::cout << "Dimension Type: (CAFFE_C4)NC4HW4" << "\n";
    }
    std::cout << "=============== ==============\n";
}

void ImgEnhance::transfor_data(const cv::Mat &mat){
    int d_w =mat.size().width;
    int d_h = mat.size().height;
    int d_c = mat.channels();

    auto input_tensor = _mnn_interpreter->getSessionInput(_mnn_session, nullptr);
    int input_batch = input_tensor->batch();
    int input_channel = input_tensor->channel();
    int input_height = input_tensor->height();
    int input_width = input_tensor->width();
    int dimension_type = input_tensor->getDimensionType();
    print_input_info("resize before");

    std::vector<int> target_dims{1, d_c, d_h, d_w};
    std::vector<int> input_dims{input_batch, input_channel, input_height, input_width};
    if(input_dims != target_dims){
        _mnn_interpreter->resizeTensor(input_tensor, target_dims);
        _mnn_interpreter->resizeSession(_mnn_session);
    }

    input_tensor = _mnn_interpreter->getSessionInput(_mnn_session, nullptr);
    auto nhwc_tensor = new MNN::Tensor(input_tensor, input_tensor->getDimensionType());
    memccpy(nhwc_tensor->host<float>(), reinterpret_cast<float*>(const_cast<unsigned char*>(mat.data)), 0, d_c*d_h*d_w);
    input_tensor->copyFromHostTensor(nhwc_tensor);
    input_tensor = _mnn_interpreter->getSessionInput(_mnn_session, nullptr);
    std::cout << input_tensor->host<float>()[0] << "," << input_tensor->host<float>()[d_h*d_w] << std::endl;
    print_input_info("resize after");
}