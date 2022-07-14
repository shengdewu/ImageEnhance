#include <iostream>
#include <vector>
#include <opencv2/dnn/dnn.hpp>
#include <time.h>
#include <chrono>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/simd_intrinsics.hpp>
#include "enhance.h"
#include "common/common.h"
#include "common/apply_spline.h"


ImgEnhance::ImgEnhance(const std::string model_path, size_t ref_size, bool map_point_wise)
            :_model_path(model_path),
            _ref_size(ref_size),
            _map_point_wise(map_point_wise),
            _env(nullptr),
            _session(nullptr){                                                     
    
    create_onnx_env();
}

ImgEnhance::~ImgEnhance(){
    if(_session != nullptr){
        delete _session;
        
    }

    if(_env != nullptr){
        delete _env;
    }
}

cv::Mat ImgEnhance::run(const cv::Mat &input_img_bgr){

    auto init_mat_time = std::chrono::system_clock::now();
    cv::Mat input_img_bgr_normal;
    input_img_bgr.convertTo(input_img_bgr_normal, CV_32FC3, 1.0/255.0);

    //resize
    // cv::Size target_size = scale_longe_edge(cv::Size(input_img_bgr_normal.cols, input_img_bgr_normal.rows), _ref_size);
    cv::Size target_size(_input_dims[3], _input_dims[2]);
	cv::Mat map_img;
	cv::resize(input_img_bgr_normal, map_img, target_size, 0, 0, cv::INTER_AREA);
    std::cout << input_img_bgr_normal.cols << "," << input_img_bgr_normal.rows << std::endl;
    std::cout << _ref_size << std::endl;
    std::cout << target_size.width << "," << target_size.height << std::endl;
    std::cout << map_img.cols << "," << map_img.rows << std::endl;

    // hwc -> nchw BGR->RGB
	cv::Mat in_img = cv::dnn::blobFromImage(map_img, 1.0, map_img.size(), cv::Scalar(), true);

    // spend_time("init_mat_time", init_mat_time);

    auto init_calc_time = std::chrono::system_clock::now();
    std::vector<int64_t> input_dims {1,  map_img.channels(), map_img.rows, map_img.cols};
    size_t input_tensor_size = vector_product(input_dims);
    std::vector<float> input_tensor_values(input_tensor_size);
    input_tensor_values.assign(in_img.begin<float>(), in_img.end<float>());
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<Ort::Value> input_tensors;
    std::vector<const char*> input_names{_session->GetInputName(0, allocator)};
    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, 
                                                            input_tensor_values.data(), 
                                                            input_tensor_size, 
                                                            input_dims.data(),
                                                            input_dims.size()));                                                          

    auto output_tensors = _session->Run(Ort::RunOptions{nullptr}, 
                                        input_names.data(), 
                                        input_tensors.data(), 
                                        input_names.size(), 
                                        _output_names.data(), 
                                        _output_names.size());
	// get parameters
    assert(output_tensors.size() == 3);  
    float* r_ptr = output_tensors[0].GetTensorMutableData<float>();
    float* g_ptr = output_tensors[1].GetTensorMutableData<float>();
    float* b_ptr = output_tensors[2].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    assert(output_shape.size() == 2);
    std::vector<float> r_tensor_values(output_shape.at(1));
    std::vector<float> g_tensor_values(output_shape.at(1));
    std::vector<float> b_tensor_values(output_shape.at(1)); 
    r_tensor_values.assign(r_ptr, r_ptr+output_shape.at(1));
    g_tensor_values.assign(g_ptr, g_ptr+output_shape.at(1));
    b_tensor_values.assign(b_ptr, b_ptr+output_shape.at(1));


    // spend_time("init_calc_time", init_calc_time);

    auto map_time = std::chrono::system_clock::now();
    cv::Mat enhance_img;
    if(_map_point_wise){
        enhance_img = apply_spline(input_img_bgr_normal, r_tensor_values, g_tensor_values, b_tensor_values);
    }
    else{
        enhance_img = apply_spline(map_img, input_img_bgr_normal, r_tensor_values, g_tensor_values, b_tensor_values);
    }

    // spend_time("map_time", map_time);

    auto convert_time = std::chrono::system_clock::now();
    enhance_img.convertTo(enhance_img, CV_8UC3);
    // spend_time("convert_time", convert_time);

    return enhance_img;
}

void ImgEnhance::create_onnx_env(){
    _env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "enhance");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    _session = new Ort::Session(*_env, _model_path.c_str(), session_options);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions allocator;

    const char* r_name = _session->GetOutputName(0, allocator);
    const char* g_name = _session->GetOutputName(1, allocator);
    const char* b_name = _session->GetOutputName(2, allocator);

    Ort::TypeInfo output_type_info = _session->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType output_type = output_tensor_info.GetElementType();
    std::vector<int64_t> output_dims = output_tensor_info.GetShape();

    _output_names.push_back(r_name);
    _output_names.push_back(g_name);
    _output_names.push_back(b_name);

    Ort::TypeInfo input_type_info = _session->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    _input_dims = input_tensor_info.GetShape();

    std::cout << "onnx input dims = " << _input_dims << std::endl;
}


