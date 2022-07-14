#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <numeric>
#include <time.h>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <opencv2/core/simd_intrinsics.hpp>
#include "enhance.h"
#include "common/common.h"
#include "common/apply_spline.h"


// Helper functions
std::string content_buffer_from(std::string path) {
    std::ifstream file(path, std::ios::binary);
    if (file.is_open()) {
        file.seekg(0, file.end);
        int size      = file.tellg();
        char* content = new char[size];
        file.seekg(0, file.beg);
        file.read(content, size);
        std::string file_content;
        file_content.assign(content, size);
        delete[] content;
        file.close();
        return file_content;
    } else {
        return "";
    }
}


const std::vector<std::string> ImgEnhance::_OUTPUT_NAMES {"r", "g", "b"};

ImgEnhance::ImgEnhance(const std::string proto_path, const std::string model_path, size_t down_scale, bool map_point_wise, size_t num_threads)
            :_proto_path(proto_path),
            _model_path(model_path),
            _down_scale(down_scale),
            _map_point_wise(map_point_wise),
            _num_threads(num_threads),
            _net(nullptr),
            _instance(nullptr){                                                     
    
    create_tnn_env();
}

ImgEnhance::~ImgEnhance(){
    if(_net != nullptr){
        _net->DeInit();
    }
}

cv::Mat ImgEnhance::run(const cv::Mat &img_bgr_normalized){

    auto init_mat_time = std::chrono::system_clock::now();
    
    cv::Mat input_img_bgr_normalized;
    cv::resize(img_bgr_normalized, input_img_bgr_normalized, cv::Size(), 1.0/_down_scale, 1.0/_down_scale);
  
    cv::Mat input_img_rgb_normalized;
    cv::cvtColor(input_img_bgr_normalized, input_img_rgb_normalized, cv::ColorConversionCodes::COLOR_BGR2RGB);

    int d_w =input_img_rgb_normalized.size().width;
    int d_h = input_img_rgb_normalized.size().height;

    spend_time("init_mat_time", init_mat_time);

    auto init_calc_time = std::chrono::system_clock::now();
    
    tnn::DimsVector target_dims = {1, 3, d_h, d_w};
    
    // print_input_info("reshape befor");

    std::shared_ptr<tnn::Mat> input_mat = std::make_shared<tnn::Mat>(_input_device_type, tnn::NCHW_FLOAT, target_dims, static_cast<void*>(input_img_rgb_normalized.data));
    if(nullptr == input_mat->GetData()){
        std::cout << "input_mat == nullptr! create4 input mat failed";
        return cv::Mat();
    }

    tnn::Status status;
    tnn::InputShapesMap reshape_input_shape{{_input_name, target_dims}};
    status = _instance->Reshape(reshape_input_shape);
    if(status != tnn::TNN_OK){
        std::cout << status.description() << std::endl;
        return cv::Mat(); 
    }

    tnn::MatConvertParam input_cvt_param;
    input_cvt_param.scale = _scale_vals;
    input_cvt_param.bias  = _bias_vals;
    status = _instance->SetInputMat(input_mat, input_cvt_param);
    if(status != tnn::TNN_OK){
        std::cout << status.description() << std::endl;
        return cv::Mat(); 
    }

    // print_input_info("reshape after");

    status = _instance->Forward();
    if(status != tnn::TNN_OK){
        std::cout << status.description() << std::endl;
        return cv::Mat(); 
    }

    std::vector<float> r_tensor_values = fetch_cure_param("r");
    std::vector<float> g_tensor_values = fetch_cure_param("g");
    std::vector<float> b_tensor_values = fetch_cure_param("b");
    if(r_tensor_values.size() == 0 || g_tensor_values.size() == 0 || b_tensor_values.size() == 0 ){
        return cv::Mat();
    }

    spend_time("init_calc_time", init_calc_time);

    if(_map_point_wise){
        return apply_spline(img_bgr_normalized, r_tensor_values, g_tensor_values, b_tensor_values);
    }
    else{
        return apply_spline(input_img_bgr_normalized, img_bgr_normalized, r_tensor_values, g_tensor_values, b_tensor_values);
    }
}


void ImgEnhance::create_tnn_env(){
    auto proto_content_buffer = content_buffer_from(_proto_path.c_str());
    auto model_content_buffer = content_buffer_from(_model_path.c_str());

    tnn::Status status;

    // create TNN
    tnn::ModelConfig model_config;
    model_config.model_type = tnn::MODEL_TYPE_TNN;
    model_config.params = {proto_content_buffer, model_content_buffer};
    _net = std::make_shared<tnn::TNN>();
    status = _net->Init(model_config);
    if(status != tnn::TNN_OK || !_net){
        std::cout << "tnn init is failed: "<< status.description() << std::endl;
        return;
    }

    //create Instance
    tnn::NetworkConfig network_config;
    _output_device_type = tnn::DEVICE_X86;
    _input_device_type = tnn::DEVICE_X86;
    _network_device_type = tnn::DEVICE_X86;
    network_config.device_type = _network_device_type;
    network_config.precision = tnn::PRECISION_AUTO;
    network_config.network_type = tnn::NETWORK_TYPE_AUTO;
    _instance = _net->CreateInst(network_config, status);
    if(status != tnn::TNN_OK || !_instance){
        std::cout << "tnn create instance is failed: "<< status.description() << std::endl;
        return;
    }

    // other
    if(_num_threads > 1){
        _instance->SetCpuNumThreads(_num_threads);
    }
    std::vector<std::string> input_names;
    tnn::BlobMap iblob_map;
    
    _instance->GetAllInputBlobs(iblob_map);
    for(const auto &item: iblob_map){
        input_names.push_back(item.first);
    }
    tnn::BlobDesc i_blob_desc = iblob_map.begin()->second->GetBlobDesc();
    tnn::DataFormat i_format = i_blob_desc.data_format;
    tnn::DataType i_type = i_blob_desc.data_type;
    tnn::DimsVector input_shape =  i_blob_desc.dims;
    assert(input_names.size() == 1);
    _input_name = input_names.front();

    // if(i_format == tnn::DATA_FORMAT_NHWC){
        // _input_batch = input_shape.at(0);
        // _input_height = input_shape.at(1);
        // _input_width = input_shape.at(2);
        // _input_channel = input_shape.at(3);
    // }
    // else if(i_format == tnn::DATA_FORMAT_NCHW){
        // _input_batch = input_shape.at(0);
        // _input_channel = input_shape.at(1);
        // _input_height = input_shape.at(2);
        // _input_width = input_shape.at(3); 
    // }
    // else{
    //     std::cout << "dont support for other: " << i_format << std::endl;
    //     return;
    // }

    // _input_value_size = _input_batch * _input_channel * _input_height * _input_width;

    std::cout << "input_names: " << input_names << std::endl;
    std::cout << "input format: " << i_format << std::endl;
    std::cout << "input type: "  << i_type << std::endl;
    std::cout << "input dims: "  << input_shape << std::endl;

    tnn::BlobMap oblob_map;
    std::vector<std::string> output_names;
    _instance->GetAllOutputBlobs(oblob_map);
    for(const auto &item: oblob_map){
        output_names.push_back(item.first);
        _output_shapes[item.first] = item.second->GetBlobDesc().dims;
        assert(std::find(_OUTPUT_NAMES.begin(), _OUTPUT_NAMES.end(), item.first) != _OUTPUT_NAMES.end());
    }

    tnn::BlobDesc o_blob_desc = oblob_map.begin()->second->GetBlobDesc();
    tnn::DataFormat o_format = o_blob_desc.data_format;
    tnn::DataType o_type = o_blob_desc.data_type;
    tnn::DimsVector o_dims =  o_blob_desc.dims;
    std::cout << "output_names: " << output_names << std::endl;
    std::cout << "output format: "  << o_format << std::endl;
    std::cout << "output type: "  << o_type << std::endl;
    std::cout << "output dims: "  << o_dims << std::endl;
    
}

std::vector<float> ImgEnhance::fetch_cure_param(std::string cure_name){

    std::vector<float> cure_param;

    tnn::MatConvertParam cvt_param;
    std::shared_ptr<tnn::Mat> cure_mat;
    tnn::Status status = _instance->GetOutputMat(cure_mat, cvt_param, cure_name, _output_device_type);
    if(status != tnn::TNN_OK){
        std::cout << status.description() << std::endl;
        return cure_param; 
    }

    tnn::DimsVector output_shape = _output_shapes[cure_name];

    float *f_data = static_cast<float*>(cure_mat->GetData());

    for(size_t i=0; i<output_shape.at(1); i++){
        cure_param.push_back(f_data[i]);
    }

    return cure_param;
}   

void ImgEnhance::print_input_info(std::string title){
    if(_instance == nullptr){
        return;
    }
    std::cout << "====== " << title << " ======"  << std::endl;
    tnn::BlobMap iblob_map;
    _instance->GetAllInputBlobs(iblob_map);
    for(const auto &item: iblob_map){
        std::string name = item.first;
        tnn::BlobDesc blob_desc = item.second->GetBlobDesc();
        std::cout << "input name  : " << name << std::endl;
        std::cout << "input format: " << blob_desc.data_format << std::endl;
        std::cout << "input type  : " << blob_desc.data_type << std::endl;
        std::cout << "input shape : " << blob_desc.dims << std::endl;

    }
}