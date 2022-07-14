#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <tnn/core/mat.h>
#include <tnn/core/common.h>
#include <tnn/core/tnn.h>



class ImgEnhance{
    public:
        ImgEnhance(const std::string proto_path, const std::string model_path, size_t down_scale=16, bool map_point_wise=false, size_t num_threads=1);

        ~ImgEnhance();

        cv::Mat run(const cv::Mat &img_bgr_normalized);

    private:
        void create_tnn_env();
                  
        std::vector<float> fetch_cure_param(std::string cure_name);

        void print_input_info(std::string title);

    private:
        std::string _proto_path;
        std::string _model_path;
        size_t _down_scale;
        bool _map_point_wise;
        size_t _num_threads;
        std::shared_ptr<tnn::TNN> _net;
        std::shared_ptr<tnn::Instance> _instance;
        tnn::DeviceType _output_device_type;
        tnn::DeviceType _input_device_type;
        tnn::DeviceType _network_device_type;
        std::map<std::string, tnn::DimsVector> _output_shapes;
        std::string _input_name;

        std::vector<float> _scale_vals = {1.f, 1.f, 1.f};
        std::vector<float> _bias_vals = {0.f, 0.f, 0.f};

        static const std::vector<std::string> _OUTPUT_NAMES;
        
};
