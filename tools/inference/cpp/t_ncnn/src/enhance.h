#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <ncnn/net.h>
#include <ncnn/layer.h>



class ImgEnhance{
    public:
        ImgEnhance(const std::string param_path, const std::string bin_path, size_t down_scale=16, bool map_point_wise=false, size_t num_threads=1);

        ~ImgEnhance();

        cv::Mat run(const cv::Mat &img_bgr_normalized);

    private:
        void create_ncnn_env();
                  
        std::vector<float> fetch_cure_param(const ncnn::Mat &mat);

        void print_input_info(std::string title);

    private:
        std::string _param_path;
        std::string _bin_path;
        size_t _down_scale;
        bool _map_point_wise;
        size_t _num_threads;
        size_t _output_shape;
        std::string _input_name;
        std::shared_ptr<ncnn::Net> _net;

        static const std::vector<std::string> _OUTPUT_NAMES;
        
};
