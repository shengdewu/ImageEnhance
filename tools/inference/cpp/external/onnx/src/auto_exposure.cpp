//
// Created by ts on 2022/1/21.
//
#include <iostream>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "auto_exposure.hpp"

cv::Mat AutoExposure::exposure(const cv::Mat &img_bgr)
{
	//----------------------------------------------------------------------------------//
	// normalize to 0 to 1.0	
    cv::Mat img_bgr_normal;
    img_bgr.convertTo(img_bgr_normal, CV_32FC3, 1.0/255.0);

    // resize
    // cv::Size target_size = scale_longe_edge(cv::Size(img_bgr_normal.cols, img_bgr_normal.rows), ref_size);
    std::vector<int64_t> input_dims = ort_->input_dims_[0];
    cv::Size target_size(input_dims[3], input_dims[2]);
	cv::Mat map_img;
	cv::resize(img_bgr_normal, map_img, target_size, 0, 0, cv::INTER_AREA);
    
    std::cout << input_dims << std::endl;
    std::cout << img_bgr_normal.cols << "," << img_bgr_normal.rows << std::endl;
    std::cout << map_img.cols << "," << map_img.rows << std::endl;

    // HWC->NCHW BGT->RGB
	cv::Mat in_img = cv::dnn::blobFromImage(map_img, 1.0, map_img.size(), cv::Scalar(), true);
    std::cout << in_img.dims << std::endl;

	// create input tensor
	// std::vector<int64_t> input_dims = {1, 3, target_size.height, target_size.width};
	size_t inputTensorSize = input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3];
	std::vector<float> inputTensorValues(inputTensorSize);
	inputTensorValues.assign(in_img.begin<float>(), in_img.end<float>());
	std::vector<Ort::Value> input_tensors;
	input_tensors.emplace_back(Ort::Value::CreateTensor<float>(ort_->memory_info_,
															   inputTensorValues.data(),
															   inputTensorSize,
															   input_dims.data(),
															   input_dims.size()));
	//----------------------------------------------------------------------------------//
	// run
	auto output_tensors = ort_->session_->Run(Ort::RunOptions {nullptr},
											  ort_->input_names_.data(),
											  input_tensors.data(),
											  ort_->input_count_,
											  ort_->output_names_.data(),
											  ort_->output_count_);
	//----------------------------------------------------------------------------------//
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

    //map image
    cv::Mat enhance_img = apply_spline(map_img, img_bgr_normal, r_tensor_values, g_tensor_values, b_tensor_values);
    enhance_img.convertTo(enhance_img, CV_8UC3);
    return enhance_img;
}

cv::Mat AutoExposure::apply_spline(const cv::Mat &img_bgr_map, const cv::Mat &img_bgr,
                                   const std::vector<float> &r_tensor_values, 
                                   const std::vector<float> &g_tensor_values, 
                                   const std::vector<float> &b_tensor_values){

    cv::Mat b_scale_map = cv::Mat(img_bgr_map.rows, img_bgr_map.cols, CV_32F, b_tensor_values[0]);
    cv::Mat g_scale_map = cv::Mat(img_bgr_map.rows, img_bgr_map.cols, CV_32F, g_tensor_values[0]);
    cv::Mat r_scale_map = cv::Mat(img_bgr_map.rows, img_bgr_map.cols, CV_32F, r_tensor_values[0]);
    
    auto cure_steps = r_tensor_values.size() - 1;

    std::vector<float> b_slope = std::vector<float>(cure_steps);
    std::vector<float> g_slope = std::vector<float>(cure_steps);
    std::vector<float> r_slope = std::vector<float>(cure_steps);

    for(int i=0; i<cure_steps-1; i++){
        b_slope[i] = b_tensor_values[i+1] - b_tensor_values[i];
        g_slope[i] = g_tensor_values[i+1] - g_tensor_values[i];
        r_slope[i] = r_tensor_values[i+1] - r_tensor_values[i];
    }

    // #pragma omp parallel
    {
        // #pragma omp for
        for(size_t nrow=0; nrow < img_bgr_map.rows; nrow++){
            
            const float* img_fptr = img_bgr_map.ptr<float>(nrow);
            float* b_scale_ptr = b_scale_map.ptr<float>(nrow);
            float* g_scale_ptr = g_scale_map.ptr<float>(nrow);
            float* r_scale_ptr = r_scale_map.ptr<float>(nrow);

            for(size_t ncol = 0; ncol < img_bgr_map.cols; ncol++){
                
                float b_img_cure = img_fptr[0] * static_cast<float>(cure_steps);
                float g_img_cure = img_fptr[1] * static_cast<float>(cure_steps);
                float r_img_cure = img_fptr[2] * static_cast<float>(cure_steps);

       
                for(int i=0; i<cure_steps-1; ++i){                   
                    b_scale_ptr[ncol] = b_scale_ptr[ncol] + b_slope[i] * (b_img_cure - static_cast<float>(i));
                    g_scale_ptr[ncol] = g_scale_ptr[ncol] + g_slope[i] * (g_img_cure - static_cast<float>(i));
                    r_scale_ptr[ncol] = r_scale_ptr[ncol] + r_slope[i] * (r_img_cure - static_cast<float>(i));   
                                                                        
                }

                img_fptr += 3;
            }          
        }
    }

    std::vector<cv::Mat> scale_map{b_scale_map, g_scale_map, r_scale_map};
    cv::Mat bgr_scale_map;
    cv::merge(scale_map, bgr_scale_map);

    cv::Mat bgr_scale; 
    cv::resize(bgr_scale_map, bgr_scale, cv::Size(img_bgr.cols, img_bgr.rows));

    return img_bgr.mul(bgr_scale)*255.0;
}  

cv::Size AutoExposure::scale_longe_edge(cv::Size size, size_t ref_size){
    int target_width = size.width;
    int target_height = size.height;
    int max_size = std::max(target_width, target_height);
    float scale = ref_size * 1.0 / max_size;
    if(scale < 1.0){
        if(target_width > target_height){
            target_width = ref_size;
            target_height = target_height * scale;
        }
        else{
            target_width = target_width * scale;
            target_height = ref_size;
        }
    }

    return cv::Size (int(target_width+0.5), int(target_height+0.5));
} 