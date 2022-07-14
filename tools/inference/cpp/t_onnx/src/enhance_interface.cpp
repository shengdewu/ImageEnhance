#include "enhance.h"


static ImgEnhance *img_enhance_model = nullptr;

void enahnce_init_model(const std::string model_path, size_t down_scale=16, bool map_point_wise=false){
    img_enhance_model = new ImgEnhance(model_path, down_scale, map_point_wise);
}

cv::Mat enahnce_run(const cv::Mat &img_bgr){
    return img_enhance_model->run(img_bgr);
}

void enahnce_release_model(){
    if(img_enhance_model != nullptr){
        delete img_enhance_model;
    }
}