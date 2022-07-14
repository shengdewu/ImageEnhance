#include <iostream>
#include <assert.h>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <numeric>
#include <time.h>
#include <chrono>
#include <iomanip>
#include <opencv2/core/simd_intrinsics.hpp>
#include "enhance.h"
#include "common/common.h"


int main(int argc, char** argv){
    
    std::string param_path = "/mnt/sda1/wokspace/ImageCureEnhance/inference/model/spline_att_model_1355.ncnn.param";
    std::string bin_path = "/mnt/sda1/wokspace/ImageCureEnhance/inference/model/spline_att_model_1355.ncnn.bin";

    auto init_time = std::chrono::system_clock::now();
    size_t down_scale = atoi(argv[1]);
    size_t num_threads = atoi(argv[2]);

    ImgEnhance img_enhance(param_path, bin_path, down_scale, false, num_threads);

    std::vector<std::string> img_path;
    img_path.push_back("/mnt/sdb/data.set/xintu.data/转档测评/20210510转档评测_tif/儿童/01浅色实景/儿童SJ (31)/16108206392879900290.tif");
    img_path.push_back("/mnt/sdb/data.set/xintu.data/转档测评/20210510转档评测_tif/儿童/01浅色实景/儿童SJ (31)/161082062385739907252.tif");
    
    // for(size_t i=0; i < img_path.size(); i++){
    //     auto read_time = std::chrono::system_clock::now();
    //     cv::Mat img_bgr = cv::imread(img_path[i], cv::ImreadModes::IMREAD_UNCHANGED);
    //     spend_time("read img", read_time);

    //     cv::Mat img_bgr_normal;
    //     img_bgr.convertTo(img_bgr_normal, CV_32FC3, 1.0/65535.0);

    //     auto en_time = std::chrono::system_clock::now();

    //     cv::Mat enhance_img = img_enhance.run(img_bgr_normal);
    //     spend_time("en_time", en_time); 

    //     auto convert_time = std::chrono::system_clock::now();
    //     cv::Mat enhance_255 =(enhance_img*255);
    //     cv::Mat enhance_8u3c;
    //     enhance_255.convertTo(enhance_8u3c, CV_8UC3);
    //     cv::Mat img_bgr_255 = (img_bgr_normal*255);
    //     cv::Mat img_bgr_8uc3;
    //     img_bgr_255.convertTo(img_bgr_8uc3, CV_8UC3);
    //     spend_time("convert time ", convert_time);
    //     cv::Mat concat;
    //     cv::hconcat(img_bgr_8uc3, enhance_8u3c, concat);
    //     size_t pos = img_path[i].rfind('/')+1;
    //     cv::imwrite(img_path[i].substr(pos)+".jpg", concat);
    // }

}