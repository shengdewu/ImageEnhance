#include <iostream>
#include <assert.h>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "auto_exposure.hpp"


int main(int argc, char** argv){

    std::string model_path = "/mnt/sda1/wokspace/ImageCureEnhance/inference/model/spline_att_model_1355.onnx";

    AutoExposure auto_exposure(model_path);
    
    std::string img_path = "/mnt/sdb/data.set/xintu.data/转档测评/20210510转档评测_tif_3000x2000/婚纱/鲜艳纯底/婚纱CS (50)/IMG_6929.tif";
    // std::string img_path = "/mnt/sdb/data.set/xintu.data/转档测评/20210510转档评测_tif/儿童/01浅色实景/儿童SJ (31)/16108206392879900290.tif";
    cv::Mat img_bgr = cv::imread(img_path, cv::ImreadModes::IMREAD_COLOR);

    // cv::resize(img_bgr, img_bgr, cv::Size(), 0.125, 0.125, cv::INTER_AREA);

    cv::Mat enhance_img = auto_exposure.exposure(img_bgr);

    cv::Mat concat;
    cv::hconcat(img_bgr, enhance_img, concat);
    size_t spos = img_path.rfind('/')+1;
    size_t epos = img_path.find(".tif");
    cv::imwrite(img_path.substr(spos, epos-spos)+".jpg", concat);

}