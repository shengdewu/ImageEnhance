#pragma once
#include <numeric>
#include <vector>
#include <iostream>
#include <time.h>
#include <iomanip>
#include <chrono>

template <typename T>
T vector_product(const std::vector<T>& v)
{
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<int64_t>());
}


template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

inline cv::Size scale_longe_edge(cv::Size size, size_t ref_size){
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

inline void spend_time(std::string title, std::chrono::system_clock::time_point start_time){
    std::cout << title <<":";
    std::cout << std::setw(9);
    std::cout << (static_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - start_time)).count();
    std::cout << std::endl;
}
