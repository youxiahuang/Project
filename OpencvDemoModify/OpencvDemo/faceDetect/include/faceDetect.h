#ifndef _FACE_DETECT_H_
#define _FACE_DETECT_H_

#include <vector>
//#include <opencv.hpp>
#include <opencv2/core.hpp> 

namespace bn 
{
int ft_init(int argc, char* argv[]);
int ft_process(const cv::Mat&     frame_,
			std::vector<cv::Rect>& faces_);
int ft_uninit(void);

}//namespace bn

#endif //_FACE_DETECT_H_
