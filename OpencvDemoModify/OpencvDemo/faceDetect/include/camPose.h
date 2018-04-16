#ifndef _CAM_POSE_H_
#define _CAM_POSE_H_

#include <vector>
//#include <opencv2/core.hpp> 
#include <core.hpp> 
#include <iostream>
#include <fstream>

using namespace std;

namespace bn 
{
int cam_pose_init(cv::Size frameSize_,string fname = "");

// Lastly detect 2D model shape [x1,x2,...xn,y1,...yn]
//cv::Mat_<double>		detected_landmarks;
cv::Vec6d get_pose(cv::Rect_<double>& Faces,cv::Mat_<double> &detected_landmarks,bool flag_ = false,float fx_ =-1, float fy_=-1, float cx_=-1, float cy_=-1);
void GetParam(float &fx_, float &fy_, float &cx_, float &cy_);
int cam_pose_uninit(void);

}//namespace bn

#endif //_CAM_POSE_H_
