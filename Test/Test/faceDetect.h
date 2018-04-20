#ifndef _FACE_DETECT_H_
#define _FACE_DETECT_H_
#include "ftDetection_based_tracker.h"
#include "IMDetector.h"
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
class FTDetectionBaseTracker;
class FaceDetectAdapter : public IMDetector
{
public:
	FaceDetectAdapter(cv::Ptr<CascadeClassifier> detector) : IMDetector(), Detector(detector)
	{
		CV_Assert(detector);
		count = 0;
	}

	int dt_init();
	void detect(const Mat& greyImage, vector<cv::Rect>& faces);
	void detect(const Mat& greyImage, vector<cv::Rect>& faces, bool firstFlag);
	void setOnlyOneFaceFlag(bool flag);

	virtual ~FaceDetectAdapter(){}
private:
	FaceDetectAdapter();
	cv::Ptr<CascadeClassifier> Detector;
	int count;
	int scale = 2;
	string lbpxml = "lbpcascade_frontalface_improved";		//训练集
	int minWidth = 90 / scale;								//匹配物体最小宽度
	FTDetectionBaseTracker* pDetector;						//检测器
};
#endif