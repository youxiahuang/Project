#ifndef IMDETECTOR_H_
#define IMDETECTOR_H_
#include <opencv2/opencv.hpp>
using namespace cv;

class IMDetector : public DetectionBasedTracker::IDetector
{
public:
	virtual void detect(const Mat& image, std::vector<Rect>& objects, bool firstflag = true) = 0;
	virtual void setOnlyOneFaceFlag(bool flag) = 0;
};
#endif