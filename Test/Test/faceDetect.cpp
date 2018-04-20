#include "faceDetect.h"
#include "ftDetection_based_tracker.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void FaceDetectAdapter::detect(const cv::Mat &Image, std::vector<cv::Rect> &objects)
{
	return;
}
void FaceDetectAdapter::setOnlyOneFaceFlag(bool flag)
{

}
void FaceDetectAdapter::detect(const cv::Mat &Image, std::vector<cv::Rect> &objects, bool firstflag)
{
	if (count == 0){
		//Detector->setFirstFrameFlag(true);
		count = 1;
	}
	else if (count == 1){
		//Detector->setFirstFrameFlag(firstflag);
		count = 2;
	}

	//有cuda的时候只取最大的头的位置
	Detector->detectMultiScale(Image, objects, scaleFactor, minNeighbours, CV_HAAR_FIND_BIGGEST_OBJECT, minObjSize, maxObjSize);
}


//检测器初始化
int FaceDetectAdapter::dt_init()
{
	//分类器全路径
	string lbpFileFullPath = "data/lbpcascades/" + lbpxml + ".xml";
	cv::Ptr<CascadeClassifier> cascade = makePtr<CascadeClassifier>(lbpFileFullPath);
	cv::Ptr<IMDetector> mainDetector = makePtr<FaceDetectAdapter>(cascade);
	if (cascade->empty())
	{
		printf("Error Cannot load %s\n", lbpFileFullPath);
		return -1;
	}

	mainDetector->setMinObjectSize(cv::Size(minWidth, minWidth));

	cascade = makePtr<CascadeClassifier>(lbpFileFullPath);
	cv::Ptr<IMDetector> trackDetector = makePtr<FaceDetectAdapter>(cascade);
	if (cascade->empty())
	{
		printf("Error cannot load %s\n", lbpFileFullPath);
		return -1;
	}

	trackDetector->setMinObjectSize(cv::Size(minWidth, minWidth));

	Parameters params;
	params.maxTrackLifetime = 0;

	pDetector = new FTDetectionBaseTracker(mainDetector, trackDetector, params);

	if (!pDetector->run())
	{
		printf("Error face detect inint failed\n");
	}
}