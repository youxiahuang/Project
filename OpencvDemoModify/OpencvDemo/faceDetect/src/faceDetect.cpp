#include <iostream>
#include "bnobjdetect.hpp"
#include <stdio.h>
#include <string>
#include <vector>
#include"faceDetect.h"
  
#include "BNImgcodecs.h"

using namespace std;
using namespace cv;
namespace bn
{
class CascadeDetectorAdapter : public faceDetect::DetectionBasedTracker::IDetector
{
public:
	CascadeDetectorAdapter(cv::Ptr<faceDetect::CascadeClassifier> detector) :
		IDetector(),
		Detector(detector)
	{
		CV_Assert(detector);

		count = 0;
	}

	void detect(const cv::Mat &Image, std::vector<cv::Rect> &objects, bool firstflag)
	{
		//Detector->detectMultiScale(Image, objects, scaleFactor, minNeighbours, 0, minObjSize, maxObjSize);
		
		if (count == 0){
			Detector->setFirstFrameFlag(true);
			count = 1;
		}
		else if (count == 1){

			Detector->setFirstFrameFlag(firstflag);
			count = 2;
		}

		//有cuda的时候只取最大的头的位置
		Detector->detectMultiScale(Image, objects, scaleFactor, minNeighbours,4, minObjSize, maxObjSize);

	}

	void setOnlyOneFaceFlag(bool flag)
	{
		//wu优化 检测到人脸就停止检测
		Detector->setOnlyOneFaceFlag(flag);

	}


	virtual ~CascadeDetectorAdapter()
	{}

private:
	CascadeDetectorAdapter();
	cv::Ptr<faceDetect::CascadeClassifier> Detector;

	int count;
};


	int scale = 2;//1;
	std::string lbpXml = "lbpcascade_frontalface_improved" ;
	int minWidth = 90/scale;
	faceDetect::DetectionBasedTracker *pDetector;
	faceDetect::CascadeClassifier eyeCascade;
	bool useEyeDetect = true;
	int noEyesCount = 0;
	
int ft_init(int argc, char* argv[])
{
	if (argc >= 2) {
		if(strcmp(argv[1], "-info") == 0){
			cout << "-s  scale:1 / 2 / 3" << endl;
			cout << "-x  lbp_xml:cascade / lbpcascade_frontalface_improved / ** " << endl;
			cout << "-w  minWidth:24 ... 96 " << endl;
			cout << "-h  minHeight:24 ... 96 " << endl;
			cout << "-eye on:  off" << endl;		
			cout << "default: " << endl;
			cout << "	scale: 2" << endl;
			cout << "	lbp_xml: lbpcascade_frontalface_improved" << endl;
			cout << "	minWidth: 45" << endl;
			cout << "	minHeight: 45" << endl;
			cout << "	useEye: true" << endl;
			return -1;
		}
			
		for(int i = 1;i+1<argc;i++){
			if(strcmp(argv[i], "-s") == 0){
				scale = atoi(argv[++i]);
				minWidth = 90/scale;
				cout << "scale:" << scale << endl;
			}
			else if(strcmp(argv[i], "-w") == 0){
				minWidth = atoi(argv[++i]);
				cout << "minWidth:" << minWidth << endl;
			}
			else if(strcmp(argv[i], "-h") == 0){
				minWidth = atoi(argv[++i]);
				cout << "minHeight:" << minWidth << endl;
			}
			else if(strcmp(argv[i], "-x") == 0){
				lbpXml = argv[++i];
				cout << "lbpXml:" << lbpXml << endl;
			}
			else if(strcmp(argv[i], "-eye") == 0){
				string strTmp = argv[++i] ;
			    useEyeDetect = (strTmp== "off")?false:true;
				cout << "useEyeDetect:" << useEyeDetect << endl;
			}
		}
	}
	
	//分类器路径
	std::string cascadeFrontalfilename = "data/lbpcascades/"+lbpXml + ".xml";
	
	//实例化分类器
	cv::Ptr<faceDetect::CascadeClassifier> cascade = makePtr<faceDetect::CascadeClassifier>(cascadeFrontalfilename);
	cv::Ptr<faceDetect::DetectionBasedTracker::IDetector> MainDetector = makePtr<CascadeDetectorAdapter>(cascade);
	if (cascade->empty())
	{
		printf("Error: Cannot load %s\n", cascadeFrontalfilename.c_str());
		return 2;
	}
	
	//MainDetector->setMinNeighbours(3);
	//MainDetector->setScaleFactor(1.1f);
	//MainDetector->setMinObjectSize(cv::Size(24,24));
	MainDetector->setMinObjectSize(cv::Size(minWidth,minWidth));
	MainDetector->setOnlyOneFaceFlag(true);

	cascade = makePtr<faceDetect::CascadeClassifier>(cascadeFrontalfilename);
	cv::Ptr<faceDetect::DetectionBasedTracker::IDetector> TrackingDetector = makePtr<CascadeDetectorAdapter>(cascade);
	if (cascade->empty())
	{
		printf("Error: Cannot load %s\n", cascadeFrontalfilename.c_str());
		return 2;
	}
	//TrackingDetector->setMinNeighbours(3);
	//TrackingDetector->setScaleFactor(1.1f);
	//TrackingDetector->setMinObjectSize(cv::Size(24, 24));
	TrackingDetector->setMinObjectSize(cv::Size(minWidth,minWidth));
	TrackingDetector->setOnlyOneFaceFlag(true);

	faceDetect::DetectionBasedTracker::Parameters params;
	//faceDetect::DetectionBasedTracker Detector(MainDetector, TrackingDetector, params);
	params.maxTrackLifetime = 0;
	//params.minDetectionPeriod = 1000;
	pDetector = new faceDetect::DetectionBasedTracker(MainDetector, TrackingDetector, params);
	
	//if (!Detector.run())
	if (!pDetector->run())
	{
		printf("Error: Detector initialization failed\n");
		return -1;
	}
	
	if(useEyeDetect == true){
		eyeCascade.load("data/haarcascades/haarcascade_eye.xml");
		//eyeCascade.load("data/haarcascades/haarcascade_eye_tree_eyeglasses.xml");
		
		if(eyeCascade.empty()){
			useEyeDetect = false;
		}
	}
}

cv::Rect m_prefFace = {0,0,0,0};
int ft_process(const cv::Mat& frame_,std::vector<cv::Rect>& faces_)
{
	if(frame_.empty()){
		printf("error::ft_process frame_ is empty!\n");
		return -1;
	}
	
	if(frame_.channels() != 3&& frame_.channels() != 1){
		printf("error::ft_process frame_ required rgb|bgr|gray!\n");
		return -1;
	}
	
	Mat GrayFrame;
	if(frame_.channels() == 3){
		cv::cvtColor(frame_, GrayFrame, COLOR_BGR2GRAY);
		//ImgProc::cvtColor(frame_, GrayFrame, COLOR_BGR2GRAY);
	}else{
		frame_.copyTo(GrayFrame);
	}
	
	cv::resize(GrayFrame, GrayFrame, cv::Size(frame_.cols/scale,frame_.rows/scale));
	double freq = getTickFrequency();
	int64 process_start = getTickCount();
	pDetector->process(GrayFrame);
	printf("Detector process Time = %.4fms\n", ((double)(getTickCount()-process_start))/freq * 1000.0);
	pDetector->getObjects(faces_);
	for (size_t i = 0; i < faces_.size(); i++)
	{	
		m_prefFace = faces_[i];
		double factorX = (lbpXml == "lbpcascade_frontalface")?0:0.05;
		double factorY = (lbpXml == "lbpcascade_frontalface")?0:0.15;
		faces_[i].x = faces_[i].x-faces_[i].width * factorX;
		faces_[i].y = faces_[i].y-faces_[i].height * factorY;
		faces_[i].width = faces_[i].width * (1 + factorX * 2);
		faces_[i].height = faces_[i].height * (1 + factorY * 2);	
		faces_[i].x*=scale;
		faces_[i].y*=scale;
		faces_[i].width *= scale;
		faces_[i].height *= scale;
					
		if(faces_[i].x < 0){
			faces_[i].width += faces_[i].x;
			faces_[i].x = 0;
		}else if(faces_[i].x + faces_[i].width >= frame_.cols){
			faces_[i].width = frame_.cols - 1 - faces_[i].x;
		}
		if(faces_[i].y < 0){
			faces_[i].height += faces_[i].y;
			faces_[i].y = 0;
		}else if(faces_[i].y + faces_[i].height >= frame_.rows){
			faces_[i].height = frame_.rows - 1 - faces_[i].y;
		}
		
		if(useEyeDetect == true){
			std::vector<cv::Rect> eyes_;
			cv::Rect roi(m_prefFace.x,m_prefFace.y,m_prefFace.width,m_prefFace.height);
			if(roi.x<0){
				roi.width += roi.x ;
				roi.x = 0;
			}else if(roi.x+roi.width >= GrayFrame.cols){
				roi.width = GrayFrame.cols-1-roi.x;
			}
			if(roi.y<0){
				roi.height += roi.y ;
				roi.y = 0;
			}else if(roi.y+roi.height >= GrayFrame.rows){
				roi.height = GrayFrame.rows-1-roi.y;
			}
			eyeCascade.detectMultiScale(GrayFrame(roi), eyes_, 1.1, 2,4, cv::Size(), cv::Size());
			printf("%%%%%%%%%%% eyes_ (%d )\n",eyes_.size());
			
			if(eyes_.size() < 1){
				noEyesCount ++;
				if(noEyesCount >= 6){
			    	faces_.clear();
			    }
			}else{
				noEyesCount = 0;
			}
		}
	}
}

int ft_uninit(void){
	pDetector->stop();
	return 0;
}
}
