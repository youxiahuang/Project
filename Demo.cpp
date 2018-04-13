#include <opencv2/opencv.hpp>
using namespace cv;
std::vector<cv::Rect> faces;
int main()
{
	CascadeClassifier facedetect;
	bool isSucces = facedetect.load("E:\\WorkStation\\Project\\OpencvDemoModify\\OpencvDemo\\data\\lbpcascades\\lbpcascade_frontalface_improved.xml");

	Mat frame = imread("E:\\WorkStation\\Project\\Test\\x64\\Debug\\test.jpg");
	if (frame.empty()){
		std::cout << "frame empty" << std::endl;
	}

	Mat frameGray;
	
	if (frame.channels() == 3)
	{
		cvtColor(frame, frameGray, CV_RGB2GRAY);
	}

	facedetect.detectMultiScale(frameGray, faces, 1.2, 3, 0, Size(30, 30));
	
	if (faces.size() > 0)
	{
		for (int i = 0; i < faces.size(); i++)
		{
			Rect rec = faces[i];
			rectangle(frame, Point(rec.x, rec.y), Point(rec.x + rec.width, rec.y + rec.height), Scalar(0, 255, 0), 1, 8);
			break;
		}
	}
	
	imshow("frame", frame);
	frame.release();
	frameGray.release();
	waitKey(0);
	return 0;
}
