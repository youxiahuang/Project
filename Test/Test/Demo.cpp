#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
void Test();
void TestVector();
void PictureDemo();
void VideoDemo();

int main()
{
	/*CascadeClassifier facedetect;
	bool isSucces = facedetect.load("E:\\WorkStation\\Project\\OpencvDemoModify\\OpencvDemo\\data\\lbpcascades\\lbpcascade_frontalface_improved.xml");
	VideoCapture cap("E:\\WorkStation\\Project\\OpencvDemo\\OpencvDemo\\out.avi");
	if (!cap.isOpened()){
		std::cout << "open failed" << std::endl;
		return 0;
	}

	Mat frame;
	std::vector<cv::Rect> faces;
	while (true){
		cap >> frame;
		if (frame.empty()){
			std::cout << "frame empty" << std::endl;
			break;
		}
		Mat frameGray;
		std::vector<cv::Rect> faces;

		if (frame.channels() == 3)
		{
			cvtColor(frame, frameGray, CV_RGB2GRAY);
		}

		facedetect.detectMultiScale(frameGray, faces, 1.2, 3, 0, Size(30, 30));

		for (int i = 0; i < faces.size(); i++){
			Point c;
			c.x = faces[i].x + faces[i].width / 2;
			c.y = faces[i].y + faces[i].height / 2;
			int r = faces[i].width / 2;
			cv::circle(frame, c, r, Scalar(255, 0, 0), 3, 8);
			std::string pyr("pyr:: ");
			break;
		}
		imshow("frame", frame);
		uchar c = waitKey(5);
		if (c == 'q')
		{
			break;
		}
	}*/
	VideoDemo();
	waitKey(0);
	return 0;
}

void PictureDemo()
{
	//TestVector();
	CascadeClassifier facedetect;
	bool isSucces = facedetect.load("E:\\WorkStation\\Project\\OpencvDemoModify\\OpencvDemo\\data\\lbpcascades\\lbpcascade_frontalface_improved.xml");

	Mat frame = imread("E:\\WorkStation\\Project\\Test\\x64\\Debug\\test.jpg");
	if (frame.empty()){
		std::cout << "frame empty" << std::endl;
	}

	Mat frameGray;
	std::vector<cv::Rect> faces;

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
}

void VideoDemo()
{
	CascadeClassifier facedetect;
	bool isSucces = facedetect.load("E:\\WorkStation\\Project\\OpencvDemoModify\\OpencvDemo\\data\\lbpcascades\\lbpcascade_frontalface_improved.xml");
	VideoCapture cap("E:\\WorkStation\\Project\\OpencvDemo\\OpencvDemo\\out.avi");
	if (!cap.isOpened()){
		std::cout << "open failed" << std::endl;
		return;
	}

	Mat frame;
	std::vector<cv::Rect> faces;
	while (true){
		cap >> frame;
		if (frame.empty()){
			std::cout << "frame empty" << std::endl;
			break;
		}
		Mat frameGray;
		std::vector<cv::Rect> faces;

		if (frame.channels() == 3)
		{
			cvtColor(frame, frameGray, CV_RGB2GRAY);
		}

		facedetect.detectMultiScale(frameGray, faces, 1.2, 3, 0, Size(30, 30));

		for (int i = 0; i < faces.size(); i++){
			Point c;
			c.x = faces[i].x + faces[i].width / 2;
			c.y = faces[i].y + faces[i].height / 2;
			int r = faces[i].width;
			cv::circle(frame, c, r, Scalar(255, 0, 0), 3, 8);
			std::string pyr("pyr:: ");
			break;
		}
		imshow("frame", frame);
		uchar c = waitKey(5);
		if (c == 'q')
		{
			break;
		}
	}
}
void Test()
{
	std::cout << "short" << sizeof(short) << std::endl;
	std::cout << "int" << sizeof(int) << std::endl;
	std::cout << "double" << sizeof(double) << std::endl;
	std::cout << "float" << sizeof(float) << std::endl;
	std::cout << "long int" << sizeof(long int) << std::endl;
	std::cout << "long long" << sizeof(long long) << std::endl;
}

struct temp
{
public:
	std::string str;
	int id;
};
void TestVector()
{
	std::vector<temp> t;
	for (int i = 0; i < 10; i++)
	{
		temp tmp;
		tmp.str = "hello";
		tmp.str.append(to_string(i));
		tmp.id = i;
		t.push_back(tmp);
	}
	cout << "capable" << t.size() << endl;
	cout << "show eleliment" << "\n" << endl;

	for (int i = 0; i < t.size(); i++)
	{
		cout << "index" << t[i].id << ":" << t[i].str << endl;
	}
}
