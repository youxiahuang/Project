#include <faceDetect.h>
#include <opencv2\videoio.hpp>
#include <opencv2\highgui.hpp>
#include <BNImgProc.h>

using namespace cv;

int main(int argc, char* argv[])
{

	bn::ft_init(argc, argv);

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
		flip(frame, frame, 1);
		bn::ft_process(frame, faces);

		for (int i = 0; i < faces.size(); i++){
			Point c;
			c.x = faces[i].x + faces[i].width / 2;
			c.y = faces[i].y + faces[i].height / 2;
			int r = faces[i].width / 2;
			cv::circle(frame, c, r, Scalar(255,0,0), 3, 8);
			std::string pyr("pyr:: ");
		}
		imshow("frame",frame);
		uchar c = waitKey(5);
		if (c == 'q')
		{
			break;
		}
	}

	bn::ft_uninit();


	return 1;
}
