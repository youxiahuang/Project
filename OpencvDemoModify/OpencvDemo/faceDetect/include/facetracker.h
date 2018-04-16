#ifndef _OPENCV_FACETRACKER_HPP_
#define _OPENCV_FACETRACKER_HPP_
#include <opencv2/core.hpp>
#include "fhog.h"

#include <string>

#define D_USE_LINEAR_KERNEL
#define D_HOG_FEATURE

class FaceTracker 
{
public:
    // Constructor
    FaceTracker();

    // Initialize tracker 
    virtual void init(const cv::Rect &roi, cv::Mat image);
    
    // Update position based on the new frame
	virtual cv::Rect update(cv::Mat image, float* p_peak, bool needTrain = true);   
    virtual void setROI(cv::Rect_<float> roi){ _roi = roi; return; };


    float interp_factor; // linear interpolation factor for adaptation
    float sigma; // gaussian kernel bandwidth
    float lambda; // regularization
    int cell_size; // HOG cell size
    int cell_sizeQ; // cell size^2, to avoid repeated operations
    //float padding;
    float padding_w,padding_h; // extra area surrounding the target
    float output_sigma_factor; // bandwidth of gaussian target
    int template_size; // template size
	
	bool isInited;
	bool isDetected;


protected:
    // Detect object in the current frame.
    cv::Point2f detect(cv::Mat z, cv::Mat x, float &peak_value);

    // train tracker with a single image
    void train(cv::Mat x, float train_interp_factor);

    // Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
    cv::Mat gaussianCorrelation(cv::Mat x1, cv::Mat x2, bool isSameInputImage = false);

    // Create Gaussian Peak. Function called only in the first frame.
    cv::Mat createGaussianPeak(int sizey, int sizex);

    // Obtain sub-window from image, with replication-padding and extract features
    cv::Mat getFeatures(const cv::Mat & image, bool inithann, float scale_adjust = 1.0f);

    // Calculate sub-pixel peak for one dimension
    float subPixelPeak(float left, float center, float right);

    cv::Mat _alphaf;
    cv::Mat _prob;
    cv::Mat _tmpl;
    cv::Mat _num;
    cv::Mat _den;

private:
    cv::Rect_<float> _roi;
    int size_patch[3];
    cv::Size _tmpl_sz;
    float _scale;
    int _gaussian_size;

	bool _lost;
	int	_lost_frames;
};
#endif //_OPENCV_FACETRACKER_HPP_

