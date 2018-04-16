#include "facetracker.h"

#include "ffttools.h"
#include "recttools.h"
#include <BNImgProc.h>

#include <opencv2/core.hpp>
//#include <opencv2/core/ocl.hpp>


static bool getDstRoi(const cv::Mat & image,int cx,int cy,float scale,cv::Size tmpl_sz,
				cv::Rect* pCutWindow,cv::Rect* pDstRoi)
{
	cv::Rect extracted_roi;

	extracted_roi.width = scale * tmpl_sz.width;
	extracted_roi.height = scale * tmpl_sz.height;

	// center roi with new size
	extracted_roi.x = cx - extracted_roi.width / 2;
	extracted_roi.y = cy - extracted_roi.height / 2;

	cv::Rect cutWindow = extracted_roi;
	RectTools::limit(cutWindow, image.cols, image.rows);
	if (cutWindow.height <= 0 || cutWindow.width <= 0) return false;

	cv::Rect dst_roi;
	dst_roi.x = (cutWindow.x - extracted_roi.x) / scale;
	dst_roi.y = (cutWindow.y - extracted_roi.y) / scale;
	dst_roi.width = cutWindow.width / scale;
	dst_roi.height = cutWindow.height / scale;

	*pCutWindow = cutWindow;
	*pDstRoi = dst_roi;

	return true;
}

static cv::Mat ResizeAndMakeBorder(const cv::Mat& image, int cx, int cy, float scale, cv::Size tmp_sz, int cl_mem_index)
{
	cv::Rect dst_roi, cutWindow;
	getDstRoi(image, cx, cy, scale, tmp_sz, &cutWindow, &dst_roi);

	cv::Mat cut_roi = image(cutWindow);

	cv::Mat dst;

	cv::resize(cut_roi, dst, cv::Size(dst_roi.width, dst_roi.height));
	//ImgProc::resize(cut_roi, dst, cv::Size(dst_roi.width, dst_roi.height));

	cv::Rect border = RectTools::getBorder(cv::Rect(0, 0, tmp_sz.width, tmp_sz.height),
		dst_roi);
	if (border != cv::Rect(0, 0, 0, 0))
	{
		cv::copyMakeBorder(dst, dst, border.y, border.height, border.x, border.width, cv::BORDER_REPLICATE);
	}

	return dst;
}

static cv::Mat ResizeAndMakeBorder(const cv::Mat& image, int cx, int cy, float scale, cv::Size tmp_sz)
{
	return ResizeAndMakeBorder(image, cx, cy, scale, tmp_sz, 1);
}

// Constructor
FaceTracker::FaceTracker()
{
	isInited = false;
	isDetected = true;

    // Parameters equal in all cases
    lambda = 0.0001;
    //padding = 2.5;
    padding_w = 2;
    padding_h = 2;
    //output_sigma_factor = 0.1;
    output_sigma_factor = 0.125;
	
	
#ifdef D_HOG_FEATURE
        interp_factor = 0.012;
        sigma = 0.6; 
		cell_size = 4;
#else
        interp_factor = 0.075;
		sigma = 0.2; 
		cell_size = 1;
#endif //D_HOG_FEATURE

	template_size = 96;
}

// Initialize tracker 
void FaceTracker::init(const cv::Rect &roi, cv::Mat image)
{
    _roi = roi;
    assert(roi.width >= 0 && roi.height >= 0);
    _tmpl = getFeatures(image, 1);
    _prob = createGaussianPeak(size_patch[0], size_patch[1]);
    _alphaf = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    train(_tmpl, 1.0); // train with initial frame

 }
// Update position based on the new frame
cv::Rect FaceTracker::update(cv::Mat image, float* p_peak, bool needTrain)
{
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 1;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 1;
    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 2;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 2;

    float cx = _roi.x + _roi.width / 2.0f;
    float cy = _roi.y + _roi.height / 2.0f;

	cv::Rect org_roi = _roi;

    float peak_value;

	cv::Point2f res;

	res = detect(_tmpl, getFeatures(image, 0, 1.0f), peak_value);



    // Adjust by cell size and _scale
    _roi.x = cx - _roi.width / 2.0f + ((float) res.x * cell_size * _scale);
    _roi.y = cy - _roi.height / 2.0f + ((float) res.y * cell_size * _scale);

    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;

	assert(_roi.width >= 0 && _roi.height >= 0);

	if (needTrain)
	{
		cv::Mat x = getFeatures(image, 0);

		//train(x, interp_factor);
		train(x, interp_factor * 1.4 * peak_value);
	}

	static float mean_pk = 0;
	static int cnt = 0;
	*p_peak = peak_value;

	mean_pk = (mean_pk * cnt + peak_value) / (cnt + 1);
	cnt++;
	//printf("mpeakvalue:%.2f    cPeakValue:%.2f\n", mean_pk, peak_value);

	return _roi;
}


// train tracker with a single image
void FaceTracker::train(cv::Mat x, float train_interp_factor)
{
    using namespace FFTTools;

#ifndef D_USE_LINEAR_KERNEL
    cv::Mat k = gaussianCorrelation(x, x, true);

	cv::Mat alphaf = complexDivision(_prob, (fftd(k,false,true) + lambda));
#else
	cv::Mat k;
#ifdef D_HOG_FEATURE
	k = cv::Mat(cv::Size(size_patch[1], size_patch[0]), CV_32FC2, cv::Scalar(0));
	cv::Mat caux;
	cv::Mat x1aux;
	cv::Mat auxfft;
	for (int i = 0; i < size_patch[2]; i++) {
		x1aux = x.row(i).reshape(1, size_patch[0]);
		auxfft = fftd(x1aux, false, true);
		cv::mulSpectrums(auxfft, auxfft, caux, 0, true);
		k = k + caux;
	}
#else //D_HOG_FEATURE
	cv::Mat auxfft = fftd(x, false, true);
	cv::mulSpectrums(auxfft, auxfft, k, 0, true);
#endif //D_HOG_FEATURE

	int size = size_patch[0] * size_patch[1] * size_patch[2];
	k = k / size;
	cv::Mat alphaf = complexDivision(_prob, (k + lambda));

#endif
	
    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor) * x;
    _alphaf = (1 - train_interp_factor) * _alphaf + (train_interp_factor) * alphaf;

}

// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, 
// which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
cv::Mat FaceTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2, bool isSameInputImage)
{
    using namespace FFTTools;
	cv::Mat c = cv::Mat(cv::Size(size_patch[1], size_patch[0]), CV_32F, cv::Scalar(0));
	//cv::Mat cc;
	//c.copyTo(cc);

#ifdef D_HOG_FEATURE
	//c = cv::Mat(cv::Size(size_patch[1], size_patch[0]), CV_64F, cv::Scalar(0));
	cv::Mat caux;
	cv::Mat x1aux;
	cv::Mat x2aux;
	//std::vector<cv::Mat_<double> > v_x1;
	//cv::split(x1, v_x1);

	if (!isSameInputImage)
	{
		
		//std::vector<cv::Mat_<double> > v_x2;
		//cv::split(x2, v_x2);

		for (int i = 0; i < size_patch[2]; i++) {
			x1aux = x1.row(i);   // Procedure do deal with cv::Mat multichannel bug
			x1aux = x1aux.reshape(1, size_patch[0]);
			x2aux = x2.row(i).reshape(1, size_patch[0]);
			//x1aux = v_x1[i];
			//x2aux = v_x2[i];
			cv::mulSpectrums(fftd(x1aux, false, true), fftd(x2aux, false, true), caux, 0, true);
			caux = fftd(caux, true, true);
			c = c + caux;
		}
	}
	else
	{
		cv::Mat auxfft;
		for (int i = 0; i < size_patch[2]; i++) {
			x1aux = x1.row(i).reshape(1, size_patch[0]);
			//x1aux = v_x1[i];
			auxfft = fftd(x1aux, false, true);
			cv::mulSpectrums(auxfft, auxfft, caux, 0, true);
			caux = fftd(caux, true, true);
			c = c + caux;
		}
	}
	rearrange(c);
#else //D_HOG_FEATURE
	//c = cv::Mat(cv::Size(size_patch[1], size_patch[0]), CV_32F, cv::Scalar(0));
	if (!isSameInputImage)
	{
		cv::mulSpectrums(fftd(x1,false,true), fftd(x2,false,true), c, 0, true);
	}
	else
	{
		cv::Mat auxfft = fftd(x1,false,true);
		cv::mulSpectrums(auxfft, auxfft, c, 0, true);
	}
	c = fftd(c, true);
	rearrange(c);
	c = real(c);

#endif //D_HOG_FEATURE
	
    cv::Mat d; 

#ifdef max
#undef max
#endif
	cv::max(( (cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul	(x2))[0])- 2. * c) 
				/ (size_patch[0]*size_patch[1]*size_patch[2]) ,
				 0, d);

    cv::Mat k;
    cv::exp((-d / (sigma * sigma)), k);
    return k;
}

// Create Gaussian Peak. Function called only in the first frame.
cv::Mat FaceTracker::createGaussianPeak(int sizey, int sizex)
{
    cv::Mat_<float> res(sizey, sizex);

    int syh = (sizey) / 2;
    int sxh = (sizex) / 2;

     //float output_sigma = std::sqrt((float) sizex * sizey) / padding * output_sigma_factor;
    float output_sigma = std::sqrt((float) sizex * sizey) / padding_h * output_sigma_factor;

    float mult = -0.5 / (output_sigma * output_sigma);

    for (int i = 0; i < sizey; i++)
        for (int j = 0; j < sizex; j++)
        {
            int ih = i - syh;
            int jh = j - sxh;
            res(i, j) = std::exp(mult * (float) (ih * ih + jh * jh));
        }
    return FFTTools::fftd(res,false,true);
}

// Obtain sub-window from image, with replication-padding and extract features
cv::Mat FaceTracker::getFeatures(const cv::Mat & image, bool inithann, float scale_adjust)
{
	cv::Rect extracted_roi;

    float cx = _roi.x + _roi.width / 2;
    float cy = _roi.y + _roi.height / 2;

    if (inithann) {
        int padded_w = _roi.width * padding_w;
        int padded_h = _roi.height * padding_h;
        
        if (template_size > 1) {  // Fit largest dimension to the given template size
            if (padded_w >= padded_h)  //fit to width
                _scale = padded_w / (float) template_size;
            else
                _scale = padded_h / (float) template_size;

            _tmpl_sz.width = padded_w / _scale;
            _tmpl_sz.height = padded_h / _scale;
        }
        else {  //No template size given, use ROI size
            _tmpl_sz.width = padded_w;
            _tmpl_sz.height = padded_h;
            _scale = 1;
        }

#ifdef D_HOG_FEATURE
            // Round to cell size and also make it even
            _tmpl_sz.width = ( ( (int)(_tmpl_sz.width / (2 * cell_size)) ) * 2 * cell_size ) + cell_size*2;
            _tmpl_sz.height = ( ( (int)(_tmpl_sz.height / (2 * cell_size)) ) * 2 * cell_size ) + cell_size*2;
#else  //Make number of pixels even (helps with some logic involving half-dimensions)
            _tmpl_sz.width = (_tmpl_sz.width / 2) * 2;
            _tmpl_sz.height = (_tmpl_sz.height / 2) * 2;
#endif //D_HOG_FEATURE
    }
	cv::Mat z = ResizeAndMakeBorder(image, cx, cy, _scale, _tmpl_sz, 1);

	cv::Mat FeaturesMap;

#ifdef D_HOG_FEATURE

		FeaturesMap = fhog::fhog(z, cell_size);


		//size_patch[0] = FeaturesMap.rows;
		//size_patch[1] = FeaturesMap.cols;
		//size_patch[2] = FeaturesMap.channels();
		size_patch[0] = max((int)round((float)z.rows / cell_size) - 2, 0);
		size_patch[1] = max((int)round((float)z.cols / cell_size) - 2, 0);
		size_patch[2] = FeaturesMap.rows-1;
#else
	FeaturesMap = RectTools::getGrayImage(z);
	//FeaturesMap -= (float) 0.5; // In Paper;

	size_patch[0] = z.rows;
	size_patch[1] = z.cols;
	size_patch[2] = 1;  
#endif //D_HOG_FEATURE
	


    return FeaturesMap;
}
    

// Calculate sub-pixel peak for one dimension
float FaceTracker::subPixelPeak(float left, float center, float right)
{   
    float divisor = 2 * center - right - left;

    if (divisor == 0)
        return 0;
    
    return 0.5 * (right - left) / divisor;
}


// Detect object in the current frame.
cv::Point2f FaceTracker::detect(cv::Mat z, cv::Mat x, float &peak_value)
{
    using namespace FFTTools;


#ifndef D_USE_LINEAR_KERNEL
	cv::Mat k = gaussianCorrelation(x, z);

#ifdef D_DO_REAL_DFT
	cv::Mat res = fftd(complexMultiplication(_alphaf, fftd(k, false, true)), true, true);
#else
	cv::Mat res = (real(fftd(complexMultiplication(_alphaf, fftd(k)), true)));
#endif
#else
	cv::Mat k;
#ifdef D_HOG_FEATURE
	k = cv::Mat(cv::Size(size_patch[1], size_patch[0]), CV_32FC2, cv::Scalar(0));
	cv::Mat caux;
	cv::Mat x1aux;
	cv::Mat x2aux;
	for (int i = 0; i < size_patch[2]; i++) {
		x1aux = x.row(i).reshape(1, size_patch[0]);
		x2aux = z.row(i).reshape(1, size_patch[0]);
		cv::mulSpectrums(fftd(x1aux, false, true), fftd(x2aux, false, true), caux, 0, true);
		k = k + caux;
	}
#else
	cv::mulSpectrums(fftd(x, false, true), fftd(z, false, true), k, 0, true);
#endif //D_HOG_FEATURE
	int size = size_patch[0] * size_patch[1] * size_patch[2];
	k = k / size;
	cv::Mat res = fftd(complexMultiplication(_alphaf, k), true, true);
#endif

    //minMaxLoc only accepts doubles for the peak, and integer points for the coordinates
    cv::Point2i pi;
    double pv;
    cv::minMaxLoc(res, NULL, &pv, NULL, &pi);
    peak_value = (float) pv;

    //subpixel peak estimation, coordinates will be non-integer
    cv::Point2f p((float)pi.x, (float)pi.y);

    if (pi.x > 0 && pi.x < res.cols-1) 
	{
        p.x += subPixelPeak(res.at<float>(pi.y, pi.x-1), peak_value, res.at<float>(pi.y, pi.x+1));
    }

    if (pi.y > 0 && pi.y < res.rows-1) {
        p.y += subPixelPeak(res.at<float>(pi.y-1, pi.x), peak_value, res.at<float>(pi.y+1, pi.x));
    }

    p.x -= (res.cols) / 2;
    p.y -= (res.rows) / 2;

    return p;
}

