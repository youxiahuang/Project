#ifndef BNIMGPROC_HPP_
#define BNIMGPROC_HPP_

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/core/core_c.h"

using namespace cv;


///////////////////////////////////// YUV420 -> RGB /////////////////////////////////////

const int ITUR_BT_601_CY = 1220542;
const int ITUR_BT_601_CUB = 2116026;
const int ITUR_BT_601_CUG = -409993;
const int ITUR_BT_601_CVG = -852492;
const int ITUR_BT_601_CVR = 1673527;
const int ITUR_BT_601_SHIFT = 20;


/** the color conversion code
@see @ref imgproc_color_conversions
@ingroup imgproc_misc
*/
enum ColorConversionCodes {
	//COLOR_BGR2BGRA = 0, //!< add alpha channel to RGB or BGR image
	//COLOR_RGB2RGBA = COLOR_BGR2BGRA,

	//COLOR_BGRA2BGR = 1, //!< remove alpha channel from RGB or BGR image
	//COLOR_RGBA2RGB = COLOR_BGRA2BGR,

	//COLOR_BGR2RGBA = 2, //!< convert between RGB and BGR color spaces (with or without alpha channel)
	//COLOR_RGB2BGRA = COLOR_BGR2RGBA,

	//COLOR_RGBA2BGR = 3,
	//COLOR_BGRA2RGB = COLOR_RGBA2BGR,

	COLOR_BGR2RGB = 4,
	COLOR_RGB2BGR = COLOR_BGR2RGB,

	//COLOR_BGRA2RGBA = 5,
	//COLOR_RGBA2BGRA = COLOR_BGRA2RGBA,

	COLOR_BGR2GRAY = 6, //!< convert between RGB/BGR and grayscale, @ref color_convert_rgb_gray "color conversions"
	COLOR_RGB2GRAY = 7,
	COLOR_GRAY2BGR = 8,
	COLOR_GRAY2RGB = COLOR_GRAY2BGR,
	//COLOR_GRAY2BGRA = 9,
	//COLOR_GRAY2RGBA = COLOR_GRAY2BGRA,
	//COLOR_BGRA2GRAY = 10,
	//COLOR_RGBA2GRAY = 11,

	//COLOR_BGR2BGR565 = 12, //!< convert between RGB/BGR and BGR565 (16-bit images)
	//COLOR_RGB2BGR565 = 13,
	//COLOR_BGR5652BGR = 14,
	//COLOR_BGR5652RGB = 15,
	//COLOR_BGRA2BGR565 = 16,
	//COLOR_RGBA2BGR565 = 17,
	//COLOR_BGR5652BGRA = 18,
	//COLOR_BGR5652RGBA = 19,

	//COLOR_GRAY2BGR565 = 20, //!< convert between grayscale to BGR565 (16-bit images)
	//COLOR_BGR5652GRAY = 21,

	//COLOR_BGR2BGR555 = 22,  //!< convert between RGB/BGR and BGR555 (16-bit images)
	//COLOR_RGB2BGR555 = 23,
	//COLOR_BGR5552BGR = 24,
	//COLOR_BGR5552RGB = 25,
	//COLOR_BGRA2BGR555 = 26,
	//COLOR_RGBA2BGR555 = 27,
	//COLOR_BGR5552BGRA = 28,
	//COLOR_BGR5552RGBA = 29,

	//COLOR_GRAY2BGR555 = 30, //!< convert between grayscale and BGR555 (16-bit images)
	//COLOR_BGR5552GRAY = 31,

	//COLOR_BGR2XYZ = 32, //!< convert RGB/BGR to CIE XYZ, @ref color_convert_rgb_xyz "color conversions"
	//COLOR_RGB2XYZ = 33,
	//COLOR_XYZ2BGR = 34,
	//COLOR_XYZ2RGB = 35,

	//COLOR_BGR2YCrCb = 36, //!< convert RGB/BGR to luma-chroma (aka YCC), @ref color_convert_rgb_ycrcb "color conversions"
	//COLOR_RGB2YCrCb = 37,
	//COLOR_YCrCb2BGR = 38,
	//COLOR_YCrCb2RGB = 39,

	//COLOR_BGR2HSV = 40, //!< convert RGB/BGR to HSV (hue saturation value), @ref color_convert_rgb_hsv "color conversions"
	//COLOR_RGB2HSV = 41,

	//COLOR_BGR2Lab = 44, //!< convert RGB/BGR to CIE Lab, @ref color_convert_rgb_lab "color conversions"
	//COLOR_RGB2Lab = 45,

	//COLOR_BGR2Luv = 50, //!< convert RGB/BGR to CIE Luv, @ref color_convert_rgb_luv "color conversions"
	//COLOR_RGB2Luv = 51,
	//COLOR_BGR2HLS = 52, //!< convert RGB/BGR to HLS (hue lightness saturation), @ref color_convert_rgb_hls "color conversions"
	//COLOR_RGB2HLS = 53,

	//COLOR_HSV2BGR = 54, //!< backward conversions to RGB/BGR
	//COLOR_HSV2RGB = 55,

	//COLOR_Lab2BGR = 56,
	//COLOR_Lab2RGB = 57,
	//COLOR_Luv2BGR = 58,
	//COLOR_Luv2RGB = 59,
	//COLOR_HLS2BGR = 60,
	//COLOR_HLS2RGB = 61,

	//COLOR_BGR2HSV_FULL = 66, //!<
	//COLOR_RGB2HSV_FULL = 67,
	//COLOR_BGR2HLS_FULL = 68,
	//COLOR_RGB2HLS_FULL = 69,

	//COLOR_HSV2BGR_FULL = 70,
	//COLOR_HSV2RGB_FULL = 71,
	//COLOR_HLS2BGR_FULL = 72,
	//COLOR_HLS2RGB_FULL = 73,

	//COLOR_LBGR2Lab = 74,
	//COLOR_LRGB2Lab = 75,
	//COLOR_LBGR2Luv = 76,
	//COLOR_LRGB2Luv = 77,

	//COLOR_Lab2LBGR = 78,
	//COLOR_Lab2LRGB = 79,
	//COLOR_Luv2LBGR = 80,
	//COLOR_Luv2LRGB = 81,

	//COLOR_BGR2YUV = 82, //!< convert between RGB/BGR and YUV
	//COLOR_RGB2YUV = 83,
	//COLOR_YUV2BGR = 84,
	//COLOR_YUV2RGB = 85,

	////! YUV 4:2:0 family to RGB
	//COLOR_YUV2RGB_NV12 = 90,
	//COLOR_YUV2BGR_NV12 = 91,
	//COLOR_YUV2RGB_NV21 = 92,
	//COLOR_YUV2BGR_NV21 = 93,
	//COLOR_YUV420sp2RGB = COLOR_YUV2RGB_NV21,
	//COLOR_YUV420sp2BGR = COLOR_YUV2BGR_NV21,

	//COLOR_YUV2RGBA_NV12 = 94,
	//COLOR_YUV2BGRA_NV12 = 95,
	//COLOR_YUV2RGBA_NV21 = 96,
	//COLOR_YUV2BGRA_NV21 = 97,
	//COLOR_YUV420sp2RGBA = COLOR_YUV2RGBA_NV21,
	//COLOR_YUV420sp2BGRA = COLOR_YUV2BGRA_NV21,

	//COLOR_YUV2RGB_YV12 = 98,
	COLOR_YUV2BGR_YV12 = 99,
	//COLOR_YUV2RGB_IYUV = 100,
	//COLOR_YUV2BGR_IYUV = 101,
	//COLOR_YUV2RGB_I420 = COLOR_YUV2RGB_IYUV,
	//COLOR_YUV2BGR_I420 = COLOR_YUV2BGR_IYUV,
	//COLOR_YUV420p2RGB = COLOR_YUV2RGB_YV12,
	//COLOR_YUV420p2BGR = COLOR_YUV2BGR_YV12,

	//COLOR_YUV2RGBA_YV12 = 102,
	//COLOR_YUV2BGRA_YV12 = 103,
	//COLOR_YUV2RGBA_IYUV = 104,
	//COLOR_YUV2BGRA_IYUV = 105,
	//COLOR_YUV2RGBA_I420 = COLOR_YUV2RGBA_IYUV,
	//COLOR_YUV2BGRA_I420 = COLOR_YUV2BGRA_IYUV,
	//COLOR_YUV420p2RGBA = COLOR_YUV2RGBA_YV12,
	//COLOR_YUV420p2BGRA = COLOR_YUV2BGRA_YV12,

	//COLOR_YUV2GRAY_420 = 106,
	//COLOR_YUV2GRAY_NV21 = COLOR_YUV2GRAY_420,
	//COLOR_YUV2GRAY_NV12 = COLOR_YUV2GRAY_420,
	//COLOR_YUV2GRAY_YV12 = COLOR_YUV2GRAY_420,
	//COLOR_YUV2GRAY_IYUV = COLOR_YUV2GRAY_420,
	//COLOR_YUV2GRAY_I420 = COLOR_YUV2GRAY_420,
	//COLOR_YUV420sp2GRAY = COLOR_YUV2GRAY_420,
	//COLOR_YUV420p2GRAY = COLOR_YUV2GRAY_420,

	////! YUV 4:2:2 family to RGB
	//COLOR_YUV2RGB_UYVY = 107,
	COLOR_YUV2BGR_UYVY = 108,
	////COLOR_YUV2RGB_VYUY = 109,
	////COLOR_YUV2BGR_VYUY = 110,
	//COLOR_YUV2RGB_Y422 = COLOR_YUV2RGB_UYVY,
	//COLOR_YUV2BGR_Y422 = COLOR_YUV2BGR_UYVY,
	//COLOR_YUV2RGB_UYNV = COLOR_YUV2RGB_UYVY,
	//COLOR_YUV2BGR_UYNV = COLOR_YUV2BGR_UYVY,

	//COLOR_YUV2RGBA_UYVY = 111,
	//COLOR_YUV2BGRA_UYVY = 112,
	////COLOR_YUV2RGBA_VYUY = 113,
	////COLOR_YUV2BGRA_VYUY = 114,
	//COLOR_YUV2RGBA_Y422 = COLOR_YUV2RGBA_UYVY,
	//COLOR_YUV2BGRA_Y422 = COLOR_YUV2BGRA_UYVY,
	//COLOR_YUV2RGBA_UYNV = COLOR_YUV2RGBA_UYVY,
	//COLOR_YUV2BGRA_UYNV = COLOR_YUV2BGRA_UYVY,

	//COLOR_YUV2RGB_YUY2 = 115,
	COLOR_YUV2BGR_YUY2 = 116,
	//COLOR_YUV2RGB_YVYU = 117,
	//COLOR_YUV2BGR_YVYU = 118,
	//COLOR_YUV2RGB_YUYV = COLOR_YUV2RGB_YUY2,
	COLOR_YUV2BGR_YUYV = COLOR_YUV2BGR_YUY2,
	//COLOR_YUV2RGB_YUNV = COLOR_YUV2RGB_YUY2,
	COLOR_YUV2BGR_YUNV = COLOR_YUV2BGR_YUY2,

	//COLOR_YUV2RGBA_YUY2 = 119,
	//COLOR_YUV2BGRA_YUY2 = 120,
	//COLOR_YUV2RGBA_YVYU = 121,
	//COLOR_YUV2BGRA_YVYU = 122,
	//COLOR_YUV2RGBA_YUYV = COLOR_YUV2RGBA_YUY2,
	//COLOR_YUV2BGRA_YUYV = COLOR_YUV2BGRA_YUY2,
	//COLOR_YUV2RGBA_YUNV = COLOR_YUV2RGBA_YUY2,
	//COLOR_YUV2BGRA_YUNV = COLOR_YUV2BGRA_YUY2,

	//COLOR_YUV2GRAY_UYVY = 123,
	//COLOR_YUV2GRAY_YUY2 = 124,
	////CV_YUV2GRAY_VYUY    = CV_YUV2GRAY_UYVY,
	//COLOR_YUV2GRAY_Y422 = COLOR_YUV2GRAY_UYVY,
	//COLOR_YUV2GRAY_UYNV = COLOR_YUV2GRAY_UYVY,
	//COLOR_YUV2GRAY_YVYU = COLOR_YUV2GRAY_YUY2,
	//COLOR_YUV2GRAY_YUYV = COLOR_YUV2GRAY_YUY2,
	//COLOR_YUV2GRAY_YUNV = COLOR_YUV2GRAY_YUY2,

	////! alpha premultiplication
	//COLOR_RGBA2mRGBA = 125,
	//COLOR_mRGBA2RGBA = 126,

	////! RGB to YUV 4:2:0 family
	//COLOR_RGB2YUV_I420 = 127,
	//COLOR_BGR2YUV_I420 = 128,
	//COLOR_RGB2YUV_IYUV = COLOR_RGB2YUV_I420,
	//COLOR_BGR2YUV_IYUV = COLOR_BGR2YUV_I420,

	//COLOR_RGBA2YUV_I420 = 129,
	//COLOR_BGRA2YUV_I420 = 130,
	//COLOR_RGBA2YUV_IYUV = COLOR_RGBA2YUV_I420,
	//COLOR_BGRA2YUV_IYUV = COLOR_BGRA2YUV_I420,
	//COLOR_RGB2YUV_YV12 = 131,
	//COLOR_BGR2YUV_YV12 = 132,
	//COLOR_RGBA2YUV_YV12 = 133,
	//COLOR_BGRA2YUV_YV12 = 134,

	////! Demosaicing
	//COLOR_BayerBG2BGR = 46,
	//COLOR_BayerGB2BGR = 47,
	//COLOR_BayerRG2BGR = 48,
	//COLOR_BayerGR2BGR = 49,

	//COLOR_BayerBG2RGB = COLOR_BayerRG2BGR,
	//COLOR_BayerGB2RGB = COLOR_BayerGR2BGR,
	//COLOR_BayerRG2RGB = COLOR_BayerBG2BGR,
	//COLOR_BayerGR2RGB = COLOR_BayerGB2BGR,

	//COLOR_BayerBG2GRAY = 86,
	//COLOR_BayerGB2GRAY = 87,
	//COLOR_BayerRG2GRAY = 88,
	//COLOR_BayerGR2GRAY = 89,

	////! Demosaicing using Variable Number of Gradients
	//COLOR_BayerBG2BGR_VNG = 62,
	//COLOR_BayerGB2BGR_VNG = 63,
	//COLOR_BayerRG2BGR_VNG = 64,
	//COLOR_BayerGR2BGR_VNG = 65,

	//COLOR_BayerBG2RGB_VNG = COLOR_BayerRG2BGR_VNG,
	//COLOR_BayerGB2RGB_VNG = COLOR_BayerGR2BGR_VNG,
	//COLOR_BayerRG2RGB_VNG = COLOR_BayerBG2BGR_VNG,
	//COLOR_BayerGR2RGB_VNG = COLOR_BayerGB2BGR_VNG,

	////! Edge-Aware Demosaicing
	//COLOR_BayerBG2BGR_EA = 135,
	//COLOR_BayerGB2BGR_EA = 136,
	//COLOR_BayerRG2BGR_EA = 137,
	//COLOR_BayerGR2BGR_EA = 138,

	//COLOR_BayerBG2RGB_EA = COLOR_BayerRG2BGR_EA,
	//COLOR_BayerGB2RGB_EA = COLOR_BayerGR2BGR_EA,
	//COLOR_BayerRG2RGB_EA = COLOR_BayerBG2BGR_EA,
	//COLOR_BayerGR2RGB_EA = COLOR_BayerGB2BGR_EA,

	////! Demosaicing with alpha channel
	//COLOR_BayerBG2BGRA = 139,
	//COLOR_BayerGB2BGRA = 140,
	//COLOR_BayerRG2BGRA = 141,
	//COLOR_BayerGR2BGRA = 142,

	//COLOR_BayerBG2RGBA = COLOR_BayerRG2BGRA,
	//COLOR_BayerGB2RGBA = COLOR_BayerGR2BGRA,
	//COLOR_BayerRG2RGBA = COLOR_BayerBG2BGRA,
	//COLOR_BayerGR2RGBA = COLOR_BayerGB2BGRA,

	COLOR_COLORCVT_MAX = 143
};

//! interpolation algorithm
enum InterpolationFlags{
	///** nearest neighbor interpolation */
	//INTER_NEAREST = 0,
	/** bilinear interpolation */
	INTER_LINEAR = 1,
	///** bicubic interpolation */
	//INTER_CUBIC = 2,
	///** resampling using pixel area relation. It may be a preferred method for image decimation, as
	//it gives moire'-free results. But when the image is zoomed, it is similar to the INTER_NEAREST
	//method. */
	INTER_AREA = 3,
	///** Lanczos interpolation over 8x8 neighborhood */
	//INTER_LANCZOS4 = 4,
	///** mask for interpolation codes */
	//INTER_MAX = 7,
	///** flag, fills all of the destination image pixels. If some of them correspond to outliers in the
	//source image, they are set to zero */
	//WARP_FILL_OUTLIERS = 8,
	///** flag, inverse transformation
	//
	//For example, @ref cv::linearPolar or @ref cv::logPolar transforms:
	//- flag is __not__ set: \f$dst( \rho , \phi ) = src(x,y)\f$
	//- flag is set: \f$dst(x,y) = src( \rho , \phi )\f$
	//*/
	//WARP_INVERSE_MAP = 16
};


#undef R2Y
#undef G2Y
#undef B2Y

enum
{
	yuv_shift = 14,
	xyz_shift = 12,
	R2Y = 4899, // == R2YF*16384
	G2Y = 9617, // == G2YF*16384
	B2Y = 1868, // == B2YF*16384
	BLOCK_SIZE = 256
};

//constants for conversion from/to RGB and Gray, YUV, YCrCb according to BT.601
const float B2YF = 0.114f;
const float G2YF = 0.587f;
const float R2YF = 0.299f;

const int INTER_RESIZE_COEF_BITS = 11;
const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;
static const int MAX_ESIZE = 16;


namespace cv{
	void cvtColor(InputArray src, OutputArray dst, int code, int dstCn = 0);
	inline void cvCvtColor(const CvArr* srcarr, CvArr* dstarr, int code)
	{
		cv::Mat src = cv::cvarrToMat(srcarr), dst0 = cv::cvarrToMat(dstarr), dst = dst0;
		CV_Assert(src.depth() == dst.depth());
		cvtColor(src, dst, code, dst.channels());
		CV_Assert(dst.data == dst0.data);
	}

	void cvtBGRtoGray(const uchar * src_data, size_t src_step,
		uchar * dst_data, size_t dst_step,
		int width, int height,
		int depth, int scn, bool swapBlue);

	void cvtGraytoBGR(const uchar * src_data, size_t src_step,
		uchar * dst_data, size_t dst_step,
		int width, int height,
		int depth, int dcn);

	void cvtOnePlaneYUVtoBGR(const uchar * src_data, size_t src_step,
		uchar * dst_data, size_t dst_step,
		int width, int height,
		int dcn, bool swapBlue, int uIdx, int ycn);

	void resize(InputArray src, OutputArray dst,
		Size dsize, double fx = 0, double fy = 0,
		int interpolation = INTER_LINEAR);

	inline void cvResize(const CvArr* srcarr, CvArr* dstarr, int method)
	{
		cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);
		CV_Assert(src.type() == dst.type());
		resize(src, dst, dst.size(), (double)dst.cols / src.cols,
			(double)dst.rows / src.rows, method);
	}

	void halResize(int src_type,
		const uchar * src_data, size_t src_step, int src_width, int src_height,
		uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
		double inv_scale_x, double inv_scale_y, int interpolation);

	void integral(InputArray src, OutputArray sum,
		OutputArray sqsum, OutputArray tilted,
		int sdepth = -1, int sqdepth = -1);

	inline void cvIntegral(const CvArr* image, CvArr* sumImage,
		CvArr* sumSqImage, CvArr* tiltedSumImage)
	{
		cv::Mat src = cv::cvarrToMat(image), sum = cv::cvarrToMat(sumImage), sum0 = sum;
		cv::Mat sqsum0, sqsum, tilted0, tilted;
		cv::Mat *psqsum = 0, *ptilted = 0;

		if (sumSqImage)
		{
			sqsum0 = sqsum = cv::cvarrToMat(sumSqImage);
			psqsum = &sqsum;
		}

		if (tiltedSumImage)
		{
			tilted0 = tilted = cv::cvarrToMat(tiltedSumImage);
			ptilted = &tilted;
		}
		integral(src, sum, psqsum ? cv::_OutputArray(*psqsum) : cv::_OutputArray(),
			ptilted ? cv::_OutputArray(*ptilted) : cv::_OutputArray(), sum.depth());

		CV_Assert(sum.data == sum0.data && sqsum.data == sqsum0.data && tilted.data == tilted0.data);
	}


	void halIntegral(int depth, int sdepth, int sqdepth,
		const uchar* src, size_t srcstep,
		uchar* sum, size_t sumstep,
		uchar* sqsum, size_t sqsumstep,
		uchar* tilted, size_t tstep,
		int width, int height, int cn);

	void circle(InputOutputArray img, Point center, int radius,
		const Scalar& color, int thickness = 1,
		int lineType = LINE_8, int shift = 0);


	void equalizeHist(InputArray src, OutputArray dst);

};


template<typename _Tp> struct RGB2Gray
{
	typedef _Tp channel_type;

	RGB2Gray(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
	{
		static const float coeffs0[] = { R2YF, G2YF, B2YF };
		memcpy(coeffs, _coeffs ? _coeffs : coeffs0, 3 * sizeof(coeffs[0]));
		if (blueIdx == 0)
			std::swap(coeffs[0], coeffs[2]);
	}

	void operator()(const _Tp* src, _Tp* dst, int n) const
	{
		int scn = srccn;
		float cb = coeffs[0], cg = coeffs[1], cr = coeffs[2];
		for (int i = 0; i < n; i++, src += scn)
			dst[i] = saturate_cast<_Tp>(src[0] * cb + src[1] * cg + src[2] * cr);
	}
	int srccn;
	float coeffs[3];
};

template<> struct RGB2Gray<uchar>
{
	typedef uchar channel_type;

	RGB2Gray(int _srccn, int blueIdx, const int* coeffs) : srccn(_srccn)
	{
		const int coeffs0[] = { R2Y, G2Y, B2Y };
		if (!coeffs) coeffs = coeffs0;

		int b = 0, g = 0, r = (1 << (yuv_shift - 1));
		int db = coeffs[blueIdx ^ 2], dg = coeffs[1], dr = coeffs[blueIdx];

		for (int i = 0; i < 256; i++, b += db, g += dg, r += dr)
		{
			tab[i] = b;
			tab[i + 256] = g;
			tab[i + 512] = r;
		}
	}
	void operator()(const uchar* src, uchar* dst, int n) const
	{
		int scn = srccn;
		const int* _tab = tab;
		for (int i = 0; i < n; i++, src += scn)
			dst[i] = (uchar)((_tab[src[0]] + _tab[src[1] + 256] + _tab[src[2] + 512]) >> yuv_shift);
	}
	int srccn;
	int tab[256 * 3];
};

///////////////////////////////// Color to/from Grayscale ////////////////////////////////
template<typename _Tp> struct ColorChannel
{
	typedef float worktype_f;
	static _Tp max() { return std::numeric_limits<_Tp>::max(); }
	static _Tp half() { return (_Tp)(max() / 2 + 1); }
};

template<> struct ColorChannel<float>
{
	typedef float worktype_f;
	static float max() { return 1.f; }
	static float half() { return 0.5f; }
};

template<typename _Tp>
struct Gray2RGB
{
	typedef _Tp channel_type;

	Gray2RGB(int _dstcn) : dstcn(_dstcn) {}
	void operator()(const _Tp* src, _Tp* dst, int n) const
	{
		if (dstcn == 3)
			for (int i = 0; i < n; i++, dst += 3)
			{
				dst[0] = dst[1] = dst[2] = src[i];
			}
		else
		{
			_Tp alpha = ColorChannel<_Tp>::max();
			for (int i = 0; i < n; i++, dst += 4)
			{
				dst[0] = dst[1] = dst[2] = src[i];
				dst[3] = alpha;
			}
		}
	}

	int dstcn;
};


///////////////////////////// Top-level template function ////////////////////////////////

template <typename Cvt>
class CvtColorLoop_Invoker : public ParallelLoopBody
{
	typedef typename Cvt::channel_type _Tp;
public:

	CvtColorLoop_Invoker(const uchar * src_data_, size_t src_step_, uchar * dst_data_, size_t dst_step_, int width_, const Cvt& _cvt) :
		ParallelLoopBody(), src_data(src_data_), src_step(src_step_), dst_data(dst_data_), dst_step(dst_step_),
		width(width_), cvt(_cvt)
	{
	}

	virtual void operator()(const Range& range) const
	{
		//CV_TRACE_FUNCTION();

		const uchar* yS = src_data + static_cast<size_t>(range.start) * src_step;
		uchar* yD = dst_data + static_cast<size_t>(range.start) * dst_step;

		for (int i = range.start; i < range.end; ++i, yS += src_step, yD += dst_step)
			cvt(reinterpret_cast<const _Tp*>(yS), reinterpret_cast<_Tp*>(yD), width);
	}

private:
	const uchar * src_data;
	size_t src_step;
	uchar * dst_data;
	size_t dst_step;
	int width;
	const Cvt& cvt;

	const CvtColorLoop_Invoker& operator= (const CvtColorLoop_Invoker&);
};

template <typename Cvt>
void CvtColorLoop(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, const Cvt& cvt)
{
	parallel_for_(Range(0, height),
		CvtColorLoop_Invoker<Cvt>(src_data, src_step, dst_data, dst_step, width, cvt),
		(width * height) / static_cast<double>(1 << 16));
}

template <typename T, typename WT, typename VecOp>
class resizeAreaFast_Invoker :
	public ParallelLoopBody
{
public:
	resizeAreaFast_Invoker(const Mat &_src, Mat &_dst,
		int _scale_x, int _scale_y, const int* _ofs, const int* _xofs) :
		ParallelLoopBody(), src(_src), dst(_dst), scale_x(_scale_x),
		scale_y(_scale_y), ofs(_ofs), xofs(_xofs)
	{
	}

	virtual void operator() (const Range& range) const
	{
		Size ssize = src.size(), dsize = dst.size();
		int cn = src.channels();
		int area = scale_x*scale_y;
		float scale = 1.f / (area);
		int dwidth1 = (ssize.width / scale_x)*cn;
		dsize.width *= cn;
		ssize.width *= cn;
		int dy, dx, k = 0;

		VecOp vop(scale_x, scale_y, src.channels(), (int)src.step/*, area_ofs*/);

		for (dy = range.start; dy < range.end; dy++)
		{
			T* D = (T*)(dst.data + dst.step*dy);
			int sy0 = dy*scale_y;
			int w = sy0 + scale_y <= ssize.height ? dwidth1 : 0;

			if (sy0 >= ssize.height)
			{
				for (dx = 0; dx < dsize.width; dx++)
					D[dx] = 0;
				continue;
			}

			dx = vop(src.template ptr<T>(sy0), D, w);
			for (; dx < w; dx++)
			{
				const T* S = src.template ptr<T>(sy0) +xofs[dx];
				WT sum = 0;
				k = 0;
//#if CV_ENABLE_UNROLLED
//				for (; k <= area - 4; k += 4)
//					sum += S[ofs[k]] + S[ofs[k + 1]] + S[ofs[k + 2]] + S[ofs[k + 3]];
//#endif
				for (; k < area; k++)
					sum += S[ofs[k]];

				D[dx] = saturate_cast<T>(sum * scale);
			}

			for (; dx < dsize.width; dx++)
			{
				WT sum = 0;
				int count = 0, sx0 = xofs[dx];
				if (sx0 >= ssize.width)
					D[dx] = 0;

				for (int sy = 0; sy < scale_y; sy++)
				{
					if (sy0 + sy >= ssize.height)
						break;
					const T* S = src.template ptr<T>(sy0 + sy) + sx0;
					for (int sx = 0; sx < scale_x*cn; sx += cn)
					{
						if (sx0 + sx >= ssize.width)
							break;
						sum += S[sx];
						count++;
					}
				}

				D[dx] = saturate_cast<T>((float)sum / count);
			}
		}
	}

private:
	Mat src;
	Mat dst;
	int scale_x, scale_y;
	const int *ofs, *xofs;
};

template<typename T, typename WT, typename VecOp>
static void resizeAreaFast_(const Mat& src, Mat& dst, const int* ofs, const int* xofs,
	int scale_x, int scale_y)
{
	Range range(0, dst.rows);
	resizeAreaFast_Invoker<T, WT, VecOp> invoker(src, dst, scale_x,
		scale_y, ofs, xofs);
	parallel_for_(range, invoker, dst.total() / (double)(1 << 16));
}

template<typename T, typename SIMDVecOp>
struct ResizeAreaFastVec
{
	ResizeAreaFastVec(int _scale_x, int _scale_y, int _cn, int _step) :
		scale_x(_scale_x), scale_y(_scale_y), cn(_cn), step(_step), vecOp(_cn, _step)
	{
		fast_mode = scale_x == 2 && scale_y == 2 && (cn == 1 || cn == 3 || cn == 4);
	}

	int operator() (const T* S, T* D, int w) const
	{
		if (!fast_mode)
			return 0;

		const T* nextS = (const T*)((const uchar*)S + step);
		int dx = vecOp(S, D, w);

		if (cn == 1)
			for (; dx < w; ++dx)
			{
				int index = dx * 2;
				D[dx] = (T)((S[index] + S[index + 1] + nextS[index] + nextS[index + 1] + 2) >> 2);
			}
		else if (cn == 3)
			for (; dx < w; dx += 3)
			{
				int index = dx * 2;
				D[dx] = (T)((S[index] + S[index + 3] + nextS[index] + nextS[index + 3] + 2) >> 2);
				D[dx + 1] = (T)((S[index + 1] + S[index + 4] + nextS[index + 1] + nextS[index + 4] + 2) >> 2);
				D[dx + 2] = (T)((S[index + 2] + S[index + 5] + nextS[index + 2] + nextS[index + 5] + 2) >> 2);
			}
		else
		{
			CV_Assert(cn == 4);
			for (; dx < w; dx += 4)
			{
				int index = dx * 2;
				D[dx] = (T)((S[index] + S[index + 4] + nextS[index] + nextS[index + 4] + 2) >> 2);
				D[dx + 1] = (T)((S[index + 1] + S[index + 5] + nextS[index + 1] + nextS[index + 5] + 2) >> 2);
				D[dx + 2] = (T)((S[index + 2] + S[index + 6] + nextS[index + 2] + nextS[index + 6] + 2) >> 2);
				D[dx + 3] = (T)((S[index + 3] + S[index + 7] + nextS[index + 3] + nextS[index + 7] + 2) >> 2);
			}
		}

		return dx;
	}

private:
	int scale_x, scale_y;
	int cn;
	bool fast_mode;
	int step;
	SIMDVecOp vecOp;
};


inline bool swapBlue(int code)
{
	switch (code)
	{
	//case COLOR_BGR2BGRA: case COLOR_BGRA2BGR:
	//case COLOR_BGR2BGR565: case COLOR_BGR2BGR555: case COLOR_BGRA2BGR565: case COLOR_BGRA2BGR555:
	//case COLOR_BGR5652BGR: case COLOR_BGR5552BGR: case COLOR_BGR5652BGRA: case COLOR_BGR5552BGRA:
	case COLOR_BGR2GRAY: 
	//case COLOR_BGRA2GRAY:
	//case COLOR_BGR2YCrCb: case COLOR_BGR2YUV:
	//case COLOR_YCrCb2BGR: case COLOR_YUV2BGR:
	//case COLOR_BGR2XYZ: case COLOR_XYZ2BGR:
	//case COLOR_BGR2HSV: case COLOR_BGR2HLS: case COLOR_BGR2HSV_FULL: case COLOR_BGR2HLS_FULL:
	//case COLOR_YUV2BGR_YV12: case COLOR_YUV2BGRA_YV12: case COLOR_YUV2BGR_IYUV: case COLOR_YUV2BGRA_IYUV:
	//case COLOR_YUV2BGR_NV21: case COLOR_YUV2BGRA_NV21: case COLOR_YUV2BGR_NV12: case COLOR_YUV2BGRA_NV12:
	//case COLOR_Lab2BGR: case COLOR_Luv2BGR: case COLOR_Lab2LBGR: case COLOR_Luv2LBGR:
	//case COLOR_BGR2Lab: case COLOR_BGR2Luv: case COLOR_LBGR2Lab: case COLOR_LBGR2Luv:
	//case COLOR_HSV2BGR: case COLOR_HLS2BGR: case COLOR_HSV2BGR_FULL: case COLOR_HLS2BGR_FULL:
	//case COLOR_YUV2BGR_UYVY: case COLOR_YUV2BGRA_UYVY: 
	case COLOR_YUV2BGR_YUY2:
	//case COLOR_YUV2BGRA_YUY2:  case COLOR_YUV2BGR_YVYU: case COLOR_YUV2BGRA_YVYU:
	//case COLOR_BGR2YUV_IYUV: case COLOR_BGRA2YUV_IYUV: case COLOR_BGR2YUV_YV12: case COLOR_BGRA2YUV_YV12:
		return false;
	default:
		return true;
	}
}

inline void cv::cvtColor(InputArray _src, OutputArray _dst, int code, int dcn)
{
	int stype = _src.type();
	int scn = CV_MAT_CN(stype), depth = CV_MAT_DEPTH(stype), uidx, gbits, ycn;

	//CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat() && !(depth == CV_8U && (code == COLOR_Luv2BGR || code == COLOR_Luv2RGB)),
	//	ocl_cvtColor(_src, _dst, code, dcn))

	Mat src, dst;
	if (_src.getObj() == _dst.getObj()) // inplace processing (#6653)
		_src.copyTo(src);
	else
		src = _src.getMat();
	Size sz = src.size();
	CV_Assert(depth == CV_8U || depth == CV_16U || depth == CV_32F);

	switch (code)
	{
	//case COLOR_BGR2BGRA: case COLOR_RGB2BGRA: case COLOR_BGRA2BGR:
	//case COLOR_RGBA2BGR: case COLOR_RGB2BGR: case COLOR_BGRA2RGBA:
	//	CV_Assert(scn == 3 || scn == 4);
	//	dcn = code == COLOR_BGR2BGRA || code == COLOR_RGB2BGRA || code == COLOR_BGRA2RGBA ? 4 : 3;
	//	_dst.create(sz, CV_MAKETYPE(depth, dcn));
	//	dst = _dst.getMat();
	//	hal::cvtBGRtoBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
	//		depth, scn, dcn, swapBlue(code));
	//	break;
	//
	//case COLOR_BGR2BGR565: case COLOR_BGR2BGR555: case COLOR_RGB2BGR565: case COLOR_RGB2BGR555:
	//case COLOR_BGRA2BGR565: case COLOR_BGRA2BGR555: case COLOR_RGBA2BGR565: case COLOR_RGBA2BGR555:
	//	CV_Assert((scn == 3 || scn == 4) && depth == CV_8U);
	//	gbits = code == COLOR_BGR2BGR565 || code == COLOR_RGB2BGR565 ||
	//		code == COLOR_BGRA2BGR565 || code == COLOR_RGBA2BGR565 ? 6 : 5;
	//	_dst.create(sz, CV_8UC2);
	//	dst = _dst.getMat();
	//	hal::cvtBGRtoBGR5x5(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
	//		scn, swapBlue(code), gbits);
	//	break;
	//
	//case COLOR_BGR5652BGR: case COLOR_BGR5552BGR: case COLOR_BGR5652RGB: case COLOR_BGR5552RGB:
	//case COLOR_BGR5652BGRA: case COLOR_BGR5552BGRA: case COLOR_BGR5652RGBA: case COLOR_BGR5552RGBA:
	//	if (dcn <= 0) dcn = (code == COLOR_BGR5652BGRA || code == COLOR_BGR5552BGRA || code == COLOR_BGR5652RGBA || code == COLOR_BGR5552RGBA) ? 4 : 3;
	//	CV_Assert((dcn == 3 || dcn == 4) && scn == 2 && depth == CV_8U);
	//	gbits = code == COLOR_BGR5652BGR || code == COLOR_BGR5652RGB ||
	//		code == COLOR_BGR5652BGRA || code == COLOR_BGR5652RGBA ? 6 : 5;
	//	_dst.create(sz, CV_MAKETYPE(depth, dcn));
	//	dst = _dst.getMat();
	//	hal::cvtBGR5x5toBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
	//		dcn, swapBlue(code), gbits);
	//	break;

	case COLOR_BGR2GRAY: 
	//case COLOR_BGRA2GRAY: 
	case COLOR_RGB2GRAY: 
	//case COLOR_RGBA2GRAY:
		CV_Assert(scn == 3 || scn == 4);
		_dst.create(sz, CV_MAKETYPE(depth, 1));
		dst = _dst.getMat();
		cvtBGRtoGray(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
			depth, scn, swapBlue(code));
		break;

	//case COLOR_BGR5652GRAY: case COLOR_BGR5552GRAY:
	//	CV_Assert(scn == 2 && depth == CV_8U);
	//	gbits = code == COLOR_BGR5652GRAY ? 6 : 5;
	//	_dst.create(sz, CV_8UC1);
	//	dst = _dst.getMat();
	//	hal::cvtBGR5x5toGray(src.data, src.step, dst.data, dst.step, src.cols, src.rows, gbits);
	//	break;

	case COLOR_GRAY2BGR: 
	//case COLOR_GRAY2BGRA:
		//if (dcn <= 0) dcn = (code == COLOR_GRAY2BGRA) ? 4 : 3;
		if (dcn <= 0) dcn = 3;
		CV_Assert(scn == 1 && (dcn == 3 || dcn == 4));
		_dst.create(sz, CV_MAKETYPE(depth, dcn));
		dst = _dst.getMat();
		cvtGraytoBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows, depth, dcn);
		break;

//	case COLOR_GRAY2BGR565: case COLOR_GRAY2BGR555:
//		CV_Assert(scn == 1 && depth == CV_8U);
//		gbits = code == COLOR_GRAY2BGR565 ? 6 : 5;
//		_dst.create(sz, CV_8UC2);
//		dst = _dst.getMat();
//		hal::cvtGraytoBGR5x5(src.data, src.step, dst.data, dst.step, src.cols, src.rows, gbits);
//		break;
//
//	case COLOR_BGR2YCrCb: case COLOR_RGB2YCrCb:
//	case COLOR_BGR2YUV: case COLOR_RGB2YUV:
//		CV_Assert(scn == 3 || scn == 4);
//		_dst.create(sz, CV_MAKETYPE(depth, 3));
//		dst = _dst.getMat();
//		hal::cvtBGRtoYUV(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
//			depth, scn, swapBlue(code), code == COLOR_BGR2YCrCb || code == COLOR_RGB2YCrCb);
//		break;
//
//	case COLOR_YCrCb2BGR: case COLOR_YCrCb2RGB:
//	case COLOR_YUV2BGR: case COLOR_YUV2RGB:
//		if (dcn <= 0) dcn = 3;
//		CV_Assert(scn == 3 && (dcn == 3 || dcn == 4));
//		_dst.create(sz, CV_MAKETYPE(depth, dcn));
//		dst = _dst.getMat();
//		hal::cvtYUVtoBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
//			depth, dcn, swapBlue(code), code == COLOR_YCrCb2BGR || code == COLOR_YCrCb2RGB);
//		break;
//
//	case COLOR_BGR2XYZ: case COLOR_RGB2XYZ:
//		CV_Assert(scn == 3 || scn == 4);
//		_dst.create(sz, CV_MAKETYPE(depth, 3));
//		dst = _dst.getMat();
//		hal::cvtBGRtoXYZ(src.data, src.step, dst.data, dst.step, src.cols, src.rows, depth, scn, swapBlue(code));
//		break;
//
//	case COLOR_XYZ2BGR: case COLOR_XYZ2RGB:
//		if (dcn <= 0) dcn = 3;
//		CV_Assert(scn == 3 && (dcn == 3 || dcn == 4));
//		_dst.create(sz, CV_MAKETYPE(depth, dcn));
//		dst = _dst.getMat();
//		hal::cvtXYZtoBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows, depth, dcn, swapBlue(code));
//		break;
//
//	case COLOR_BGR2HSV: case COLOR_RGB2HSV: case COLOR_BGR2HSV_FULL: case COLOR_RGB2HSV_FULL:
//	case COLOR_BGR2HLS: case COLOR_RGB2HLS: case COLOR_BGR2HLS_FULL: case COLOR_RGB2HLS_FULL:
//		CV_Assert((scn == 3 || scn == 4) && (depth == CV_8U || depth == CV_32F));
//		_dst.create(sz, CV_MAKETYPE(depth, 3));
//		dst = _dst.getMat();
//		hal::cvtBGRtoHSV(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
//			depth, scn, swapBlue(code), isFullRange(code), isHSV(code));
//		break;
//
//	case COLOR_HSV2BGR: case COLOR_HSV2RGB: case COLOR_HSV2BGR_FULL: case COLOR_HSV2RGB_FULL:
//	case COLOR_HLS2BGR: case COLOR_HLS2RGB: case COLOR_HLS2BGR_FULL: case COLOR_HLS2RGB_FULL:
//		if (dcn <= 0) dcn = 3;
//		CV_Assert(scn == 3 && (dcn == 3 || dcn == 4) && (depth == CV_8U || depth == CV_32F));
//		_dst.create(sz, CV_MAKETYPE(depth, dcn));
//		dst = _dst.getMat();
//		hal::cvtHSVtoBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
//			depth, dcn, swapBlue(code), isFullRange(code), isHSV(code));
//		break;
//
//	case COLOR_BGR2Lab: case COLOR_RGB2Lab: case COLOR_LBGR2Lab: case COLOR_LRGB2Lab:
//	case COLOR_BGR2Luv: case COLOR_RGB2Luv: case COLOR_LBGR2Luv: case COLOR_LRGB2Luv:
//		CV_Assert((scn == 3 || scn == 4) && (depth == CV_8U || depth == CV_32F));
//		_dst.create(sz, CV_MAKETYPE(depth, 3));
//		dst = _dst.getMat();
//		hal::cvtBGRtoLab(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
//			depth, scn, swapBlue(code), isLab(code), issRGB(code));
//		break;
//
//	case COLOR_Lab2BGR: case COLOR_Lab2RGB: case COLOR_Lab2LBGR: case COLOR_Lab2LRGB:
//	case COLOR_Luv2BGR: case COLOR_Luv2RGB: case COLOR_Luv2LBGR: case COLOR_Luv2LRGB:
//		if (dcn <= 0) dcn = 3;
//		CV_Assert(scn == 3 && (dcn == 3 || dcn == 4) && (depth == CV_8U || depth == CV_32F));
//		_dst.create(sz, CV_MAKETYPE(depth, dcn));
//		dst = _dst.getMat();
//		hal::cvtLabtoBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
//			depth, dcn, swapBlue(code), isLab(code), issRGB(code));
//		break;
//
//	case COLOR_BayerBG2GRAY: case COLOR_BayerGB2GRAY: case COLOR_BayerRG2GRAY: case COLOR_BayerGR2GRAY:
//	case COLOR_BayerBG2BGR: case COLOR_BayerGB2BGR: case COLOR_BayerRG2BGR: case COLOR_BayerGR2BGR:
//	case COLOR_BayerBG2BGR_VNG: case COLOR_BayerGB2BGR_VNG: case COLOR_BayerRG2BGR_VNG: case COLOR_BayerGR2BGR_VNG:
//	case COLOR_BayerBG2BGR_EA: case COLOR_BayerGB2BGR_EA: case COLOR_BayerRG2BGR_EA: case COLOR_BayerGR2BGR_EA:
//	case COLOR_BayerBG2BGRA: case COLOR_BayerGB2BGRA: case COLOR_BayerRG2BGRA: case COLOR_BayerGR2BGRA:
//		demosaicing(src, _dst, code, dcn);
//		break;
//
//	case COLOR_YUV2BGR_NV21:  case COLOR_YUV2RGB_NV21:  case COLOR_YUV2BGR_NV12:  case COLOR_YUV2RGB_NV12:
//	case COLOR_YUV2BGRA_NV21: case COLOR_YUV2RGBA_NV21: case COLOR_YUV2BGRA_NV12: case COLOR_YUV2RGBA_NV12:
//		// http://www.fourcc.org/yuv.php#NV21 == yuv420sp -> a plane of 8 bit Y samples followed by an interleaved V/U plane containing 8 bit 2x2 subsampled chroma samples
//		// http://www.fourcc.org/yuv.php#NV12 -> a plane of 8 bit Y samples followed by an interleaved U/V plane containing 8 bit 2x2 subsampled colour difference samples
//		if (dcn <= 0) dcn = (code == COLOR_YUV420sp2BGRA || code == COLOR_YUV420sp2RGBA || code == COLOR_YUV2BGRA_NV12 || code == COLOR_YUV2RGBA_NV12) ? 4 : 3;
//		uidx = (code == COLOR_YUV2BGR_NV21 || code == COLOR_YUV2BGRA_NV21 || code == COLOR_YUV2RGB_NV21 || code == COLOR_YUV2RGBA_NV21) ? 1 : 0;
//		CV_Assert(dcn == 3 || dcn == 4);
//		CV_Assert(sz.width % 2 == 0 && sz.height % 3 == 0 && depth == CV_8U);
//		_dst.create(Size(sz.width, sz.height * 2 / 3), CV_MAKETYPE(depth, dcn));
//		dst = _dst.getMat();
//		hal::cvtTwoPlaneYUVtoBGR(src.data, src.step, dst.data, dst.step, dst.cols, dst.rows,
//			dcn, swapBlue(code), uidx);
//		break;
//	case COLOR_YUV2BGR_YV12: case COLOR_YUV2RGB_YV12: case COLOR_YUV2BGRA_YV12: case COLOR_YUV2RGBA_YV12:
//	case COLOR_YUV2BGR_IYUV: case COLOR_YUV2RGB_IYUV: case COLOR_YUV2BGRA_IYUV: case COLOR_YUV2RGBA_IYUV:
//		//http://www.fourcc.org/yuv.php#YV12 == yuv420p -> It comprises an NxM Y plane followed by (N/2)x(M/2) V and U planes.
//		//http://www.fourcc.org/yuv.php#IYUV == I420 -> It comprises an NxN Y plane followed by (N/2)x(N/2) U and V planes
//		if (dcn <= 0) dcn = (code == COLOR_YUV2BGRA_YV12 || code == COLOR_YUV2RGBA_YV12 || code == COLOR_YUV2RGBA_IYUV || code == COLOR_YUV2BGRA_IYUV) ? 4 : 3;
//		uidx = (code == COLOR_YUV2BGR_YV12 || code == COLOR_YUV2RGB_YV12 || code == COLOR_YUV2BGRA_YV12 || code == COLOR_YUV2RGBA_YV12) ? 1 : 0;
//		CV_Assert(dcn == 3 || dcn == 4);
//		CV_Assert(sz.width % 2 == 0 && sz.height % 3 == 0 && depth == CV_8U);
//		_dst.create(Size(sz.width, sz.height * 2 / 3), CV_MAKETYPE(depth, dcn));
//		dst = _dst.getMat();
//		hal::cvtThreePlaneYUVtoBGR(src.data, src.step, dst.data, dst.step, dst.cols, dst.rows,
//			dcn, swapBlue(code), uidx);
//		break;
//
//	case COLOR_YUV2GRAY_420:
//	{
//		if (dcn <= 0) dcn = 1;
//
//		CV_Assert(dcn == 1);
//		CV_Assert(sz.width % 2 == 0 && sz.height % 3 == 0 && depth == CV_8U);
//
//		Size dstSz(sz.width, sz.height * 2 / 3);
//		_dst.create(dstSz, CV_MAKETYPE(depth, dcn));
//		dst = _dst.getMat();
//#ifdef HAVE_IPP
//#if IPP_VERSION_X100 >= 201700
//		if (CV_INSTRUMENT_FUN_IPP(ippiCopy_8u_C1R_L, src.data, (IppSizeL)src.step, dst.data, (IppSizeL)dst.step,
//			ippiSizeL(dstSz.width, dstSz.height)) >= 0)
//			break;
//#endif
//#endif
//		src(Range(0, dstSz.height), Range::all()).copyTo(dst);
//	}
//	break;
//	case COLOR_RGB2YUV_YV12: case COLOR_BGR2YUV_YV12: case COLOR_RGBA2YUV_YV12: case COLOR_BGRA2YUV_YV12:
//	case COLOR_RGB2YUV_IYUV: case COLOR_BGR2YUV_IYUV: case COLOR_RGBA2YUV_IYUV: case COLOR_BGRA2YUV_IYUV:
//		if (dcn <= 0) dcn = 1;
//		uidx = (code == COLOR_BGR2YUV_IYUV || code == COLOR_BGRA2YUV_IYUV || code == COLOR_RGB2YUV_IYUV || code == COLOR_RGBA2YUV_IYUV) ? 1 : 2;
//		CV_Assert((scn == 3 || scn == 4) && depth == CV_8U);
//		CV_Assert(dcn == 1);
//		CV_Assert(sz.width % 2 == 0 && sz.height % 2 == 0);
//		_dst.create(Size(sz.width, sz.height / 2 * 3), CV_MAKETYPE(depth, dcn));
//		dst = _dst.getMat();
//		hal::cvtBGRtoThreePlaneYUV(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
//			scn, swapBlue(code), uidx);
//		break;
//	case COLOR_YUV2RGB_UYVY: case COLOR_YUV2BGR_UYVY: case COLOR_YUV2RGBA_UYVY: case COLOR_YUV2BGRA_UYVY:
//	case COLOR_YUV2RGB_YUY2:
 case COLOR_YUV2BGR_YUY2:
// case COLOR_YUV2RGB_YVYU: case COLOR_YUV2BGR_YVYU:
//	case COLOR_YUV2RGBA_YUY2: case COLOR_YUV2BGRA_YUY2: case COLOR_YUV2RGBA_YVYU: case COLOR_YUV2BGRA_YVYU:
//		//http://www.fourcc.org/yuv.php#UYVY
//		//http://www.fourcc.org/yuv.php#YUY2
//		//http://www.fourcc.org/yuv.php#YVYU
//		if (dcn <= 0) dcn = (code == COLOR_YUV2RGBA_UYVY || code == COLOR_YUV2BGRA_UYVY || code == COLOR_YUV2RGBA_YUY2 || code == COLOR_YUV2BGRA_YUY2 || code == COLOR_YUV2RGBA_YVYU || code == COLOR_YUV2BGRA_YVYU) ? 4 : 3;
//		ycn = (code == COLOR_YUV2RGB_UYVY || code == COLOR_YUV2BGR_UYVY || code == COLOR_YUV2RGBA_UYVY || code == COLOR_YUV2BGRA_UYVY) ? 1 : 0;
//		uidx = (code == COLOR_YUV2RGB_YVYU || code == COLOR_YUV2BGR_YVYU || code == COLOR_YUV2RGBA_YVYU || code == COLOR_YUV2BGRA_YVYU) ? 1 : 0;
		 if (dcn <= 0) dcn = 3;
		 ycn = 0;
		 uidx = 0;
		 CV_Assert(dcn == 3 || dcn == 4);
		CV_Assert(scn == 2 && depth == CV_8U);
		_dst.create(sz, CV_8UC(dcn));
		dst = _dst.getMat();
		//hal::cvtOnePlaneYUVtoBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
		//	dcn, swapBlue(code), uidx, ycn);
		cvtOnePlaneYUVtoBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
			dcn, swapBlue(code), uidx, ycn);
		break;
//	case COLOR_YUV2GRAY_UYVY: case COLOR_YUV2GRAY_YUY2:
//	{
//		if (dcn <= 0) dcn = 1;
//
//		CV_Assert(dcn == 1);
//		CV_Assert(scn == 2 && depth == CV_8U);
//
//		src.release(); // T-API datarace fixup
//		extractChannel(_src, _dst, code == COLOR_YUV2GRAY_UYVY ? 1 : 0);
//	}
//	break;
//	case COLOR_RGBA2mRGBA:
//		if (dcn <= 0) dcn = 4;
//		CV_Assert(scn == 4 && dcn == 4 && depth == CV_8U);
//		_dst.create(sz, CV_MAKETYPE(depth, dcn));
//		dst = _dst.getMat();
//		hal::cvtRGBAtoMultipliedRGBA(src.data, src.step, dst.data, dst.step, src.cols, src.rows);
//		break;
//	case COLOR_mRGBA2RGBA:
//		if (dcn <= 0) dcn = 4;
//		CV_Assert(scn == 4 && dcn == 4 && depth == CV_8U);
//		_dst.create(sz, CV_MAKETYPE(depth, dcn));
//		dst = _dst.getMat();
//		hal::cvtMultipliedRGBAtoRGBA(src.data, src.step, dst.data, dst.step, src.cols, src.rows);
//		break;
	default:
		CV_Error(CV_StsBadFlag, "Unknown/unsupported color conversion code");
	}
}

// 8u, 16u, 32f
inline void cv::cvtBGRtoGray(const uchar * src_data, size_t src_step,
	uchar * dst_data, size_t dst_step,
	int width, int height,
	int depth, int scn, bool swapBlue)
{
	//CV_INSTRUMENT_REGION()
	//CALL_HAL(cvtBGRtoGray, cv_hal_cvtBGRtoGray, src_data, src_step, dst_data, dst_step, width, height, depth, scn, swapBlue);

	int blueIdx = swapBlue ? 2 : 0;
	if (depth == CV_8U)
		CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Gray<uchar>(scn, blueIdx, 0));
	else if (depth == CV_16U)
		CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Gray<ushort>(scn, blueIdx, 0));
	else
		CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Gray<float>(scn, blueIdx, 0));
}

// 8u, 16u, 32f
inline void cv::cvtGraytoBGR(const uchar * src_data, size_t src_step,
	uchar * dst_data, size_t dst_step,
	int width, int height,
	int depth, int dcn)
{
	//CV_INSTRUMENT_REGION()
	//CALL_HAL(cvtGraytoBGR, cv_hal_cvtGraytoBGR, src_data, src_step, dst_data, dst_step, width, height, depth, dcn);

	if (depth == CV_8U)
		CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Gray2RGB<uchar>(dcn));
	else if (depth == CV_16U)
		CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Gray2RGB<ushort>(dcn));
	else
		CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Gray2RGB<float>(dcn));
}

///////////////////////////////////// YUV422 -> RGB /////////////////////////////////////

template<int bIdx, int uIdx, int yIdx>
struct YUV422toRGB888Invoker : ParallelLoopBody
{
	uchar * dst_data;
	size_t dst_step;
	const uchar * src_data;
	size_t src_step;
	int width;

	YUV422toRGB888Invoker(uchar * _dst_data, size_t _dst_step,
		const uchar * _src_data, size_t _src_step,
		int _width)
		: dst_data(_dst_data), dst_step(_dst_step), src_data(_src_data), src_step(_src_step), width(_width) {}

	void operator()(const Range& range) const
	{
		int rangeBegin = range.start;
		int rangeEnd = range.end;

		const int uidx = 1 - yIdx + uIdx * 2;
		const int vidx = (2 + uidx) % 4;
		const uchar* yuv_src = src_data + rangeBegin * src_step;

		for (int j = rangeBegin; j < rangeEnd; j++, yuv_src += src_step)
		{
			uchar* row = dst_data + dst_step * j;

			for (int i = 0; i < 2 * width; i += 4, row += 6)
			{
				int u = int(yuv_src[i + uidx]) - 128;
				int v = int(yuv_src[i + vidx]) - 128;

				int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
				int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
				int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

				int y00 = std::max(0, int(yuv_src[i + yIdx]) - 16) * ITUR_BT_601_CY;
				row[2 - bIdx] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
				row[1] = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
				row[bIdx] = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);

				int y01 = std::max(0, int(yuv_src[i + yIdx + 2]) - 16) * ITUR_BT_601_CY;
				row[5 - bIdx] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
				row[4] = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
				row[3 + bIdx] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);
			}
		}
	}
};

#define MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION (320*240)

template<int bIdx, int uIdx, int yIdx>
inline void cvtYUV422toRGB(uchar * dst_data, size_t dst_step, const uchar * src_data, size_t src_step,
	int width, int height)
{
	YUV422toRGB888Invoker<bIdx, uIdx, yIdx> converter(dst_data, dst_step, src_data, src_step, width);
	if (width * height >= MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION)
		parallel_for_(Range(0, height), converter);
	else
		converter(Range(0, height));
}

inline void cv::cvtOnePlaneYUVtoBGR(const uchar * src_data, size_t src_step,
	uchar * dst_data, size_t dst_step,
	int width, int height,
	int dcn, bool swapBlue, int uIdx, int ycn)
{
	//CV_INSTRUMENT_REGION()

	//CALL_HAL(cvtOnePlaneYUVtoBGR, cv_hal_cvtOnePlaneYUVtoBGR, src_data, src_step, dst_data, dst_step, width, height, dcn, swapBlue, uIdx, ycn);
	int blueIdx = swapBlue ? 2 : 0;
	switch (dcn * 1000 + blueIdx * 100 + uIdx * 10 + ycn)
	{
	case 3000: cvtYUV422toRGB<0, 0, 0>(dst_data, dst_step, src_data, src_step, width, height); break;
	case 3001: cvtYUV422toRGB<0, 0, 1>(dst_data, dst_step, src_data, src_step, width, height); break;
	case 3010: cvtYUV422toRGB<0, 1, 0>(dst_data, dst_step, src_data, src_step, width, height); break;
	case 3200: cvtYUV422toRGB<2, 0, 0>(dst_data, dst_step, src_data, src_step, width, height); break;
	case 3201: cvtYUV422toRGB<2, 0, 1>(dst_data, dst_step, src_data, src_step, width, height); break;
	case 3210: cvtYUV422toRGB<2, 1, 0>(dst_data, dst_step, src_data, src_step, width, height); break;
		//case 4000: cvtYUV422toRGBA<0,0,0>(dst_data, dst_step, src_data, src_step, width, height); break;
		//case 4001: cvtYUV422toRGBA<0,0,1>(dst_data, dst_step, src_data, src_step, width, height); break;
		//case 4010: cvtYUV422toRGBA<0,1,0>(dst_data, dst_step, src_data, src_step, width, height); break;
		//case 4200: cvtYUV422toRGBA<2,0,0>(dst_data, dst_step, src_data, src_step, width, height); break;
		//case 4201: cvtYUV422toRGBA<2,0,1>(dst_data, dst_step, src_data, src_step, width, height); break;
		//case 4210: cvtYUV422toRGBA<2,1,0>(dst_data, dst_step, src_data, src_step, width, height); break;
	default: CV_Error(CV_StsBadFlag, "Unknown/unsupported color conversion code"); break;
	};
}

inline void cv::resize(InputArray _src, OutputArray _dst, Size dsize,
	double inv_scale_x, double inv_scale_y, int interpolation)
{
	Size ssize = _src.size();

	CV_Assert(ssize.width > 0 && ssize.height > 0);
	CV_Assert(dsize.area() > 0 || (inv_scale_x > 0 && inv_scale_y > 0));

	//如果dsize大小为0则dsize设置为原图大小乘以传进来的比例值（inv_scale_x，inv_scale_y）否则
	//传进来的比例值设置为dsize除以原图大小（width, height）
	if (dsize.area() == 0)
	{
		dsize = Size(saturate_cast<int>(ssize.width * inv_scale_x),
			saturate_cast<int>(ssize.height * inv_scale_y));
		CV_Assert(dsize.area() > 0);
	}
	else
	{
		inv_scale_x = (double)dsize.width / ssize.width;
		inv_scale_y = (double)dsize.height / ssize.height;
	}

	Mat src = _src.getMat();
	_dst.create(dsize, src.type());
	Mat dst = _dst.getMat();

	if (dsize == ssize)
	{
		// Source and destination are of same size. Use simple copy.
		src.copyTo(dst);
		return;
	}

	halResize(src.type(), src.data, src.step, src.cols, src.rows, dst.data, dst.step, dst.cols, dst.rows, inv_scale_x, inv_scale_y, interpolation);
}

typedef void(*ResizeFunc)(const Mat& src, Mat& dst,
	const int* xofs, const void* alpha,
	const int* yofs, const void* beta,
	int xmin, int xmax, int ksize);
typedef void(*ResizeAreaFastFunc)(const Mat& src, Mat& dst,
	const int* ofs, const int *xofs,
	int scale_x, int scale_y);

//typedef void(*ResizeAreaFunc)(const Mat& src, Mat& dst,
//	const DecimateAlpha* xtab, int xtab_size,
//	const DecimateAlpha* ytab, int ytab_size,
//	const int* yofs);

static inline int clip(int x, int a, int b)
{
	return x >= a ? (x < b ? x : b - 1) : a;
}


template <typename HResize, typename VResize>
class resizeGeneric_Invoker :
	public ParallelLoopBody
{
public:
	typedef typename HResize::value_type T;
	typedef typename HResize::buf_type WT;
	typedef typename HResize::alpha_type AT;

	resizeGeneric_Invoker(const Mat& _src, Mat &_dst, const int *_xofs, const int *_yofs,
		const AT* _alpha, const AT* __beta, const Size& _ssize, const Size &_dsize,
		int _ksize, int _xmin, int _xmax) :
		ParallelLoopBody(), src(_src), dst(_dst), xofs(_xofs), yofs(_yofs),
		alpha(_alpha), _beta(__beta), ssize(_ssize), dsize(_dsize),
		ksize(_ksize), xmin(_xmin), xmax(_xmax)
	{
		CV_Assert(ksize <= MAX_ESIZE);
	}

	virtual void operator() (const Range& range) const
	{
		int dy, cn = src.channels();
		HResize hresize;
		VResize vresize;

		int bufstep = (int)alignSize(dsize.width, 16);
		AutoBuffer<WT> _buffer(bufstep*ksize);
		const T* srows[MAX_ESIZE] = { 0 };
		WT* rows[MAX_ESIZE] = { 0 };
		int prev_sy[MAX_ESIZE];

		for (int k = 0; k < ksize; k++)
		{
			prev_sy[k] = -1;
			rows[k] = (WT*)_buffer + bufstep*k;
		}

		const AT* beta = _beta + ksize * range.start;

		for (dy = range.start; dy < range.end; dy++, beta += ksize)
		{
			int sy0 = yofs[dy], k0 = ksize, k1 = 0, ksize2 = ksize / 2;

			for (int k = 0; k < ksize; k++)
			{
				int sy = clip(sy0 - ksize2 + 1 + k, 0, ssize.height);
				for (k1 = std::max(k1, k); k1 < ksize; k1++)
				{
					if (k1 < MAX_ESIZE && sy == prev_sy[k1]) // if the sy-th row has been computed already, reuse it.
					{
						if (k1 > k)
							memcpy(rows[k], rows[k1], bufstep*sizeof(rows[0][0]));
						break;
					}
				}
				if (k1 == ksize)
					k0 = std::min(k0, k); // remember the first row that needs to be computed
				srows[k] = src.template ptr<T>(sy);
				prev_sy[k] = sy;
			}

			if (k0 < ksize)
				hresize((const T**)(srows + k0), (WT**)(rows + k0), ksize - k0, xofs, (const AT*)(alpha),
				ssize.width, dsize.width, cn, xmin, xmax);
			vresize((const WT**)rows, (T*)(dst.data + dst.step*dy), beta, dsize.width);
		}
	}

private:
	Mat src;
	Mat dst;
	const int* xofs, *yofs;
	const AT* alpha, *_beta;
	Size ssize, dsize;
	const int ksize, xmin, xmax;

	resizeGeneric_Invoker& operator = (const resizeGeneric_Invoker&);
};

template<class HResize, class VResize>
static void resizeGeneric_(const Mat& src, Mat& dst,
	const int* xofs, const void* _alpha,
	const int* yofs, const void* _beta,
	int xmin, int xmax, int ksize)
{
	typedef typename HResize::alpha_type AT;

	const AT* beta = (const AT*)_beta;
	Size ssize = src.size(), dsize = dst.size();
	int cn = src.channels();
	ssize.width *= cn;
	dsize.width *= cn;
	xmin *= cn;
	xmax *= cn;
	// image resize is a separable operation. In case of not too strong

	Range range(0, dsize.height);
	resizeGeneric_Invoker<HResize, VResize> invoker(src, dst, xofs, yofs, (const AT*)_alpha, beta,
		ssize, dsize, ksize, xmin, xmax);
	parallel_for_(range, invoker, dst.total() / (double)(1 << 16));
}

struct HResizeNoVec
{
	int operator()(const uchar**, uchar**, int, const int*,
		const uchar*, int, int, int, int, int) const {
		return 0;
	}
};

struct VResizeNoVec
{
	int operator()(const uchar**, uchar*, const uchar*, int) const { return 0; }
};

template<typename T, typename WT, typename AT, int ONE, class VecOp>
struct HResizeLinear
{
	typedef T value_type;
	typedef WT buf_type;
	typedef AT alpha_type;

	void operator()(const T** src, WT** dst, int count,
		const int* xofs, const AT* alpha,
		int swidth, int dwidth, int cn, int xmin, int xmax) const
	{
		int dx, k;
		VecOp vecOp;

		int dx0 = vecOp((const uchar**)src, (uchar**)dst, count,
			xofs, (const uchar*)alpha, swidth, dwidth, cn, xmin, xmax);

		for (k = 0; k <= count - 2; k++)
		{
			const T *S0 = src[k], *S1 = src[k + 1];
			WT *D0 = dst[k], *D1 = dst[k + 1];
			for (dx = dx0; dx < xmax; dx++)
			{
				int sx = xofs[dx];
				WT a0 = alpha[dx * 2], a1 = alpha[dx * 2 + 1];
				WT t0 = S0[sx] * a0 + S0[sx + cn] * a1;
				WT t1 = S1[sx] * a0 + S1[sx + cn] * a1;
				D0[dx] = t0; D1[dx] = t1;
			}

			for (; dx < dwidth; dx++)
			{
				int sx = xofs[dx];
				D0[dx] = WT(S0[sx] * ONE); D1[dx] = WT(S1[sx] * ONE);
			}
		}

		for (; k < count; k++)
		{
			const T *S = src[k];
			WT *D = dst[k];
			for (dx = 0; dx < xmax; dx++)
			{
				int sx = xofs[dx];
				D[dx] = S[sx] * alpha[dx * 2] + S[sx + cn] * alpha[dx * 2 + 1];
			}

			for (; dx < dwidth; dx++)
				D[dx] = WT(S[xofs[dx]] * ONE);
		}
	}
};

template<typename T, typename WT, typename AT, class CastOp, class VecOp>
struct VResizeLinear
{
	typedef T value_type;
	typedef WT buf_type;
	typedef AT alpha_type;

	void operator()(const WT** src, T* dst, const AT* beta, int width) const
	{
		WT b0 = beta[0], b1 = beta[1];
		const WT *S0 = src[0], *S1 = src[1];
		CastOp castOp;
		VecOp vecOp;

		int x = vecOp((const uchar**)src, (uchar*)dst, (const uchar*)beta, width);
#if CV_ENABLE_UNROLLED
		for (; x <= width - 4; x += 4)
		{
			WT t0, t1;
			t0 = S0[x] * b0 + S1[x] * b1;
			t1 = S0[x + 1] * b0 + S1[x + 1] * b1;
			dst[x] = castOp(t0); dst[x + 1] = castOp(t1);
			t0 = S0[x + 2] * b0 + S1[x + 2] * b1;
			t1 = S0[x + 3] * b0 + S1[x + 3] * b1;
			dst[x + 2] = castOp(t0); dst[x + 3] = castOp(t1);
		}
#endif
		for (; x < width; x++)
			dst[x] = castOp(S0[x] * b0 + S1[x] * b1);
	}
};

template<typename ST, typename DT, int bits> struct FixedPtCast
{
	typedef ST type1;
	typedef DT rtype;
	enum { SHIFT = bits, DELTA = 1 << (bits - 1) };

	DT operator()(ST val) const { return saturate_cast<DT>((val + DELTA) >> SHIFT); }
};

#if CV_SSE2_

struct VResizeLinearVec_32s8u
{
	int operator()(const uchar** _src, uchar* dst, const uchar* _beta, int width) const
	{
		if (!checkHardwareSupport(CV_CPU_SSE2))
			return 0;

		const int** src = (const int**)_src;
		const short* beta = (const short*)_beta;
		const int *S0 = src[0], *S1 = src[1];
		int x = 0;
		__m128i b0 = _mm_set1_epi16(beta[0]), b1 = _mm_set1_epi16(beta[1]);
		__m128i delta = _mm_set1_epi16(2);

		if ((((size_t)S0 | (size_t)S1) & 15) == 0)
			for (; x <= width - 16; x += 16)
			{
				__m128i x0, x1, x2, y0, y1, y2;
				x0 = _mm_load_si128((const __m128i*)(S0 + x));
				x1 = _mm_load_si128((const __m128i*)(S0 + x + 4));
				y0 = _mm_load_si128((const __m128i*)(S1 + x));
				y1 = _mm_load_si128((const __m128i*)(S1 + x + 4));
				x0 = _mm_packs_epi32(_mm_srai_epi32(x0, 4), _mm_srai_epi32(x1, 4));
				y0 = _mm_packs_epi32(_mm_srai_epi32(y0, 4), _mm_srai_epi32(y1, 4));

				x1 = _mm_load_si128((const __m128i*)(S0 + x + 8));
				x2 = _mm_load_si128((const __m128i*)(S0 + x + 12));
				y1 = _mm_load_si128((const __m128i*)(S1 + x + 8));
				y2 = _mm_load_si128((const __m128i*)(S1 + x + 12));
				x1 = _mm_packs_epi32(_mm_srai_epi32(x1, 4), _mm_srai_epi32(x2, 4));
				y1 = _mm_packs_epi32(_mm_srai_epi32(y1, 4), _mm_srai_epi32(y2, 4));

				x0 = _mm_adds_epi16(_mm_mulhi_epi16(x0, b0), _mm_mulhi_epi16(y0, b1));
				x1 = _mm_adds_epi16(_mm_mulhi_epi16(x1, b0), _mm_mulhi_epi16(y1, b1));

				x0 = _mm_srai_epi16(_mm_adds_epi16(x0, delta), 2);
				x1 = _mm_srai_epi16(_mm_adds_epi16(x1, delta), 2);
				_mm_storeu_si128((__m128i*)(dst + x), _mm_packus_epi16(x0, x1));
			}
		else
			for (; x <= width - 16; x += 16)
			{
				__m128i x0, x1, x2, y0, y1, y2;
				x0 = _mm_loadu_si128((const __m128i*)(S0 + x));
				x1 = _mm_loadu_si128((const __m128i*)(S0 + x + 4));
				y0 = _mm_loadu_si128((const __m128i*)(S1 + x));
				y1 = _mm_loadu_si128((const __m128i*)(S1 + x + 4));
				x0 = _mm_packs_epi32(_mm_srai_epi32(x0, 4), _mm_srai_epi32(x1, 4));
				y0 = _mm_packs_epi32(_mm_srai_epi32(y0, 4), _mm_srai_epi32(y1, 4));

				x1 = _mm_loadu_si128((const __m128i*)(S0 + x + 8));
				x2 = _mm_loadu_si128((const __m128i*)(S0 + x + 12));
				y1 = _mm_loadu_si128((const __m128i*)(S1 + x + 8));
				y2 = _mm_loadu_si128((const __m128i*)(S1 + x + 12));
				x1 = _mm_packs_epi32(_mm_srai_epi32(x1, 4), _mm_srai_epi32(x2, 4));
				y1 = _mm_packs_epi32(_mm_srai_epi32(y1, 4), _mm_srai_epi32(y2, 4));

				x0 = _mm_adds_epi16(_mm_mulhi_epi16(x0, b0), _mm_mulhi_epi16(y0, b1));
				x1 = _mm_adds_epi16(_mm_mulhi_epi16(x1, b0), _mm_mulhi_epi16(y1, b1));

				x0 = _mm_srai_epi16(_mm_adds_epi16(x0, delta), 2);
				x1 = _mm_srai_epi16(_mm_adds_epi16(x1, delta), 2);
				_mm_storeu_si128((__m128i*)(dst + x), _mm_packus_epi16(x0, x1));
			}

		for (; x < width - 4; x += 4)
		{
			__m128i x0, y0;
			x0 = _mm_srai_epi32(_mm_loadu_si128((const __m128i*)(S0 + x)), 4);
			y0 = _mm_srai_epi32(_mm_loadu_si128((const __m128i*)(S1 + x)), 4);
			x0 = _mm_packs_epi32(x0, x0);
			y0 = _mm_packs_epi32(y0, y0);
			x0 = _mm_adds_epi16(_mm_mulhi_epi16(x0, b0), _mm_mulhi_epi16(y0, b1));
			x0 = _mm_srai_epi16(_mm_adds_epi16(x0, delta), 2);
			x0 = _mm_packus_epi16(x0, x0);
			*(int*)(dst + x) = _mm_cvtsi128_si32(x0);
		}

		return x;
	}
};


template<int shiftval> struct VResizeLinearVec_32f16
{
	int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
	{
		if (!checkHardwareSupport(CV_CPU_SSE2))
			return 0;

		const float** src = (const float**)_src;
		const float* beta = (const float*)_beta;
		const float *S0 = src[0], *S1 = src[1];
		ushort* dst = (ushort*)_dst;
		int x = 0;

		__m128 b0 = _mm_set1_ps(beta[0]), b1 = _mm_set1_ps(beta[1]);
		__m128i preshift = _mm_set1_epi32(shiftval);
		__m128i postshift = _mm_set1_epi16((short)shiftval);

		if ((((size_t)S0 | (size_t)S1) & 15) == 0)
			for (; x <= width - 16; x += 16)
			{
				__m128 x0, x1, y0, y1;
				__m128i t0, t1, t2;
				x0 = _mm_load_ps(S0 + x);
				x1 = _mm_load_ps(S0 + x + 4);
				y0 = _mm_load_ps(S1 + x);
				y1 = _mm_load_ps(S1 + x + 4);

				x0 = _mm_add_ps(_mm_mul_ps(x0, b0), _mm_mul_ps(y0, b1));
				x1 = _mm_add_ps(_mm_mul_ps(x1, b0), _mm_mul_ps(y1, b1));
				t0 = _mm_add_epi32(_mm_cvtps_epi32(x0), preshift);
				t2 = _mm_add_epi32(_mm_cvtps_epi32(x1), preshift);
				t0 = _mm_add_epi16(_mm_packs_epi32(t0, t2), postshift);

				x0 = _mm_load_ps(S0 + x + 8);
				x1 = _mm_load_ps(S0 + x + 12);
				y0 = _mm_load_ps(S1 + x + 8);
				y1 = _mm_load_ps(S1 + x + 12);

				x0 = _mm_add_ps(_mm_mul_ps(x0, b0), _mm_mul_ps(y0, b1));
				x1 = _mm_add_ps(_mm_mul_ps(x1, b0), _mm_mul_ps(y1, b1));
				t1 = _mm_add_epi32(_mm_cvtps_epi32(x0), preshift);
				t2 = _mm_add_epi32(_mm_cvtps_epi32(x1), preshift);
				t1 = _mm_add_epi16(_mm_packs_epi32(t1, t2), postshift);

				_mm_storeu_si128((__m128i*)(dst + x), t0);
				_mm_storeu_si128((__m128i*)(dst + x + 8), t1);
			}
		else
			for (; x <= width - 16; x += 16)
			{
				__m128 x0, x1, y0, y1;
				__m128i t0, t1, t2;
				x0 = _mm_loadu_ps(S0 + x);
				x1 = _mm_loadu_ps(S0 + x + 4);
				y0 = _mm_loadu_ps(S1 + x);
				y1 = _mm_loadu_ps(S1 + x + 4);

				x0 = _mm_add_ps(_mm_mul_ps(x0, b0), _mm_mul_ps(y0, b1));
				x1 = _mm_add_ps(_mm_mul_ps(x1, b0), _mm_mul_ps(y1, b1));
				t0 = _mm_add_epi32(_mm_cvtps_epi32(x0), preshift);
				t2 = _mm_add_epi32(_mm_cvtps_epi32(x1), preshift);
				t0 = _mm_add_epi16(_mm_packs_epi32(t0, t2), postshift);

				x0 = _mm_loadu_ps(S0 + x + 8);
				x1 = _mm_loadu_ps(S0 + x + 12);
				y0 = _mm_loadu_ps(S1 + x + 8);
				y1 = _mm_loadu_ps(S1 + x + 12);

				x0 = _mm_add_ps(_mm_mul_ps(x0, b0), _mm_mul_ps(y0, b1));
				x1 = _mm_add_ps(_mm_mul_ps(x1, b0), _mm_mul_ps(y1, b1));
				t1 = _mm_add_epi32(_mm_cvtps_epi32(x0), preshift);
				t2 = _mm_add_epi32(_mm_cvtps_epi32(x1), preshift);
				t1 = _mm_add_epi16(_mm_packs_epi32(t1, t2), postshift);

				_mm_storeu_si128((__m128i*)(dst + x), t0);
				_mm_storeu_si128((__m128i*)(dst + x + 8), t1);
			}

		for (; x < width - 4; x += 4)
		{
			__m128 x0, y0;
			__m128i t0;
			x0 = _mm_loadu_ps(S0 + x);
			y0 = _mm_loadu_ps(S1 + x);

			x0 = _mm_add_ps(_mm_mul_ps(x0, b0), _mm_mul_ps(y0, b1));
			t0 = _mm_add_epi32(_mm_cvtps_epi32(x0), preshift);
			t0 = _mm_add_epi16(_mm_packs_epi32(t0, t0), postshift);
			_mm_storel_epi64((__m128i*)(dst + x), t0);
		}

		return x;
	}
};

typedef VResizeLinearVec_32f16<SHRT_MIN> VResizeLinearVec_32f16u;
typedef VResizeLinearVec_32f16<0> VResizeLinearVec_32f16s;

struct VResizeLinearVec_32f
{
	int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
	{
		if (!checkHardwareSupport(CV_CPU_SSE))
			return 0;

		const float** src = (const float**)_src;
		const float* beta = (const float*)_beta;
		const float *S0 = src[0], *S1 = src[1];
		float* dst = (float*)_dst;
		int x = 0;

		__m128 b0 = _mm_set1_ps(beta[0]), b1 = _mm_set1_ps(beta[1]);

		if ((((size_t)S0 | (size_t)S1) & 15) == 0)
			for (; x <= width - 8; x += 8)
			{
				__m128 x0, x1, y0, y1;
				x0 = _mm_load_ps(S0 + x);
				x1 = _mm_load_ps(S0 + x + 4);
				y0 = _mm_load_ps(S1 + x);
				y1 = _mm_load_ps(S1 + x + 4);

				x0 = _mm_add_ps(_mm_mul_ps(x0, b0), _mm_mul_ps(y0, b1));
				x1 = _mm_add_ps(_mm_mul_ps(x1, b0), _mm_mul_ps(y1, b1));

				_mm_storeu_ps(dst + x, x0);
				_mm_storeu_ps(dst + x + 4, x1);
			}
		else
			for (; x <= width - 8; x += 8)
			{
				__m128 x0, x1, y0, y1;
				x0 = _mm_loadu_ps(S0 + x);
				x1 = _mm_loadu_ps(S0 + x + 4);
				y0 = _mm_loadu_ps(S1 + x);
				y1 = _mm_loadu_ps(S1 + x + 4);

				x0 = _mm_add_ps(_mm_mul_ps(x0, b0), _mm_mul_ps(y0, b1));
				x1 = _mm_add_ps(_mm_mul_ps(x1, b0), _mm_mul_ps(y1, b1));

				_mm_storeu_ps(dst + x, x0);
				_mm_storeu_ps(dst + x + 4, x1);
			}

		return x;
	}
};


struct VResizeCubicVec_32s8u
{
	int operator()(const uchar** _src, uchar* dst, const uchar* _beta, int width) const
	{
		if (!checkHardwareSupport(CV_CPU_SSE2))
			return 0;

		const int** src = (const int**)_src;
		const short* beta = (const short*)_beta;
		const int *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
		int x = 0;
		float scale = 1.f / (INTER_RESIZE_COEF_SCALE*INTER_RESIZE_COEF_SCALE);
		__m128 b0 = _mm_set1_ps(beta[0] * scale), b1 = _mm_set1_ps(beta[1] * scale),
			b2 = _mm_set1_ps(beta[2] * scale), b3 = _mm_set1_ps(beta[3] * scale);

		if ((((size_t)S0 | (size_t)S1 | (size_t)S2 | (size_t)S3) & 15) == 0)
			for (; x <= width - 8; x += 8)
			{
				__m128i x0, x1, y0, y1;
				__m128 s0, s1, f0, f1;
				x0 = _mm_load_si128((const __m128i*)(S0 + x));
				x1 = _mm_load_si128((const __m128i*)(S0 + x + 4));
				y0 = _mm_load_si128((const __m128i*)(S1 + x));
				y1 = _mm_load_si128((const __m128i*)(S1 + x + 4));

				s0 = _mm_mul_ps(_mm_cvtepi32_ps(x0), b0);
				s1 = _mm_mul_ps(_mm_cvtepi32_ps(x1), b0);
				f0 = _mm_mul_ps(_mm_cvtepi32_ps(y0), b1);
				f1 = _mm_mul_ps(_mm_cvtepi32_ps(y1), b1);
				s0 = _mm_add_ps(s0, f0);
				s1 = _mm_add_ps(s1, f1);

				x0 = _mm_load_si128((const __m128i*)(S2 + x));
				x1 = _mm_load_si128((const __m128i*)(S2 + x + 4));
				y0 = _mm_load_si128((const __m128i*)(S3 + x));
				y1 = _mm_load_si128((const __m128i*)(S3 + x + 4));

				f0 = _mm_mul_ps(_mm_cvtepi32_ps(x0), b2);
				f1 = _mm_mul_ps(_mm_cvtepi32_ps(x1), b2);
				s0 = _mm_add_ps(s0, f0);
				s1 = _mm_add_ps(s1, f1);
				f0 = _mm_mul_ps(_mm_cvtepi32_ps(y0), b3);
				f1 = _mm_mul_ps(_mm_cvtepi32_ps(y1), b3);
				s0 = _mm_add_ps(s0, f0);
				s1 = _mm_add_ps(s1, f1);

				x0 = _mm_cvtps_epi32(s0);
				x1 = _mm_cvtps_epi32(s1);

				x0 = _mm_packs_epi32(x0, x1);
				_mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(x0, x0));
			}
		else
			for (; x <= width - 8; x += 8)
			{
				__m128i x0, x1, y0, y1;
				__m128 s0, s1, f0, f1;
				x0 = _mm_loadu_si128((const __m128i*)(S0 + x));
				x1 = _mm_loadu_si128((const __m128i*)(S0 + x + 4));
				y0 = _mm_loadu_si128((const __m128i*)(S1 + x));
				y1 = _mm_loadu_si128((const __m128i*)(S1 + x + 4));

				s0 = _mm_mul_ps(_mm_cvtepi32_ps(x0), b0);
				s1 = _mm_mul_ps(_mm_cvtepi32_ps(x1), b0);
				f0 = _mm_mul_ps(_mm_cvtepi32_ps(y0), b1);
				f1 = _mm_mul_ps(_mm_cvtepi32_ps(y1), b1);
				s0 = _mm_add_ps(s0, f0);
				s1 = _mm_add_ps(s1, f1);

				x0 = _mm_loadu_si128((const __m128i*)(S2 + x));
				x1 = _mm_loadu_si128((const __m128i*)(S2 + x + 4));
				y0 = _mm_loadu_si128((const __m128i*)(S3 + x));
				y1 = _mm_loadu_si128((const __m128i*)(S3 + x + 4));

				f0 = _mm_mul_ps(_mm_cvtepi32_ps(x0), b2);
				f1 = _mm_mul_ps(_mm_cvtepi32_ps(x1), b2);
				s0 = _mm_add_ps(s0, f0);
				s1 = _mm_add_ps(s1, f1);
				f0 = _mm_mul_ps(_mm_cvtepi32_ps(y0), b3);
				f1 = _mm_mul_ps(_mm_cvtepi32_ps(y1), b3);
				s0 = _mm_add_ps(s0, f0);
				s1 = _mm_add_ps(s1, f1);

				x0 = _mm_cvtps_epi32(s0);
				x1 = _mm_cvtps_epi32(s1);

				x0 = _mm_packs_epi32(x0, x1);
				_mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(x0, x0));
			}

		return x;
	}
};


template<int shiftval> struct VResizeCubicVec_32f16
{
	int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
	{
		if (!checkHardwareSupport(CV_CPU_SSE2))
			return 0;

		const float** src = (const float**)_src;
		const float* beta = (const float*)_beta;
		const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
		ushort* dst = (ushort*)_dst;
		int x = 0;
		__m128 b0 = _mm_set1_ps(beta[0]), b1 = _mm_set1_ps(beta[1]),
			b2 = _mm_set1_ps(beta[2]), b3 = _mm_set1_ps(beta[3]);
		__m128i preshift = _mm_set1_epi32(shiftval);
		__m128i postshift = _mm_set1_epi16((short)shiftval);

		for (; x <= width - 8; x += 8)
		{
			__m128 x0, x1, y0, y1, s0, s1;
			__m128i t0, t1;
			x0 = _mm_loadu_ps(S0 + x);
			x1 = _mm_loadu_ps(S0 + x + 4);
			y0 = _mm_loadu_ps(S1 + x);
			y1 = _mm_loadu_ps(S1 + x + 4);

			s0 = _mm_mul_ps(x0, b0);
			s1 = _mm_mul_ps(x1, b0);
			y0 = _mm_mul_ps(y0, b1);
			y1 = _mm_mul_ps(y1, b1);
			s0 = _mm_add_ps(s0, y0);
			s1 = _mm_add_ps(s1, y1);

			x0 = _mm_loadu_ps(S2 + x);
			x1 = _mm_loadu_ps(S2 + x + 4);
			y0 = _mm_loadu_ps(S3 + x);
			y1 = _mm_loadu_ps(S3 + x + 4);

			x0 = _mm_mul_ps(x0, b2);
			x1 = _mm_mul_ps(x1, b2);
			y0 = _mm_mul_ps(y0, b3);
			y1 = _mm_mul_ps(y1, b3);
			s0 = _mm_add_ps(s0, x0);
			s1 = _mm_add_ps(s1, x1);
			s0 = _mm_add_ps(s0, y0);
			s1 = _mm_add_ps(s1, y1);

			t0 = _mm_add_epi32(_mm_cvtps_epi32(s0), preshift);
			t1 = _mm_add_epi32(_mm_cvtps_epi32(s1), preshift);

			t0 = _mm_add_epi16(_mm_packs_epi32(t0, t1), postshift);
			_mm_storeu_si128((__m128i*)(dst + x), t0);
		}

		return x;
	}
};

typedef VResizeCubicVec_32f16<SHRT_MIN> VResizeCubicVec_32f16u;
typedef VResizeCubicVec_32f16<0> VResizeCubicVec_32f16s;

struct VResizeCubicVec_32f
{
	int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
	{
		if (!checkHardwareSupport(CV_CPU_SSE))
			return 0;

		const float** src = (const float**)_src;
		const float* beta = (const float*)_beta;
		const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
		float* dst = (float*)_dst;
		int x = 0;
		__m128 b0 = _mm_set1_ps(beta[0]), b1 = _mm_set1_ps(beta[1]),
			b2 = _mm_set1_ps(beta[2]), b3 = _mm_set1_ps(beta[3]);

		for (; x <= width - 8; x += 8)
		{
			__m128 x0, x1, y0, y1, s0, s1;
			x0 = _mm_loadu_ps(S0 + x);
			x1 = _mm_loadu_ps(S0 + x + 4);
			y0 = _mm_loadu_ps(S1 + x);
			y1 = _mm_loadu_ps(S1 + x + 4);

			s0 = _mm_mul_ps(x0, b0);
			s1 = _mm_mul_ps(x1, b0);
			y0 = _mm_mul_ps(y0, b1);
			y1 = _mm_mul_ps(y1, b1);
			s0 = _mm_add_ps(s0, y0);
			s1 = _mm_add_ps(s1, y1);

			x0 = _mm_loadu_ps(S2 + x);
			x1 = _mm_loadu_ps(S2 + x + 4);
			y0 = _mm_loadu_ps(S3 + x);
			y1 = _mm_loadu_ps(S3 + x + 4);

			x0 = _mm_mul_ps(x0, b2);
			x1 = _mm_mul_ps(x1, b2);
			y0 = _mm_mul_ps(y0, b3);
			y1 = _mm_mul_ps(y1, b3);
			s0 = _mm_add_ps(s0, x0);
			s1 = _mm_add_ps(s1, x1);
			s0 = _mm_add_ps(s0, y0);
			s1 = _mm_add_ps(s1, y1);

			_mm_storeu_ps(dst + x, s0);
			_mm_storeu_ps(dst + x + 4, s1);
		}

		return x;
	}
};

#if CV_TRY_SSE4_1

struct VResizeLanczos4Vec_32f16u
{
	int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
	{
		if (CV_CPU_HAS_SUPPORT_SSE4_1) return opt_SSE4_1::VResizeLanczos4Vec_32f16u_SSE41(_src, _dst, _beta, width);
		else return 0;
	}
};

#else

typedef VResizeNoVec VResizeLanczos4Vec_32f16u;

#endif

struct VResizeLanczos4Vec_32f16s
{
	int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
	{
		const float** src = (const float**)_src;
		const float* beta = (const float*)_beta;
		const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3],
			*S4 = src[4], *S5 = src[5], *S6 = src[6], *S7 = src[7];
		short * dst = (short*)_dst;
		int x = 0;
		__m128 v_b0 = _mm_set1_ps(beta[0]), v_b1 = _mm_set1_ps(beta[1]),
			v_b2 = _mm_set1_ps(beta[2]), v_b3 = _mm_set1_ps(beta[3]),
			v_b4 = _mm_set1_ps(beta[4]), v_b5 = _mm_set1_ps(beta[5]),
			v_b6 = _mm_set1_ps(beta[6]), v_b7 = _mm_set1_ps(beta[7]);

		for (; x <= width - 8; x += 8)
		{
			__m128 v_dst0 = _mm_mul_ps(v_b0, _mm_loadu_ps(S0 + x));
			v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b1, _mm_loadu_ps(S1 + x)));
			v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b2, _mm_loadu_ps(S2 + x)));
			v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b3, _mm_loadu_ps(S3 + x)));
			v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b4, _mm_loadu_ps(S4 + x)));
			v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b5, _mm_loadu_ps(S5 + x)));
			v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b6, _mm_loadu_ps(S6 + x)));
			v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b7, _mm_loadu_ps(S7 + x)));

			__m128 v_dst1 = _mm_mul_ps(v_b0, _mm_loadu_ps(S0 + x + 4));
			v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b1, _mm_loadu_ps(S1 + x + 4)));
			v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b2, _mm_loadu_ps(S2 + x + 4)));
			v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b3, _mm_loadu_ps(S3 + x + 4)));
			v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b4, _mm_loadu_ps(S4 + x + 4)));
			v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b5, _mm_loadu_ps(S5 + x + 4)));
			v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b6, _mm_loadu_ps(S6 + x + 4)));
			v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b7, _mm_loadu_ps(S7 + x + 4)));

			__m128i v_dsti0 = _mm_cvtps_epi32(v_dst0);
			__m128i v_dsti1 = _mm_cvtps_epi32(v_dst1);

			_mm_storeu_si128((__m128i *)(dst + x), _mm_packs_epi32(v_dsti0, v_dsti1));
		}

		return x;
	}
};


struct VResizeLanczos4Vec_32f
{
	int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
	{
		const float** src = (const float**)_src;
		const float* beta = (const float*)_beta;
		const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3],
			*S4 = src[4], *S5 = src[5], *S6 = src[6], *S7 = src[7];
		float* dst = (float*)_dst;
		int x = 0;

		__m128 v_b0 = _mm_set1_ps(beta[0]), v_b1 = _mm_set1_ps(beta[1]),
			v_b2 = _mm_set1_ps(beta[2]), v_b3 = _mm_set1_ps(beta[3]),
			v_b4 = _mm_set1_ps(beta[4]), v_b5 = _mm_set1_ps(beta[5]),
			v_b6 = _mm_set1_ps(beta[6]), v_b7 = _mm_set1_ps(beta[7]);

		for (; x <= width - 4; x += 4)
		{
			__m128 v_dst = _mm_mul_ps(v_b0, _mm_loadu_ps(S0 + x));
			v_dst = _mm_add_ps(v_dst, _mm_mul_ps(v_b1, _mm_loadu_ps(S1 + x)));
			v_dst = _mm_add_ps(v_dst, _mm_mul_ps(v_b2, _mm_loadu_ps(S2 + x)));
			v_dst = _mm_add_ps(v_dst, _mm_mul_ps(v_b3, _mm_loadu_ps(S3 + x)));
			v_dst = _mm_add_ps(v_dst, _mm_mul_ps(v_b4, _mm_loadu_ps(S4 + x)));
			v_dst = _mm_add_ps(v_dst, _mm_mul_ps(v_b5, _mm_loadu_ps(S5 + x)));
			v_dst = _mm_add_ps(v_dst, _mm_mul_ps(v_b6, _mm_loadu_ps(S6 + x)));
			v_dst = _mm_add_ps(v_dst, _mm_mul_ps(v_b7, _mm_loadu_ps(S7 + x)));

			_mm_storeu_ps(dst + x, v_dst);
		}

		return x;
	}
};


#elif CV_NEON

struct VResizeLinearVec_32s8u
{
	int operator()(const uchar** _src, uchar* dst, const uchar* _beta, int width) const
	{
		const int** src = (const int**)_src, *S0 = src[0], *S1 = src[1];
		const short* beta = (const short*)_beta;
		int x = 0;
		int16x8_t v_b0 = vdupq_n_s16(beta[0]), v_b1 = vdupq_n_s16(beta[1]), v_delta = vdupq_n_s16(2);

		for (; x <= width - 16; x += 16)
		{
			int32x4_t v_src00 = vshrq_n_s32(vld1q_s32(S0 + x), 4), v_src10 = vshrq_n_s32(vld1q_s32(S1 + x), 4);
			int32x4_t v_src01 = vshrq_n_s32(vld1q_s32(S0 + x + 4), 4), v_src11 = vshrq_n_s32(vld1q_s32(S1 + x + 4), 4);

			int16x8_t v_src0 = vcombine_s16(vmovn_s32(v_src00), vmovn_s32(v_src01));
			int16x8_t v_src1 = vcombine_s16(vmovn_s32(v_src10), vmovn_s32(v_src11));

			int16x8_t v_dst0 = vaddq_s16(vshrq_n_s16(vqdmulhq_s16(v_src0, v_b0), 1),
				vshrq_n_s16(vqdmulhq_s16(v_src1, v_b1), 1));
			v_dst0 = vshrq_n_s16(vaddq_s16(v_dst0, v_delta), 2);

			v_src00 = vshrq_n_s32(vld1q_s32(S0 + x + 8), 4);
			v_src10 = vshrq_n_s32(vld1q_s32(S1 + x + 8), 4);
			v_src01 = vshrq_n_s32(vld1q_s32(S0 + x + 12), 4);
			v_src11 = vshrq_n_s32(vld1q_s32(S1 + x + 12), 4);

			v_src0 = vcombine_s16(vmovn_s32(v_src00), vmovn_s32(v_src01));
			v_src1 = vcombine_s16(vmovn_s32(v_src10), vmovn_s32(v_src11));

			int16x8_t v_dst1 = vaddq_s16(vshrq_n_s16(vqdmulhq_s16(v_src0, v_b0), 1),
				vshrq_n_s16(vqdmulhq_s16(v_src1, v_b1), 1));
			v_dst1 = vshrq_n_s16(vaddq_s16(v_dst1, v_delta), 2);

			vst1q_u8(dst + x, vcombine_u8(vqmovun_s16(v_dst0), vqmovun_s16(v_dst1)));
		}

		return x;
	}
};

struct VResizeLinearVec_32f16u
{
	int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
	{
		const float** src = (const float**)_src;
		const float* beta = (const float*)_beta;
		const float *S0 = src[0], *S1 = src[1];
		ushort* dst = (ushort*)_dst;
		int x = 0;

		float32x4_t v_b0 = vdupq_n_f32(beta[0]), v_b1 = vdupq_n_f32(beta[1]);

		for (; x <= width - 8; x += 8)
		{
			float32x4_t v_src00 = vld1q_f32(S0 + x), v_src01 = vld1q_f32(S0 + x + 4);
			float32x4_t v_src10 = vld1q_f32(S1 + x), v_src11 = vld1q_f32(S1 + x + 4);

			float32x4_t v_dst0 = vmlaq_f32(vmulq_f32(v_src00, v_b0), v_src10, v_b1);
			float32x4_t v_dst1 = vmlaq_f32(vmulq_f32(v_src01, v_b0), v_src11, v_b1);

			vst1q_u16(dst + x, vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst0)),
				vqmovn_u32(cv_vrndq_u32_f32(v_dst1))));
		}

		return x;
	}
};

struct VResizeLinearVec_32f16s
{
	int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
	{
		const float** src = (const float**)_src;
		const float* beta = (const float*)_beta;
		const float *S0 = src[0], *S1 = src[1];
		short* dst = (short*)_dst;
		int x = 0;

		float32x4_t v_b0 = vdupq_n_f32(beta[0]), v_b1 = vdupq_n_f32(beta[1]);

		for (; x <= width - 8; x += 8)
		{
			float32x4_t v_src00 = vld1q_f32(S0 + x), v_src01 = vld1q_f32(S0 + x + 4);
			float32x4_t v_src10 = vld1q_f32(S1 + x), v_src11 = vld1q_f32(S1 + x + 4);

			float32x4_t v_dst0 = vmlaq_f32(vmulq_f32(v_src00, v_b0), v_src10, v_b1);
			float32x4_t v_dst1 = vmlaq_f32(vmulq_f32(v_src01, v_b0), v_src11, v_b1);

			vst1q_s16(dst + x, vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst0)),
				vqmovn_s32(cv_vrndq_s32_f32(v_dst1))));
		}

		return x;
	}
};

struct VResizeLinearVec_32f
{
	int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
	{
		const float** src = (const float**)_src;
		const float* beta = (const float*)_beta;
		const float *S0 = src[0], *S1 = src[1];
		float* dst = (float*)_dst;
		int x = 0;

		float32x4_t v_b0 = vdupq_n_f32(beta[0]), v_b1 = vdupq_n_f32(beta[1]);

		for (; x <= width - 8; x += 8)
		{
			float32x4_t v_src00 = vld1q_f32(S0 + x), v_src01 = vld1q_f32(S0 + x + 4);
			float32x4_t v_src10 = vld1q_f32(S1 + x), v_src11 = vld1q_f32(S1 + x + 4);

			vst1q_f32(dst + x, vmlaq_f32(vmulq_f32(v_src00, v_b0), v_src10, v_b1));
			vst1q_f32(dst + x + 4, vmlaq_f32(vmulq_f32(v_src01, v_b0), v_src11, v_b1));
		}

		return x;
	}
};

typedef VResizeNoVec VResizeCubicVec_32s8u;

struct VResizeCubicVec_32f16u
{
	int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
	{
		const float** src = (const float**)_src;
		const float* beta = (const float*)_beta;
		const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
		ushort* dst = (ushort*)_dst;
		int x = 0;
		float32x4_t v_b0 = vdupq_n_f32(beta[0]), v_b1 = vdupq_n_f32(beta[1]),
			v_b2 = vdupq_n_f32(beta[2]), v_b3 = vdupq_n_f32(beta[3]);

		for (; x <= width - 8; x += 8)
		{
			float32x4_t v_dst0 = vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b0, vld1q_f32(S0 + x)),
				v_b1, vld1q_f32(S1 + x)),
				v_b2, vld1q_f32(S2 + x)),
				v_b3, vld1q_f32(S3 + x));
			float32x4_t v_dst1 = vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b0, vld1q_f32(S0 + x + 4)),
				v_b1, vld1q_f32(S1 + x + 4)),
				v_b2, vld1q_f32(S2 + x + 4)),
				v_b3, vld1q_f32(S3 + x + 4));

			vst1q_u16(dst + x, vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst0)),
				vqmovn_u32(cv_vrndq_u32_f32(v_dst1))));
		}

		return x;
	}
};

struct VResizeCubicVec_32f16s
{
	int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
	{
		const float** src = (const float**)_src;
		const float* beta = (const float*)_beta;
		const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
		short* dst = (short*)_dst;
		int x = 0;
		float32x4_t v_b0 = vdupq_n_f32(beta[0]), v_b1 = vdupq_n_f32(beta[1]),
			v_b2 = vdupq_n_f32(beta[2]), v_b3 = vdupq_n_f32(beta[3]);

		for (; x <= width - 8; x += 8)
		{
			float32x4_t v_dst0 = vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b0, vld1q_f32(S0 + x)),
				v_b1, vld1q_f32(S1 + x)),
				v_b2, vld1q_f32(S2 + x)),
				v_b3, vld1q_f32(S3 + x));
			float32x4_t v_dst1 = vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b0, vld1q_f32(S0 + x + 4)),
				v_b1, vld1q_f32(S1 + x + 4)),
				v_b2, vld1q_f32(S2 + x + 4)),
				v_b3, vld1q_f32(S3 + x + 4));

			vst1q_s16(dst + x, vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst0)),
				vqmovn_s32(cv_vrndq_s32_f32(v_dst1))));
		}

		return x;
	}
};

struct VResizeCubicVec_32f
{
	int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
	{
		const float** src = (const float**)_src;
		const float* beta = (const float*)_beta;
		const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
		float* dst = (float*)_dst;
		int x = 0;
		float32x4_t v_b0 = vdupq_n_f32(beta[0]), v_b1 = vdupq_n_f32(beta[1]),
			v_b2 = vdupq_n_f32(beta[2]), v_b3 = vdupq_n_f32(beta[3]);

		for (; x <= width - 8; x += 8)
		{
			vst1q_f32(dst + x, vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b0, vld1q_f32(S0 + x)),
				v_b1, vld1q_f32(S1 + x)),
				v_b2, vld1q_f32(S2 + x)),
				v_b3, vld1q_f32(S3 + x)));
			vst1q_f32(dst + x + 4, vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b0, vld1q_f32(S0 + x + 4)),
				v_b1, vld1q_f32(S1 + x + 4)),
				v_b2, vld1q_f32(S2 + x + 4)),
				v_b3, vld1q_f32(S3 + x + 4)));
		}

		return x;
	}
};

struct VResizeLanczos4Vec_32f16u
{
	int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
	{
		const float** src = (const float**)_src;
		const float* beta = (const float*)_beta;
		const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3],
			*S4 = src[4], *S5 = src[5], *S6 = src[6], *S7 = src[7];
		ushort * dst = (ushort*)_dst;
		int x = 0;
		float32x4_t v_b0 = vdupq_n_f32(beta[0]), v_b1 = vdupq_n_f32(beta[1]),
			v_b2 = vdupq_n_f32(beta[2]), v_b3 = vdupq_n_f32(beta[3]),
			v_b4 = vdupq_n_f32(beta[4]), v_b5 = vdupq_n_f32(beta[5]),
			v_b6 = vdupq_n_f32(beta[6]), v_b7 = vdupq_n_f32(beta[7]);

		for (; x <= width - 8; x += 8)
		{
			float32x4_t v_dst0 = vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b0, vld1q_f32(S0 + x)),
				v_b1, vld1q_f32(S1 + x)),
				v_b2, vld1q_f32(S2 + x)),
				v_b3, vld1q_f32(S3 + x));
			float32x4_t v_dst1 = vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b4, vld1q_f32(S4 + x)),
				v_b5, vld1q_f32(S5 + x)),
				v_b6, vld1q_f32(S6 + x)),
				v_b7, vld1q_f32(S7 + x));
			float32x4_t v_dst = vaddq_f32(v_dst0, v_dst1);

			v_dst0 = vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b0, vld1q_f32(S0 + x + 4)),
				v_b1, vld1q_f32(S1 + x + 4)),
				v_b2, vld1q_f32(S2 + x + 4)),
				v_b3, vld1q_f32(S3 + x + 4));
			v_dst1 = vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b4, vld1q_f32(S4 + x + 4)),
				v_b5, vld1q_f32(S5 + x + 4)),
				v_b6, vld1q_f32(S6 + x + 4)),
				v_b7, vld1q_f32(S7 + x + 4));
			v_dst1 = vaddq_f32(v_dst0, v_dst1);

			vst1q_u16(dst + x, vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst)),
				vqmovn_u32(cv_vrndq_u32_f32(v_dst1))));
		}

		return x;
	}
};

struct VResizeLanczos4Vec_32f16s
{
	int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
	{
		const float** src = (const float**)_src;
		const float* beta = (const float*)_beta;
		const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3],
			*S4 = src[4], *S5 = src[5], *S6 = src[6], *S7 = src[7];
		short * dst = (short*)_dst;
		int x = 0;
		float32x4_t v_b0 = vdupq_n_f32(beta[0]), v_b1 = vdupq_n_f32(beta[1]),
			v_b2 = vdupq_n_f32(beta[2]), v_b3 = vdupq_n_f32(beta[3]),
			v_b4 = vdupq_n_f32(beta[4]), v_b5 = vdupq_n_f32(beta[5]),
			v_b6 = vdupq_n_f32(beta[6]), v_b7 = vdupq_n_f32(beta[7]);

		for (; x <= width - 8; x += 8)
		{
			float32x4_t v_dst0 = vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b0, vld1q_f32(S0 + x)),
				v_b1, vld1q_f32(S1 + x)),
				v_b2, vld1q_f32(S2 + x)),
				v_b3, vld1q_f32(S3 + x));
			float32x4_t v_dst1 = vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b4, vld1q_f32(S4 + x)),
				v_b5, vld1q_f32(S5 + x)),
				v_b6, vld1q_f32(S6 + x)),
				v_b7, vld1q_f32(S7 + x));
			float32x4_t v_dst = vaddq_f32(v_dst0, v_dst1);

			v_dst0 = vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b0, vld1q_f32(S0 + x + 4)),
				v_b1, vld1q_f32(S1 + x + 4)),
				v_b2, vld1q_f32(S2 + x + 4)),
				v_b3, vld1q_f32(S3 + x + 4));
			v_dst1 = vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b4, vld1q_f32(S4 + x + 4)),
				v_b5, vld1q_f32(S5 + x + 4)),
				v_b6, vld1q_f32(S6 + x + 4)),
				v_b7, vld1q_f32(S7 + x + 4));
			v_dst1 = vaddq_f32(v_dst0, v_dst1);

			vst1q_s16(dst + x, vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst)),
				vqmovn_s32(cv_vrndq_s32_f32(v_dst1))));
		}

		return x;
	}
};

struct VResizeLanczos4Vec_32f
{
	int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
	{
		const float** src = (const float**)_src;
		const float* beta = (const float*)_beta;
		const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3],
			*S4 = src[4], *S5 = src[5], *S6 = src[6], *S7 = src[7];
		float* dst = (float*)_dst;
		int x = 0;
		float32x4_t v_b0 = vdupq_n_f32(beta[0]), v_b1 = vdupq_n_f32(beta[1]),
			v_b2 = vdupq_n_f32(beta[2]), v_b3 = vdupq_n_f32(beta[3]),
			v_b4 = vdupq_n_f32(beta[4]), v_b5 = vdupq_n_f32(beta[5]),
			v_b6 = vdupq_n_f32(beta[6]), v_b7 = vdupq_n_f32(beta[7]);

		for (; x <= width - 4; x += 4)
		{
			float32x4_t v_dst0 = vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b0, vld1q_f32(S0 + x)),
				v_b1, vld1q_f32(S1 + x)),
				v_b2, vld1q_f32(S2 + x)),
				v_b3, vld1q_f32(S3 + x));
			float32x4_t v_dst1 = vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b4, vld1q_f32(S4 + x)),
				v_b5, vld1q_f32(S5 + x)),
				v_b6, vld1q_f32(S6 + x)),
				v_b7, vld1q_f32(S7 + x));
			vst1q_f32(dst + x, vaddq_f32(v_dst0, v_dst1));
		}

		return x;
	}
};

#else

typedef VResizeNoVec VResizeLinearVec_32s8u;
typedef VResizeNoVec VResizeLinearVec_32f16u;
typedef VResizeNoVec VResizeLinearVec_32f16s;
typedef VResizeNoVec VResizeLinearVec_32f;

typedef VResizeNoVec VResizeCubicVec_32s8u;
typedef VResizeNoVec VResizeCubicVec_32f16u;
typedef VResizeNoVec VResizeCubicVec_32f16s;
typedef VResizeNoVec VResizeCubicVec_32f;

typedef VResizeNoVec VResizeLanczos4Vec_32f16u;
typedef VResizeNoVec VResizeLanczos4Vec_32f16s;
typedef VResizeNoVec VResizeLanczos4Vec_32f;

#endif

typedef HResizeNoVec HResizeLinearVec_8u32s;
typedef HResizeNoVec HResizeLinearVec_16u32f;
typedef HResizeNoVec HResizeLinearVec_16s32f;
typedef HResizeNoVec HResizeLinearVec_32f;
typedef HResizeNoVec HResizeLinearVec_64f;

template<typename ST, typename DT> struct Cast
{
	typedef ST type1;
	typedef DT rtype;

	DT operator()(ST val) const { return saturate_cast<DT>(val); }
};

template<>
struct VResizeLinear<uchar, int, short, FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS * 2>, VResizeLinearVec_32s8u>
{
	typedef uchar value_type;
	typedef int buf_type;
	typedef short alpha_type;

	void operator()(const buf_type** src, value_type* dst, const alpha_type* beta, int width) const
	{
		alpha_type b0 = beta[0], b1 = beta[1];
		const buf_type *S0 = src[0], *S1 = src[1];
		VResizeLinearVec_32s8u vecOp;

		int x = vecOp((const uchar**)src, (uchar*)dst, (const uchar*)beta, width);
#if CV_ENABLE_UNROLLED
		for (; x <= width - 4; x += 4)
		{
			dst[x + 0] = uchar((((b0 * (S0[x + 0] >> 4)) >> 16) + ((b1 * (S1[x + 0] >> 4)) >> 16) + 2) >> 2);
			dst[x + 1] = uchar((((b0 * (S0[x + 1] >> 4)) >> 16) + ((b1 * (S1[x + 1] >> 4)) >> 16) + 2) >> 2);
			dst[x + 2] = uchar((((b0 * (S0[x + 2] >> 4)) >> 16) + ((b1 * (S1[x + 2] >> 4)) >> 16) + 2) >> 2);
			dst[x + 3] = uchar((((b0 * (S0[x + 3] >> 4)) >> 16) + ((b1 * (S1[x + 3] >> 4)) >> 16) + 2) >> 2);
		}
#endif
		for (; x < width; x++)
			dst[x] = uchar((((b0 * (S0[x] >> 4)) >> 16) + ((b1 * (S1[x] >> 4)) >> 16) + 2) >> 2);
	}
};



//struct HResizeNoVec
//{
//	int operator()(const uchar**, uchar**, int, const int*,
//		const uchar*, int, int, int, int, int) const {
//		return 0;
//	}
//};

template <typename T, typename WT>
struct ResizeAreaFastNoVec
{
	ResizeAreaFastNoVec(int, int) { }
	ResizeAreaFastNoVec(int, int, int, int) { }
	int operator() (const T*, T*, int) const
	{
		return 0;
	}
};

#if CV_NEON

class ResizeAreaFastVec_SIMD_8u
{
public:
    ResizeAreaFastVec_SIMD_8u(int _cn, int _step) :
        cn(_cn), step(_step)
    {
    }

    int operator() (const uchar* S, uchar* D, int w) const
    {
        int dx = 0;
        const uchar* S0 = S, * S1 = S0 + step;

        uint16x8_t v_2 = vdupq_n_u16(2);

        if (cn == 1)
        {
            for ( ; dx <= w - 16; dx += 16, S0 += 32, S1 += 32, D += 16)
            {
                uint8x16x2_t v_row0 = vld2q_u8(S0), v_row1 = vld2q_u8(S1);

                uint16x8_t v_dst0 = vaddl_u8(vget_low_u8(v_row0.val[0]), vget_low_u8(v_row0.val[1]));
                v_dst0 = vaddq_u16(v_dst0, vaddl_u8(vget_low_u8(v_row1.val[0]), vget_low_u8(v_row1.val[1])));
                v_dst0 = vshrq_n_u16(vaddq_u16(v_dst0, v_2), 2);

                uint16x8_t v_dst1 = vaddl_u8(vget_high_u8(v_row0.val[0]), vget_high_u8(v_row0.val[1]));
                v_dst1 = vaddq_u16(v_dst1, vaddl_u8(vget_high_u8(v_row1.val[0]), vget_high_u8(v_row1.val[1])));
                v_dst1 = vshrq_n_u16(vaddq_u16(v_dst1, v_2), 2);

                vst1q_u8(D, vcombine_u8(vmovn_u16(v_dst0), vmovn_u16(v_dst1)));
            }
        }
        else if (cn == 4)
        {
            for ( ; dx <= w - 8; dx += 8, S0 += 16, S1 += 16, D += 8)
            {
                uint8x16_t v_row0 = vld1q_u8(S0), v_row1 = vld1q_u8(S1);

                uint16x8_t v_row00 = vmovl_u8(vget_low_u8(v_row0));
                uint16x8_t v_row01 = vmovl_u8(vget_high_u8(v_row0));
                uint16x8_t v_row10 = vmovl_u8(vget_low_u8(v_row1));
                uint16x8_t v_row11 = vmovl_u8(vget_high_u8(v_row1));

                uint16x4_t v_p0 = vadd_u16(vadd_u16(vget_low_u16(v_row00), vget_high_u16(v_row00)),
                                           vadd_u16(vget_low_u16(v_row10), vget_high_u16(v_row10)));
                uint16x4_t v_p1 = vadd_u16(vadd_u16(vget_low_u16(v_row01), vget_high_u16(v_row01)),
                                           vadd_u16(vget_low_u16(v_row11), vget_high_u16(v_row11)));
                uint16x8_t v_dst = vshrq_n_u16(vaddq_u16(vcombine_u16(v_p0, v_p1), v_2), 2);

                vst1_u8(D, vmovn_u16(v_dst));
            }
        }

        return dx;
    }

private:
    int cn, step;
};

class ResizeAreaFastVec_SIMD_16u
{
public:
    ResizeAreaFastVec_SIMD_16u(int _cn, int _step) :
        cn(_cn), step(_step)
    {
    }

    int operator() (const ushort * S, ushort * D, int w) const
    {
        int dx = 0;
        const ushort * S0 = S, * S1 = (const ushort *)((const uchar *)(S0) + step);

        uint32x4_t v_2 = vdupq_n_u32(2);

        if (cn == 1)
        {
            for ( ; dx <= w - 8; dx += 8, S0 += 16, S1 += 16, D += 8)
            {
                uint16x8x2_t v_row0 = vld2q_u16(S0), v_row1 = vld2q_u16(S1);

                uint32x4_t v_dst0 = vaddl_u16(vget_low_u16(v_row0.val[0]), vget_low_u16(v_row0.val[1]));
                v_dst0 = vaddq_u32(v_dst0, vaddl_u16(vget_low_u16(v_row1.val[0]), vget_low_u16(v_row1.val[1])));
                v_dst0 = vshrq_n_u32(vaddq_u32(v_dst0, v_2), 2);

                uint32x4_t v_dst1 = vaddl_u16(vget_high_u16(v_row0.val[0]), vget_high_u16(v_row0.val[1]));
                v_dst1 = vaddq_u32(v_dst1, vaddl_u16(vget_high_u16(v_row1.val[0]), vget_high_u16(v_row1.val[1])));
                v_dst1 = vshrq_n_u32(vaddq_u32(v_dst1, v_2), 2);

                vst1q_u16(D, vcombine_u16(vmovn_u32(v_dst0), vmovn_u32(v_dst1)));
            }
        }
        else if (cn == 4)
        {
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                uint16x8_t v_row0 = vld1q_u16(S0), v_row1 = vld1q_u16(S1);
                uint32x4_t v_dst = vaddq_u32(vaddl_u16(vget_low_u16(v_row0), vget_high_u16(v_row0)),
                                             vaddl_u16(vget_low_u16(v_row1), vget_high_u16(v_row1)));
                vst1_u16(D, vmovn_u32(vshrq_n_u32(vaddq_u32(v_dst, v_2), 2)));
            }
        }

        return dx;
    }

private:
    int cn, step;
};

class ResizeAreaFastVec_SIMD_16s
{
public:
    ResizeAreaFastVec_SIMD_16s(int _cn, int _step) :
        cn(_cn), step(_step)
    {
    }

    int operator() (const short * S, short * D, int w) const
    {
        int dx = 0;
        const short * S0 = S, * S1 = (const short *)((const uchar *)(S0) + step);

        int32x4_t v_2 = vdupq_n_s32(2);

        if (cn == 1)
        {
            for ( ; dx <= w - 8; dx += 8, S0 += 16, S1 += 16, D += 8)
            {
                int16x8x2_t v_row0 = vld2q_s16(S0), v_row1 = vld2q_s16(S1);

                int32x4_t v_dst0 = vaddl_s16(vget_low_s16(v_row0.val[0]), vget_low_s16(v_row0.val[1]));
                v_dst0 = vaddq_s32(v_dst0, vaddl_s16(vget_low_s16(v_row1.val[0]), vget_low_s16(v_row1.val[1])));
                v_dst0 = vshrq_n_s32(vaddq_s32(v_dst0, v_2), 2);

                int32x4_t v_dst1 = vaddl_s16(vget_high_s16(v_row0.val[0]), vget_high_s16(v_row0.val[1]));
                v_dst1 = vaddq_s32(v_dst1, vaddl_s16(vget_high_s16(v_row1.val[0]), vget_high_s16(v_row1.val[1])));
                v_dst1 = vshrq_n_s32(vaddq_s32(v_dst1, v_2), 2);

                vst1q_s16(D, vcombine_s16(vmovn_s32(v_dst0), vmovn_s32(v_dst1)));
            }
        }
        else if (cn == 4)
        {
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                int16x8_t v_row0 = vld1q_s16(S0), v_row1 = vld1q_s16(S1);
                int32x4_t v_dst = vaddq_s32(vaddl_s16(vget_low_s16(v_row0), vget_high_s16(v_row0)),
                                            vaddl_s16(vget_low_s16(v_row1), vget_high_s16(v_row1)));
                vst1_s16(D, vmovn_s32(vshrq_n_s32(vaddq_s32(v_dst, v_2), 2)));
            }
        }

        return dx;
    }

private:
    int cn, step;
};

struct ResizeAreaFastVec_SIMD_32f
{
    ResizeAreaFastVec_SIMD_32f(int _scale_x, int _scale_y, int _cn, int _step) :
        cn(_cn), step(_step)
    {
        fast_mode = _scale_x == 2 && _scale_y == 2 && (cn == 1 || cn == 4);
    }

    int operator() (const float * S, float * D, int w) const
    {
        if (!fast_mode)
            return 0;

        const float * S0 = S, * S1 = (const float *)((const uchar *)(S0) + step);
        int dx = 0;

        float32x4_t v_025 = vdupq_n_f32(0.25f);

        if (cn == 1)
        {
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                float32x4x2_t v_row0 = vld2q_f32(S0), v_row1 = vld2q_f32(S1);

                float32x4_t v_dst0 = vaddq_f32(v_row0.val[0], v_row0.val[1]);
                float32x4_t v_dst1 = vaddq_f32(v_row1.val[0], v_row1.val[1]);

                vst1q_f32(D, vmulq_f32(vaddq_f32(v_dst0, v_dst1), v_025));
            }
        }
        else if (cn == 4)
        {
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                float32x4_t v_dst0 = vaddq_f32(vld1q_f32(S0), vld1q_f32(S0 + 4));
                float32x4_t v_dst1 = vaddq_f32(vld1q_f32(S1), vld1q_f32(S1 + 4));

                vst1q_f32(D, vmulq_f32(vaddq_f32(v_dst0, v_dst1), v_025));
            }
        }

        return dx;
    }

private:
    int cn;
    bool fast_mode;
    int step;
};

#elif CV_SSE2_

class ResizeAreaFastVec_SIMD_8u
{
public:
    ResizeAreaFastVec_SIMD_8u(int _cn, int _step) :
        cn(_cn), step(_step)
    {
        use_simd = checkHardwareSupport(CV_CPU_SSE2);
    }

    int operator() (const uchar* S, uchar* D, int w) const
    {
        if (!use_simd)
            return 0;

        int dx = 0;
        const uchar* S0 = S;
        const uchar* S1 = S0 + step;
        __m128i zero = _mm_setzero_si128();
        __m128i delta2 = _mm_set1_epi16(2);

        if (cn == 1)
        {
            __m128i masklow = _mm_set1_epi16(0x00ff);
            for ( ; dx <= w - 8; dx += 8, S0 += 16, S1 += 16, D += 8)
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)S0);
                __m128i r1 = _mm_loadu_si128((const __m128i*)S1);

                __m128i s0 = _mm_add_epi16(_mm_srli_epi16(r0, 8), _mm_and_si128(r0, masklow));
                __m128i s1 = _mm_add_epi16(_mm_srli_epi16(r1, 8), _mm_and_si128(r1, masklow));
                s0 = _mm_add_epi16(_mm_add_epi16(s0, s1), delta2);
                s0 = _mm_packus_epi16(_mm_srli_epi16(s0, 2), zero);

                _mm_storel_epi64((__m128i*)D, s0);
            }
        }
        else if (cn == 3)
            for ( ; dx <= w - 11; dx += 6, S0 += 12, S1 += 12, D += 6)
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)S0);
                __m128i r1 = _mm_loadu_si128((const __m128i*)S1);

                __m128i r0_16l = _mm_unpacklo_epi8(r0, zero);
                __m128i r0_16h = _mm_unpacklo_epi8(_mm_srli_si128(r0, 6), zero);
                __m128i r1_16l = _mm_unpacklo_epi8(r1, zero);
                __m128i r1_16h = _mm_unpacklo_epi8(_mm_srli_si128(r1, 6), zero);

                __m128i s0 = _mm_add_epi16(r0_16l, _mm_srli_si128(r0_16l, 6));
                __m128i s1 = _mm_add_epi16(r1_16l, _mm_srli_si128(r1_16l, 6));
                s0 = _mm_add_epi16(s1, _mm_add_epi16(s0, delta2));
                s0 = _mm_packus_epi16(_mm_srli_epi16(s0, 2), zero);
                _mm_storel_epi64((__m128i*)D, s0);

                s0 = _mm_add_epi16(r0_16h, _mm_srli_si128(r0_16h, 6));
                s1 = _mm_add_epi16(r1_16h, _mm_srli_si128(r1_16h, 6));
                s0 = _mm_add_epi16(s1, _mm_add_epi16(s0, delta2));
                s0 = _mm_packus_epi16(_mm_srli_epi16(s0, 2), zero);
                _mm_storel_epi64((__m128i*)(D+3), s0);
            }
        else
        {
            CV_Assert(cn == 4);
            int v[] = { 0, 0, -1, -1 };
            __m128i mask = _mm_loadu_si128((const __m128i*)v);

            for ( ; dx <= w - 8; dx += 8, S0 += 16, S1 += 16, D += 8)
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)S0);
                __m128i r1 = _mm_loadu_si128((const __m128i*)S1);

                __m128i r0_16l = _mm_unpacklo_epi8(r0, zero);
                __m128i r0_16h = _mm_unpackhi_epi8(r0, zero);
                __m128i r1_16l = _mm_unpacklo_epi8(r1, zero);
                __m128i r1_16h = _mm_unpackhi_epi8(r1, zero);

                __m128i s0 = _mm_add_epi16(r0_16l, _mm_srli_si128(r0_16l, 8));
                __m128i s1 = _mm_add_epi16(r1_16l, _mm_srli_si128(r1_16l, 8));
                s0 = _mm_add_epi16(s1, _mm_add_epi16(s0, delta2));
                __m128i res0 = _mm_srli_epi16(s0, 2);

                s0 = _mm_add_epi16(r0_16h, _mm_srli_si128(r0_16h, 8));
                s1 = _mm_add_epi16(r1_16h, _mm_srli_si128(r1_16h, 8));
                s0 = _mm_add_epi16(s1, _mm_add_epi16(s0, delta2));
                __m128i res1 = _mm_srli_epi16(s0, 2);
                s0 = _mm_packus_epi16(_mm_or_si128(_mm_andnot_si128(mask, res0),
                                                   _mm_and_si128(mask, _mm_slli_si128(res1, 8))), zero);
                _mm_storel_epi64((__m128i*)(D), s0);
            }
        }

        return dx;
    }

private:
    int cn;
    bool use_simd;
    int step;
};

class ResizeAreaFastVec_SIMD_16u
{
public:
    ResizeAreaFastVec_SIMD_16u(int _cn, int _step) :
        cn(_cn), step(_step)
    {
        use_simd = checkHardwareSupport(CV_CPU_SSE2);
    }

    int operator() (const ushort* S, ushort* D, int w) const
    {
        if (!use_simd)
            return 0;

        int dx = 0;
        const ushort* S0 = (const ushort*)S;
        const ushort* S1 = (const ushort*)((const uchar*)(S) + step);
        __m128i masklow = _mm_set1_epi32(0x0000ffff);
        __m128i zero = _mm_setzero_si128();
        __m128i delta2 = _mm_set1_epi32(2);

#define _mm_packus_epi32(a, zero) _mm_packs_epi32(_mm_srai_epi32(_mm_slli_epi32(a, 16), 16), zero)

        if (cn == 1)
        {
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)S0);
                __m128i r1 = _mm_loadu_si128((const __m128i*)S1);

                __m128i s0 = _mm_add_epi32(_mm_srli_epi32(r0, 16), _mm_and_si128(r0, masklow));
                __m128i s1 = _mm_add_epi32(_mm_srli_epi32(r1, 16), _mm_and_si128(r1, masklow));
                s0 = _mm_add_epi32(_mm_add_epi32(s0, s1), delta2);
                s0 = _mm_srli_epi32(s0, 2);
                s0 = _mm_packus_epi32(s0, zero);

                _mm_storel_epi64((__m128i*)D, s0);
            }
        }
        else if (cn == 3)
            for ( ; dx <= w - 4; dx += 3, S0 += 6, S1 += 6, D += 3)
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)S0);
                __m128i r1 = _mm_loadu_si128((const __m128i*)S1);

                __m128i r0_16l = _mm_unpacklo_epi16(r0, zero);
                __m128i r0_16h = _mm_unpacklo_epi16(_mm_srli_si128(r0, 6), zero);
                __m128i r1_16l = _mm_unpacklo_epi16(r1, zero);
                __m128i r1_16h = _mm_unpacklo_epi16(_mm_srli_si128(r1, 6), zero);

                __m128i s0 = _mm_add_epi32(r0_16l, r0_16h);
                __m128i s1 = _mm_add_epi32(r1_16l, r1_16h);
                s0 = _mm_add_epi32(delta2, _mm_add_epi32(s0, s1));
                s0 = _mm_packus_epi32(_mm_srli_epi32(s0, 2), zero);
                _mm_storel_epi64((__m128i*)D, s0);
            }
        else
        {
            CV_Assert(cn == 4);
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)S0);
                __m128i r1 = _mm_loadu_si128((const __m128i*)S1);

                __m128i r0_32l = _mm_unpacklo_epi16(r0, zero);
                __m128i r0_32h = _mm_unpackhi_epi16(r0, zero);
                __m128i r1_32l = _mm_unpacklo_epi16(r1, zero);
                __m128i r1_32h = _mm_unpackhi_epi16(r1, zero);

                __m128i s0 = _mm_add_epi32(r0_32l, r0_32h);
                __m128i s1 = _mm_add_epi32(r1_32l, r1_32h);
                s0 = _mm_add_epi32(s1, _mm_add_epi32(s0, delta2));
                s0 = _mm_packus_epi32(_mm_srli_epi32(s0, 2), zero);
                _mm_storel_epi64((__m128i*)D, s0);
            }
        }

#undef _mm_packus_epi32

        return dx;
    }

private:
    int cn;
    int step;
    bool use_simd;
};

class ResizeAreaFastVec_SIMD_16s
{
public:
    ResizeAreaFastVec_SIMD_16s(int _cn, int _step) :
        cn(_cn), step(_step)
    {
        use_simd = checkHardwareSupport(CV_CPU_SSE2);
    }

    int operator() (const short* S, short* D, int w) const
    {
        if (!use_simd)
            return 0;

        int dx = 0;
        const short* S0 = (const short*)S;
        const short* S1 = (const short*)((const uchar*)(S) + step);
        __m128i masklow = _mm_set1_epi32(0x0000ffff);
        __m128i zero = _mm_setzero_si128();
        __m128i delta2 = _mm_set1_epi32(2);

        if (cn == 1)
        {
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)S0);
                __m128i r1 = _mm_loadu_si128((const __m128i*)S1);

                __m128i s0 = _mm_add_epi32(_mm_srai_epi32(r0, 16),
                    _mm_srai_epi32(_mm_slli_epi32(_mm_and_si128(r0, masklow), 16), 16));
                __m128i s1 = _mm_add_epi32(_mm_srai_epi32(r1, 16),
                    _mm_srai_epi32(_mm_slli_epi32(_mm_and_si128(r1, masklow), 16), 16));
                s0 = _mm_add_epi32(_mm_add_epi32(s0, s1), delta2);
                s0 = _mm_srai_epi32(s0, 2);
                s0 = _mm_packs_epi32(s0, zero);

                _mm_storel_epi64((__m128i*)D, s0);
            }
        }
        else if (cn == 3)
            for ( ; dx <= w - 4; dx += 3, S0 += 6, S1 += 6, D += 3)
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)S0);
                __m128i r1 = _mm_loadu_si128((const __m128i*)S1);

                __m128i r0_16l = _mm_srai_epi32(_mm_unpacklo_epi16(zero, r0), 16);
                __m128i r0_16h = _mm_srai_epi32(_mm_unpacklo_epi16(zero, _mm_srli_si128(r0, 6)), 16);
                __m128i r1_16l = _mm_srai_epi32(_mm_unpacklo_epi16(zero, r1), 16);
                __m128i r1_16h = _mm_srai_epi32(_mm_unpacklo_epi16(zero, _mm_srli_si128(r1, 6)), 16);

                __m128i s0 = _mm_add_epi32(r0_16l, r0_16h);
                __m128i s1 = _mm_add_epi32(r1_16l, r1_16h);
                s0 = _mm_add_epi32(delta2, _mm_add_epi32(s0, s1));
                s0 = _mm_packs_epi32(_mm_srai_epi32(s0, 2), zero);
                _mm_storel_epi64((__m128i*)D, s0);
            }
        else
        {
            CV_Assert(cn == 4);
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)S0);
                __m128i r1 = _mm_loadu_si128((const __m128i*)S1);

                __m128i r0_32l = _mm_srai_epi32(_mm_unpacklo_epi16(zero, r0), 16);
                __m128i r0_32h = _mm_srai_epi32(_mm_unpackhi_epi16(zero, r0), 16);
                __m128i r1_32l = _mm_srai_epi32(_mm_unpacklo_epi16(zero, r1), 16);
                __m128i r1_32h = _mm_srai_epi32(_mm_unpackhi_epi16(zero, r1), 16);

                __m128i s0 = _mm_add_epi32(r0_32l, r0_32h);
                __m128i s1 = _mm_add_epi32(r1_32l, r1_32h);
                s0 = _mm_add_epi32(s1, _mm_add_epi32(s0, delta2));
                s0 = _mm_packs_epi32(_mm_srai_epi32(s0, 2), zero);
                _mm_storel_epi64((__m128i*)D, s0);
            }
        }

        return dx;
    }

private:
    int cn;
    int step;
    bool use_simd;
};

struct ResizeAreaFastVec_SIMD_32f
{
    ResizeAreaFastVec_SIMD_32f(int _scale_x, int _scale_y, int _cn, int _step) :
        cn(_cn), step(_step)
    {
        fast_mode = _scale_x == 2 && _scale_y == 2 && (cn == 1 || cn == 4);
        fast_mode = fast_mode && checkHardwareSupport(CV_CPU_SSE2);
    }

    int operator() (const float * S, float * D, int w) const
    {
        if (!fast_mode)
            return 0;

        const float * S0 = S, * S1 = (const float *)((const uchar *)(S0) + step);
        int dx = 0;

        __m128 v_025 = _mm_set1_ps(0.25f);

        if (cn == 1)
        {
            const int shuffle_lo = _MM_SHUFFLE(2, 0, 2, 0), shuffle_hi = _MM_SHUFFLE(3, 1, 3, 1);
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                __m128 v_row00 = _mm_loadu_ps(S0), v_row01 = _mm_loadu_ps(S0 + 4),
                       v_row10 = _mm_loadu_ps(S1), v_row11 = _mm_loadu_ps(S1 + 4);

                __m128 v_dst0 = _mm_add_ps(_mm_shuffle_ps(v_row00, v_row01, shuffle_lo),
                                           _mm_shuffle_ps(v_row00, v_row01, shuffle_hi));
                __m128 v_dst1 = _mm_add_ps(_mm_shuffle_ps(v_row10, v_row11, shuffle_lo),
                                           _mm_shuffle_ps(v_row10, v_row11, shuffle_hi));

                _mm_storeu_ps(D, _mm_mul_ps(_mm_add_ps(v_dst0, v_dst1), v_025));
            }
        }
        else if (cn == 4)
        {
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                __m128 v_dst0 = _mm_add_ps(_mm_loadu_ps(S0), _mm_loadu_ps(S0 + 4));
                __m128 v_dst1 = _mm_add_ps(_mm_loadu_ps(S1), _mm_loadu_ps(S1 + 4));

                _mm_storeu_ps(D, _mm_mul_ps(_mm_add_ps(v_dst0, v_dst1), v_025));
            }
        }

        return dx;
    }

private:
    int cn;
    bool fast_mode;
    int step;
};

#else

typedef ResizeAreaFastNoVec<uchar, uchar> ResizeAreaFastVec_SIMD_8u;
typedef ResizeAreaFastNoVec<ushort, ushort> ResizeAreaFastVec_SIMD_16u;
typedef ResizeAreaFastNoVec<short, short> ResizeAreaFastVec_SIMD_16s;
typedef ResizeAreaFastNoVec<float, float> ResizeAreaFastVec_SIMD_32f;

#endif

inline void cv::halResize(int src_type,
	const uchar * src_data, size_t src_step, int src_width, int src_height,
	uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
	double inv_scale_x, double inv_scale_y, int interpolation)
{
	//CV_INSTRUMENT_REGION()

#ifdef use_vs
	CV_Assert((dst_width * dst_height > 0) || (inv_scale_x > 0 && inv_scale_y > 0));
#endif //use_vs

	if (inv_scale_x < DBL_EPSILON || inv_scale_y < DBL_EPSILON)
	{
		inv_scale_x = static_cast<double>(dst_width) / src_width;
		inv_scale_y = static_cast<double>(dst_height) / src_height;
	}

	//CALL_HAL(resize, cv_hal_resize, src_type, src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, inv_scale_x, inv_scale_y, interpolation);

	int  depth = CV_MAT_DEPTH(src_type), cn = CV_MAT_CN(src_type);
	Size dsize = Size(saturate_cast<int>(src_width*inv_scale_x),
		saturate_cast<int>(src_height*inv_scale_y));

#ifdef use_vs
	CV_Assert(dsize.area() > 0);
#endif //use_vs

	//CV_IPP_RUN_FAST(ipp_resize(src_data, src_step, src_width, src_height, dst_data, dst_step, dsize.width, dsize.height, inv_scale_x, inv_scale_y, depth, cn, interpolation))

	static ResizeFunc linear_tab[] =
	{
		resizeGeneric_<
		HResizeLinear<uchar, int, short,
		INTER_RESIZE_COEF_SCALE,
		HResizeLinearVec_8u32s>,
		VResizeLinear<uchar, int, short,
		FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS * 2>,
		VResizeLinearVec_32s8u> >,
		0,
		resizeGeneric_<
		HResizeLinear<ushort, float, float, 1,
		HResizeLinearVec_16u32f>,
		VResizeLinear<ushort, float, float, Cast<float, ushort>,
		VResizeLinearVec_32f16u> >,
		resizeGeneric_<
		HResizeLinear<short, float, float, 1,
		HResizeLinearVec_16s32f>,
		VResizeLinear<short, float, float, Cast<float, short>,
		VResizeLinearVec_32f16s> >,
		0,
		resizeGeneric_<
		HResizeLinear<float, float, float, 1,
		HResizeLinearVec_32f>,
		VResizeLinear<float, float, float, Cast<float, float>,
		VResizeLinearVec_32f> >,
		resizeGeneric_<
		HResizeLinear<double, double, float, 1,
		HResizeNoVec>,
		VResizeLinear<double, double, float, Cast<double, double>,
		VResizeNoVec> >,
		0
	};

	//static ResizeFunc cubic_tab[] =
	//{
	//	resizeGeneric_<
	//	HResizeCubic<uchar, int, short>,
	//	VResizeCubic<uchar, int, short,
	//	FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS * 2>,
	//	VResizeCubicVec_32s8u> >,
	//	0,
	//	resizeGeneric_<
	//	HResizeCubic<ushort, float, float>,
	//	VResizeCubic<ushort, float, float, Cast<float, ushort>,
	//	VResizeCubicVec_32f16u> >,
	//	resizeGeneric_<
	//	HResizeCubic<short, float, float>,
	//	VResizeCubic<short, float, float, Cast<float, short>,
	//	VResizeCubicVec_32f16s> >,
	//	0,
	//	resizeGeneric_<
	//	HResizeCubic<float, float, float>,
	//	VResizeCubic<float, float, float, Cast<float, float>,
	//	VResizeCubicVec_32f> >,
	//	resizeGeneric_<
	//	HResizeCubic<double, double, float>,
	//	VResizeCubic<double, double, float, Cast<double, double>,
	//	VResizeNoVec> >,
	//	0
	//};
	//
	//static ResizeFunc lanczos4_tab[] =
	//{
	//	resizeGeneric_<HResizeLanczos4<uchar, int, short>,
	//	VResizeLanczos4<uchar, int, short,
	//	FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS * 2>,
	//	VResizeNoVec> >,
	//	0,
	//	resizeGeneric_<HResizeLanczos4<ushort, float, float>,
	//	VResizeLanczos4<ushort, float, float, Cast<float, ushort>,
	//	VResizeLanczos4Vec_32f16u> >,
	//	resizeGeneric_<HResizeLanczos4<short, float, float>,
	//	VResizeLanczos4<short, float, float, Cast<float, short>,
	//	VResizeLanczos4Vec_32f16s> >,
	//	0,
	//	resizeGeneric_<HResizeLanczos4<float, float, float>,
	//	VResizeLanczos4<float, float, float, Cast<float, float>,
	//	VResizeLanczos4Vec_32f> >,
	//	resizeGeneric_<HResizeLanczos4<double, double, float>,
	//	VResizeLanczos4<double, double, float, Cast<double, double>,
	//	VResizeNoVec> >,
	//	0
	//};
	
	static ResizeAreaFastFunc areafast_tab[] =
	{
		resizeAreaFast_<uchar, int, ResizeAreaFastVec<uchar, ResizeAreaFastVec_SIMD_8u> >,
		0,
		resizeAreaFast_<ushort, float, ResizeAreaFastVec<ushort, ResizeAreaFastVec_SIMD_16u> >,
		resizeAreaFast_<short, float, ResizeAreaFastVec<short, ResizeAreaFastVec_SIMD_16s> >,
		0,
		resizeAreaFast_<float, float, ResizeAreaFastVec_SIMD_32f>,
		resizeAreaFast_<double, double, ResizeAreaFastNoVec<double, double> >,
		0
	};
	
	//static ResizeAreaFunc area_tab[] =
	//{
	//	resizeArea_<uchar, float>, 0, resizeArea_<ushort, float>,
	//	resizeArea_<short, float>, 0, resizeArea_<float, float>,
	//	resizeArea_<double, double>, 0
	//};

	double scale_x = 1. / inv_scale_x, scale_y = 1. / inv_scale_y;

	int iscale_x = saturate_cast<int>(scale_x);
	int iscale_y = saturate_cast<int>(scale_y);

	bool is_area_fast = std::abs(scale_x - iscale_x) < DBL_EPSILON &&
		std::abs(scale_y - iscale_y) < DBL_EPSILON;

	Mat src(Size(src_width, src_height), src_type, const_cast<uchar*>(src_data), src_step);
	Mat dst(dsize, src_type, dst_data, dst_step);

	//if (interpolation == INTER_NEAREST)
	//{
	//	resizeNN(src, dst, inv_scale_x, inv_scale_y);
	//	return;
	//}

	int k, sx, sy, dx, dy;


	{
		// in case of scale_x && scale_y is equal to 2
		// INTER_AREA (fast) also is equal to INTER_LINEAR
		if (interpolation == INTER_LINEAR && is_area_fast && iscale_x == 2 && iscale_y == 2)
			interpolation = INTER_AREA;
	
		// true "area" interpolation is only implemented for the case (scale_x <= 1 && scale_y <= 1).
		// In other cases it is emulated using some variant of bilinear interpolation
		if (interpolation == INTER_AREA && scale_x >= 1 && scale_y >= 1)
		{
			if (is_area_fast)
			{
				int area = iscale_x*iscale_y;
				size_t srcstep = src_step / src.elemSize1();
				AutoBuffer<int> _ofs(area + dsize.width*cn);
				int* ofs = _ofs;
				int* xofs = ofs + area;
				ResizeAreaFastFunc func = areafast_tab[depth];
				CV_Assert(func != 0);
	
				for (sy = 0, k = 0; sy < iscale_y; sy++)
					for (sx = 0; sx < iscale_x; sx++)
						ofs[k++] = (int)(sy*srcstep + sx*cn);
	
				for (dx = 0; dx < dsize.width; dx++)
				{
					int j = dx * cn;
					sx = iscale_x * j;
					for (k = 0; k < cn; k++)
						xofs[j + k] = sx + k;
				}
	
				func(src, dst, ofs, xofs, iscale_x, iscale_y);
				return;
			}
	
			//ResizeAreaFunc func = area_tab[depth];
			//CV_Assert(func != 0 && cn <= 4);
	
			//AutoBuffer<DecimateAlpha> _xytab((src_width + src_height) * 2);
			//DecimateAlpha* xtab = _xytab, *ytab = xtab + src_width * 2;
	
			//int xtab_size = computeResizeAreaTab(src_width, dsize.width, cn, scale_x, xtab);
			//int ytab_size = computeResizeAreaTab(src_height, dsize.height, 1, scale_y, ytab);
	
			//AutoBuffer<int> _tabofs(dsize.height + 1);
			//int* tabofs = _tabofs;
			//for (k = 0, dy = 0; k < ytab_size; k++)
			//{
			//	if (k == 0 || ytab[k].di != ytab[k - 1].di)
			//	{
			//		assert(ytab[k].di == dy);
			//		tabofs[dy++] = k;
			//	}
			//}
			//tabofs[dy] = ytab_size;
	
			//func(src, dst, xtab, xtab_size, ytab, ytab_size, tabofs);
			//return;
		}
	}

	int xmin = 0, xmax = dsize.width, width = dsize.width*cn;
	//bool area_mode = interpolation == INTER_AREA;
	bool fixpt = depth == CV_8U;
	float fx, fy;
	ResizeFunc func = 0;
	int ksize = 0, ksize2;

	//if (interpolation == INTER_CUBIC)
	//	ksize = 4, func = cubic_tab[depth];
	//else if (interpolation == INTER_LANCZOS4)
	//	ksize = 8, func = lanczos4_tab[depth];
	//else if (interpolation == INTER_LINEAR || interpolation == INTER_AREA)
	if (interpolation == INTER_LINEAR )
		ksize = 2, func = linear_tab[depth];
	else
		CV_Error(CV_StsBadArg, "Unknown interpolation method");
	ksize2 = ksize / 2;

	CV_Assert(func != 0);

	AutoBuffer<uchar> _buffer((width + dsize.height)*(sizeof(int) + sizeof(float)*ksize));
	int* xofs = (int*)(uchar*)_buffer;
	int* yofs = xofs + width;
	float* alpha = (float*)(yofs + dsize.height);
	short* ialpha = (short*)alpha;
	float* beta = alpha + width*ksize;
	short* ibeta = ialpha + width*ksize;
	float cbuf[MAX_ESIZE] = { 0 };

	for (dx = 0; dx < dsize.width; dx++)
	{
		//if (!area_mode)
		{
			fx = (float)((dx + 0.5)*scale_x - 0.5);
			sx = cvFloor(fx);
			fx -= sx;
		}
		//else
		//{
		//	sx = cvFloor(dx*scale_x);
		//	fx = (float)((dx + 1) - (sx + 1)*inv_scale_x);
		//	fx = fx <= 0 ? 0.f : fx - cvFloor(fx);
		//}

		if (sx < ksize2 - 1)
		{
			xmin = dx + 1;
			//if (sx < 0 && (interpolation != INTER_CUBIC && interpolation != INTER_LANCZOS4))
			//	fx = 0, sx = 0;
		}

		if (sx + ksize2 >= src_width)
		{
			xmax = std::min(xmax, dx);
			//if (sx >= src_width - 1 && (interpolation != INTER_CUBIC && interpolation != INTER_LANCZOS4))
			//	fx = 0, sx = src_width - 1;
		}

		for (k = 0, sx *= cn; k < cn; k++)
			xofs[dx*cn + k] = sx + k;

		//if (interpolation == INTER_CUBIC)
		//	interpolateCubic(fx, cbuf);
		//else if (interpolation == INTER_LANCZOS4)
		//	interpolateLanczos4(fx, cbuf);
		//else
		{
			cbuf[0] = 1.f - fx;
			cbuf[1] = fx;
		}
		if (fixpt)
		{
			for (k = 0; k < ksize; k++)
				ialpha[dx*cn*ksize + k] = saturate_cast<short>(cbuf[k] * INTER_RESIZE_COEF_SCALE);
			for (; k < cn*ksize; k++)
				ialpha[dx*cn*ksize + k] = ialpha[dx*cn*ksize + k - ksize];
		}
		else
		{
			for (k = 0; k < ksize; k++)
				alpha[dx*cn*ksize + k] = cbuf[k];
			for (; k < cn*ksize; k++)
				alpha[dx*cn*ksize + k] = alpha[dx*cn*ksize + k - ksize];
		}
	}

	for (dy = 0; dy < dsize.height; dy++)
	{
		//if (!area_mode)
		{
			fy = (float)((dy + 0.5)*scale_y - 0.5);
			sy = cvFloor(fy);
			fy -= sy;
		}
		//else
		//{
		//	sy = cvFloor(dy*scale_y);
		//	fy = (float)((dy + 1) - (sy + 1)*inv_scale_y);
		//	fy = fy <= 0 ? 0.f : fy - cvFloor(fy);
		//}

		yofs[dy] = sy;
		//if (interpolation == INTER_CUBIC)
		//	interpolateCubic(fy, cbuf);
		//else if (interpolation == INTER_LANCZOS4)
		//	interpolateLanczos4(fy, cbuf);
		//else
		{
			cbuf[0] = 1.f - fy;
			cbuf[1] = fy;
		}

		if (fixpt)
		{
			for (k = 0; k < ksize; k++)
				ibeta[dy*ksize + k] = saturate_cast<short>(cbuf[k] * INTER_RESIZE_COEF_SCALE);
		}
		else
		{
			for (k = 0; k < ksize; k++)
				beta[dy*ksize + k] = cbuf[k];
		}
	}

	func(src, dst, xofs, fixpt ? (void*)ialpha : (void*)alpha, yofs,
		fixpt ? (void*)ibeta : (void*)beta, xmin, xmax, ksize);
}

inline void cv::integral(InputArray _src, OutputArray _sum, OutputArray _sqsum, OutputArray _tilted, int sdepth, int sqdepth)
{
	//CV_INSTRUMENT_REGION()

	int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
	if (sdepth <= 0)
		sdepth = depth == CV_8U ? CV_32S : CV_64F;
	if (sqdepth <= 0)
		sqdepth = CV_64F;
	sdepth = CV_MAT_DEPTH(sdepth), sqdepth = CV_MAT_DEPTH(sqdepth);

	Size ssize = _src.size(), isize(ssize.width + 1, ssize.height + 1);
	_sum.create(isize, CV_MAKETYPE(sdepth, cn));
	Mat src = _src.getMat(), sum = _sum.getMat(), sqsum, tilted;

	if (_sqsum.needed())
	{
		_sqsum.create(isize, CV_MAKETYPE(sqdepth, cn));
		sqsum = _sqsum.getMat();
	};

	if (_tilted.needed())
	{
		_tilted.create(isize, CV_MAKETYPE(sdepth, cn));
		tilted = _tilted.getMat();
	}

	halIntegral(depth, sdepth, sqdepth,
		src.ptr(), src.step,
		sum.ptr(), sum.step,
		sqsum.ptr(), sqsum.step,
		tilted.ptr(), tilted.step,
		src.cols, src.rows, cn);
}

template <typename T, typename ST, typename QT>
struct Integral_SIMD
{
	bool operator()(const T *, size_t,
		ST *, size_t,
		QT *, size_t,
		ST *, size_t,
		int, int, int) const
	{
		return false;
	}
};

template<typename T, typename ST, typename QT>
void integral_(const T* src, size_t _srcstep, ST* sum, size_t _sumstep,
	QT* sqsum, size_t _sqsumstep, ST* tilted, size_t _tiltedstep,
	int width, int height, int cn)
{
	int x, y, k;

	if (Integral_SIMD<T, ST, QT>()(src, _srcstep,
		sum, _sumstep,
		sqsum, _sqsumstep,
		tilted, _tiltedstep,
		width, height, cn))
		return;

	int srcstep = (int)(_srcstep / sizeof(T));
	int sumstep = (int)(_sumstep / sizeof(ST));
	int tiltedstep = (int)(_tiltedstep / sizeof(ST));
	int sqsumstep = (int)(_sqsumstep / sizeof(QT));

	width *= cn;

	memset(sum, 0, (width + cn)*sizeof(sum[0]));
	sum += sumstep + cn;

	if (sqsum)
	{
		memset(sqsum, 0, (width + cn)*sizeof(sqsum[0]));
		sqsum += sqsumstep + cn;
	}

	if (tilted)
	{
		memset(tilted, 0, (width + cn)*sizeof(tilted[0]));
		tilted += tiltedstep + cn;
	}

	if (sqsum == 0 && tilted == 0)
	{
		for (y = 0; y < height; y++, src += srcstep - cn, sum += sumstep - cn)
		{
			for (k = 0; k < cn; k++, src++, sum++)
			{
				ST s = sum[-cn] = 0;
				for (x = 0; x < width; x += cn)
				{
					s += src[x];
					sum[x] = sum[x - sumstep] + s;
				}
			}
		}
	}
	else if (tilted == 0)
	{
		for (y = 0; y < height; y++, src += srcstep - cn,
			sum += sumstep - cn, sqsum += sqsumstep - cn)
		{
			for (k = 0; k < cn; k++, src++, sum++, sqsum++)
			{
				ST s = sum[-cn] = 0;
				QT sq = sqsum[-cn] = 0;
				for (x = 0; x < width; x += cn)
				{
					T it = src[x];
					s += it;
					sq += (QT)it*it;
					ST t = sum[x - sumstep] + s;
					QT tq = sqsum[x - sqsumstep] + sq;
					sum[x] = t;
					sqsum[x] = tq;
				}
			}
		}
	}
	else
	{
		AutoBuffer<ST> _buf(width + cn);
		ST* buf = _buf;
		ST s;
		QT sq;
		for (k = 0; k < cn; k++, src++, sum++, tilted++, buf++)
		{
			sum[-cn] = tilted[-cn] = 0;

			for (x = 0, s = 0, sq = 0; x < width; x += cn)
			{
				T it = src[x];
				buf[x] = tilted[x] = it;
				s += it;
				sq += (QT)it*it;
				sum[x] = s;
				if (sqsum)
					sqsum[x] = sq;
			}

			if (width == cn)
				buf[cn] = 0;

			if (sqsum)
			{
				sqsum[-cn] = 0;
				sqsum++;
			}
		}

		for (y = 1; y < height; y++)
		{
			src += srcstep - cn;
			sum += sumstep - cn;
			tilted += tiltedstep - cn;
			buf += -cn;

			if (sqsum)
				sqsum += sqsumstep - cn;

			for (k = 0; k < cn; k++, src++, sum++, tilted++, buf++)
			{
				T it = src[0];
				ST t0 = s = it;
				QT tq0 = sq = (QT)it*it;

				sum[-cn] = 0;
				if (sqsum)
					sqsum[-cn] = 0;
				tilted[-cn] = tilted[-tiltedstep];

				sum[0] = sum[-sumstep] + t0;
				if (sqsum)
					sqsum[0] = sqsum[-sqsumstep] + tq0;
				tilted[0] = tilted[-tiltedstep] + t0 + buf[cn];

				for (x = cn; x < width - cn; x += cn)
				{
					ST t1 = buf[x];
					buf[x - cn] = t1 + t0;
					t0 = it = src[x];
					tq0 = (QT)it*it;
					s += t0;
					sq += tq0;
					sum[x] = sum[x - sumstep] + s;
					if (sqsum)
						sqsum[x] = sqsum[x - sqsumstep] + sq;
					t1 += buf[x + cn] + t0 + tilted[x - tiltedstep - cn];
					tilted[x] = t1;
				}

				if (width > cn)
				{
					ST t1 = buf[x];
					buf[x - cn] = t1 + t0;
					t0 = it = src[x];
					tq0 = (QT)it*it;
					s += t0;
					sq += tq0;
					sum[x] = sum[x - sumstep] + s;
					if (sqsum)
						sqsum[x] = sqsum[x - sqsumstep] + sq;
					tilted[x] = t0 + t1 + tilted[x - tiltedstep - cn];
					buf[x] = t0;
				}

				if (sqsum)
					sqsum++;
			}
		}
	}
}

inline void cv::halIntegral(int depth, int sdepth, int sqdepth,
	const uchar* src, size_t srcstep,
	uchar* sum, size_t sumstep,
	uchar* sqsum, size_t sqsumstep,
	uchar* tilted, size_t tstep,
	int width, int height, int cn)
{
	//CALL_HAL(integral, cv_hal_integral, depth, sdepth, sqdepth, src, srcstep, sum, sumstep, sqsum, sqsumstep, tilted, tstep, width, height, cn);
	//CV_IPP_RUN_FAST(ipp_integral(depth, sdepth, sqdepth, src, srcstep, sum, sumstep, sqsum, sqsumstep, tilted, tstep, width, height, cn));

#define ONE_CALL(A, B, C) integral_<A, B, C>((const A*)src, srcstep, (B*)sum, sumstep, (C*)sqsum, sqsumstep, (B*)tilted, tstep, width, height, cn)

	if (depth == CV_8U && sdepth == CV_32S && sqdepth == CV_64F)
		ONE_CALL(uchar, int, double);
	else if (depth == CV_8U && sdepth == CV_32S && sqdepth == CV_32F)
		ONE_CALL(uchar, int, float);
	else if (depth == CV_8U && sdepth == CV_32S && sqdepth == CV_32S)
		ONE_CALL(uchar, int, int);
	else if (depth == CV_8U && sdepth == CV_32F && sqdepth == CV_64F)
		ONE_CALL(uchar, float, double);
	else if (depth == CV_8U && sdepth == CV_32F && sqdepth == CV_32F)
		ONE_CALL(uchar, float, float);
	else if (depth == CV_8U && sdepth == CV_64F && sqdepth == CV_64F)
		ONE_CALL(uchar, double, double);
	else if (depth == CV_16U && sdepth == CV_64F && sqdepth == CV_64F)
		ONE_CALL(ushort, double, double);
	else if (depth == CV_16S && sdepth == CV_64F && sqdepth == CV_64F)
		ONE_CALL(short, double, double);
	else if (depth == CV_32F && sdepth == CV_32F && sqdepth == CV_64F)
		ONE_CALL(float, float, double);
	else if (depth == CV_32F && sdepth == CV_32F && sqdepth == CV_32F)
		ONE_CALL(float, float, float);
	else if (depth == CV_32F && sdepth == CV_64F && sqdepth == CV_64F)
		ONE_CALL(float, double, double);
	else if (depth == CV_64F && sdepth == CV_64F && sqdepth == CV_64F)
		ONE_CALL(double, double, double);
	else
		CV_Error(CV_StsUnsupportedFormat, "");

#undef ONE_CALL
}


template <typename T> static inline
void scalarToRawData_(const Scalar& s, T * const buf, const int cn, const int unroll_to)
{
	int i = 0;
	for (; i < cn; i++)
		buf[i] = saturate_cast<T>(s.val[i]);
	for (; i < unroll_to; i++)
		buf[i] = buf[i - cn];
}

//static void scalarToRawData(const Scalar& s, void* _buf, int type, int unroll_to)
inline void ImgprocScalarToRawData(const Scalar& s, void* _buf, int type, int unroll_to)
{
	//CV_INSTRUMENT_REGION()

	const int depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
	CV_Assert(cn <= 4);
	switch (depth)
	{
	case CV_8U:
		scalarToRawData_<uchar>(s, (uchar*)_buf, cn, unroll_to);
		break;
	case CV_8S:
		scalarToRawData_<schar>(s, (schar*)_buf, cn, unroll_to);
		break;
	case CV_16U:
		scalarToRawData_<ushort>(s, (ushort*)_buf, cn, unroll_to);
		break;
	case CV_16S:
		scalarToRawData_<short>(s, (short*)_buf, cn, unroll_to);
		break;
	case CV_32S:
		scalarToRawData_<int>(s, (int*)_buf, cn, unroll_to);
		break;
	case CV_32F:
		scalarToRawData_<float>(s, (float*)_buf, cn, unroll_to);
		break;
	case CV_64F:
		scalarToRawData_<double>(s, (double*)_buf, cn, unroll_to);
		break;
	default:
		CV_Error(CV_StsUnsupportedFormat, "");
	}
}

#define CV_AA 16
enum { XY_SHIFT = 16, XY_ONE = 1 << XY_SHIFT, DRAWING_STORAGE_BLOCK = (1 << 12) - 256 };

static inline void ICV_HLINE_X(uchar* ptr, int xl, int xr, const uchar* color, int pix_size)
{
	uchar* hline_min_ptr = (uchar*)(ptr)+(xl)*(pix_size);
	uchar* hline_end_ptr = (uchar*)(ptr)+(xr + 1)*(pix_size);
	uchar* hline_ptr = hline_min_ptr;
	if (pix_size == 1)
		memset(hline_min_ptr, *color, hline_end_ptr - hline_min_ptr);
	else//if (pix_size != 1)
	{
		if (hline_min_ptr < hline_end_ptr)
		{
			memcpy(hline_ptr, color, pix_size);
			hline_ptr += pix_size;
		}//end if (hline_min_ptr < hline_end_ptr)
		size_t sizeToCopy = pix_size;
		while (hline_ptr < hline_end_ptr)
		{
			memcpy(hline_ptr, hline_min_ptr, sizeToCopy);
			hline_ptr += sizeToCopy;
			sizeToCopy = std::min(2 * sizeToCopy, static_cast<size_t>(hline_end_ptr - hline_ptr));
		}//end while(hline_ptr < hline_end_ptr)
	}//end if (pix_size != 1)
}

static inline void ICV_HLINE(uchar* ptr, int xl, int xr, const void* color, int pix_size)
{
	ICV_HLINE_X(ptr, xl, xr, reinterpret_cast<const uchar*>(color), pix_size);
}

/* draws simple or filled circle */
static void Circle(Mat& img, Point center, int radius, const void* color, int fill)
{
	Size size = img.size();
	size_t step = img.step;
	int pix_size = (int)img.elemSize();
	uchar* ptr = img.ptr();
	int err = 0, dx = radius, dy = 0, plus = 1, minus = (radius << 1) - 1;
	int inside = center.x >= radius && center.x < size.width - radius &&
		center.y >= radius && center.y < size.height - radius;

#define ICV_PUT_POINT( ptr, x )     \
        memcpy( ptr + (x)*pix_size, color, pix_size );

	while (dx >= dy)
	{
		int mask;
		int y11 = center.y - dy, y12 = center.y + dy, y21 = center.y - dx, y22 = center.y + dx;
		int x11 = center.x - dx, x12 = center.x + dx, x21 = center.x - dy, x22 = center.x + dy;

		if (inside)
		{
			uchar *tptr0 = ptr + y11 * step;
			uchar *tptr1 = ptr + y12 * step;

			if (!fill)
			{
				ICV_PUT_POINT(tptr0, x11);
				ICV_PUT_POINT(tptr1, x11);
				ICV_PUT_POINT(tptr0, x12);
				ICV_PUT_POINT(tptr1, x12);
			}
			else
			{
				ICV_HLINE(tptr0, x11, x12, color, pix_size);
				ICV_HLINE(tptr1, x11, x12, color, pix_size);
			}

			tptr0 = ptr + y21 * step;
			tptr1 = ptr + y22 * step;

			if (!fill)
			{
				ICV_PUT_POINT(tptr0, x21);
				ICV_PUT_POINT(tptr1, x21);
				ICV_PUT_POINT(tptr0, x22);
				ICV_PUT_POINT(tptr1, x22);
			}
			else
			{
				ICV_HLINE(tptr0, x21, x22, color, pix_size);
				ICV_HLINE(tptr1, x21, x22, color, pix_size);
			}
		}
		else if (x11 < size.width && x12 >= 0 && y21 < size.height && y22 >= 0)
		{
			if (fill)
			{
				x11 = std::max(x11, 0);
				x12 = MIN(x12, size.width - 1);
			}

			if ((unsigned)y11 < (unsigned)size.height)
			{
				uchar *tptr = ptr + y11 * step;

				if (!fill)
				{
					if (x11 >= 0)
						ICV_PUT_POINT(tptr, x11);
					if (x12 < size.width)
						ICV_PUT_POINT(tptr, x12);
				}
				else
					ICV_HLINE(tptr, x11, x12, color, pix_size);
			}

			if ((unsigned)y12 < (unsigned)size.height)
			{
				uchar *tptr = ptr + y12 * step;

				if (!fill)
				{
					if (x11 >= 0)
						ICV_PUT_POINT(tptr, x11);
					if (x12 < size.width)
						ICV_PUT_POINT(tptr, x12);
				}
				else
					ICV_HLINE(tptr, x11, x12, color, pix_size);
			}

			if (x21 < size.width && x22 >= 0)
			{
				if (fill)
				{
					x21 = std::max(x21, 0);
					x22 = MIN(x22, size.width - 1);
				}

				if ((unsigned)y21 < (unsigned)size.height)
				{
					uchar *tptr = ptr + y21 * step;

					if (!fill)
					{
						if (x21 >= 0)
							ICV_PUT_POINT(tptr, x21);
						if (x22 < size.width)
							ICV_PUT_POINT(tptr, x22);
					}
					else
						ICV_HLINE(tptr, x21, x22, color, pix_size);
				}

				if ((unsigned)y22 < (unsigned)size.height)
				{
					uchar *tptr = ptr + y22 * step;

					if (!fill)
					{
						if (x21 >= 0)
							ICV_PUT_POINT(tptr, x21);
						if (x22 < size.width)
							ICV_PUT_POINT(tptr, x22);
					}
					else
						ICV_HLINE(tptr, x21, x22, color, pix_size);
				}
			}
		}
		dy++;
		err += plus;
		plus += 2;

		mask = (err <= 0) - 1;

		err -= minus & mask;
		dx += mask;
		minus -= mask & 2;
	}

#undef  ICV_PUT_POINT
}

inline void cv::circle(InputOutputArray _img, Point center, int radius,
	const Scalar& color, int thickness, int line_type, int shift)
{
	Mat img = _img.getMat();

	if (line_type == CV_AA && img.depth() != CV_8U)
		line_type = 8;

	double buf[4];
	ImgprocScalarToRawData(color, buf, img.type(), 0);
	Circle(img, center, radius, buf, thickness < 0);
}

class EqualizeHistCalcHist_Invoker : public cv::ParallelLoopBody
{
public:
	enum { HIST_SZ = 256 };

	EqualizeHistCalcHist_Invoker(cv::Mat& src, int* histogram, cv::Mutex* histogramLock)
		: src_(src), globalHistogram_(histogram), histogramLock_(histogramLock)
	{ }

	void operator()(const cv::Range& rowRange) const
	{
		int localHistogram[HIST_SZ] = { 0, };

		const size_t sstep = src_.step;

		int width = src_.cols;
		int height = rowRange.end - rowRange.start;

		if (src_.isContinuous())
		{
			width *= height;
			height = 1;
		}

		for (const uchar* ptr = src_.ptr<uchar>(rowRange.start); height--; ptr += sstep)
		{
			int x = 0;
			for (; x <= width - 4; x += 4)
			{
				int t0 = ptr[x], t1 = ptr[x + 1];
				localHistogram[t0]++; localHistogram[t1]++;
				t0 = ptr[x + 2]; t1 = ptr[x + 3];
				localHistogram[t0]++; localHistogram[t1]++;
			}

			for (; x < width; ++x)
				localHistogram[ptr[x]]++;
		}

		cv::AutoLock lock(*histogramLock_);

		for (int i = 0; i < HIST_SZ; i++)
			globalHistogram_[i] += localHistogram[i];
	}

	static bool isWorthParallel(const cv::Mat& src)
	{
		return (src.total() >= 640 * 480);
	}

private:
	EqualizeHistCalcHist_Invoker& operator=(const EqualizeHistCalcHist_Invoker&);

	cv::Mat& src_;
	int* globalHistogram_;
	cv::Mutex* histogramLock_;
};

class EqualizeHistLut_Invoker : public cv::ParallelLoopBody
{
public:
	EqualizeHistLut_Invoker(cv::Mat& src, cv::Mat& dst, int* lut)
		: src_(src),
		dst_(dst),
		lut_(lut)
	{ }

	void operator()(const cv::Range& rowRange) const
	{
		const size_t sstep = src_.step;
		const size_t dstep = dst_.step;

		int width = src_.cols;
		int height = rowRange.end - rowRange.start;
		int* lut = lut_;

		if (src_.isContinuous() && dst_.isContinuous())
		{
			width *= height;
			height = 1;
		}

		const uchar* sptr = src_.ptr<uchar>(rowRange.start);
		uchar* dptr = dst_.ptr<uchar>(rowRange.start);

		for (; height--; sptr += sstep, dptr += dstep)
		{
			int x = 0;
			for (; x <= width - 4; x += 4)
			{
				int v0 = sptr[x];
				int v1 = sptr[x + 1];
				int x0 = lut[v0];
				int x1 = lut[v1];
				dptr[x] = (uchar)x0;
				dptr[x + 1] = (uchar)x1;

				v0 = sptr[x + 2];
				v1 = sptr[x + 3];
				x0 = lut[v0];
				x1 = lut[v1];
				dptr[x + 2] = (uchar)x0;
				dptr[x + 3] = (uchar)x1;
			}

			for (; x < width; ++x)
				dptr[x] = (uchar)lut[sptr[x]];
		}
	}

	static bool isWorthParallel(const cv::Mat& src)
	{
		return (src.total() >= 640 * 480);
	}

private:
	EqualizeHistLut_Invoker& operator=(const EqualizeHistLut_Invoker&);

	cv::Mat& src_;
	cv::Mat& dst_;
	int* lut_;
};

inline void cv::equalizeHist(InputArray _src, OutputArray _dst)
{
	//CV_INSTRUMENT_REGION()

	//CV_Assert(_src.type() == CV_8UC1);

	if (_src.empty())
		return;

	//CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
		//ocl_equalizeHist(_src, _dst))

		Mat src = _src.getMat();
	_dst.create(src.size(), src.type());
	Mat dst = _dst.getMat();

	//CV_OVX_RUN(!ovx::skipSmallImages<VX_KERNEL_EQUALIZE_HISTOGRAM>(src.cols, src.rows),
	//	openvx_equalize_hist(src, dst))

		Mutex histogramLockInstance;

	const int hist_sz = EqualizeHistCalcHist_Invoker::HIST_SZ;
	int hist[hist_sz] = { 0, };
	int lut[hist_sz];

	EqualizeHistCalcHist_Invoker calcBody(src, hist, &histogramLockInstance);
	EqualizeHistLut_Invoker      lutBody(src, dst, lut);
	cv::Range heightRange(0, src.rows);

	if (EqualizeHistCalcHist_Invoker::isWorthParallel(src))
		parallel_for_(heightRange, calcBody);
	else
		calcBody(heightRange);

	int i = 0;
	while (!hist[i]) ++i;

	int total = (int)src.total();
	if (hist[i] == total)
	{
		dst.setTo(i);
		return;
	}

	float scale = (hist_sz - 1.f) / (total - hist[i]);
	int sum = 0;

	for (lut[i++] = 0; i < hist_sz; ++i)
	{
		sum += hist[i];
		lut[i] = saturate_cast<uchar>(sum * scale);
	}

	if (EqualizeHistLut_Invoker::isWorthParallel(src))
		parallel_for_(heightRange, lutBody);
	else
		lutBody(heightRange);
}

#endif //BNIMGPROC_HPP_
