/* Haar features calculation */

#ifndef OPENCV_OBJDETECT_HAAR_HPP
#define OPENCV_OBJDETECT_HAAR_HPP

//#include<opencv.hpp>
//#include "bnobjdetect_c.h"
//using namespace cv;

namespace faceDetect
{

#define CV_HAAR_FEATURE_MAX_LOCAL 3

	typedef int sumtype;
	typedef double sqsumtype;

	static inline void* cvAlignPtr(const void* ptr, int align = 32)
	{
		CV_DbgAssert((align & (align - 1)) == 0);
		return (void*)(((size_t)ptr + align - 1) & ~(size_t)(align - 1));
	}


	typedef struct CvHidHaarFeature
	{
		struct
		{
			sumtype *p0, *p1, *p2, *p3;
			float weight;
		}
		rect[CV_HAAR_FEATURE_MAX_LOCAL];
	} CvHidHaarFeature;


	typedef struct CvHidHaarTreeNode
	{
		CvHidHaarFeature feature;
		float threshold;
		int left;
		int right;
	} CvHidHaarTreeNode;


	typedef struct CvHidHaarClassifier
	{
		int count;
		CvHaarFeature* orig_feature;
		CvHidHaarTreeNode* node;
		float* alpha;
	} CvHidHaarClassifier;

	//typedef struct CvHidHaarStageClassifier
	//{
	//	int  count;
	//	float threshold;
	//	CvHidHaarClassifier* classifier;
	//	int two_rects;

	//	struct CvHidHaarStageClassifier* next;
	//	struct CvHidHaarStageClassifier* child;
	//	struct CvHidHaarStageClassifier* parent;
	//} CvHidHaarStageClassifier;


	//typedef struct CvHidHaarClassifierCascade
	//{
	//	int  count;
	//	int  has_tilted_features;
	//	double inv_window_area;
	//	CvMat sum, sqsum, tilted;
	//	CvHidHaarStageClassifier* stage_classifier;
	//	sqsumtype *pq0, *pq1, *pq2, *pq3;
	//	sumtype *p0, *p1, *p2, *p3;

	//	void** ipp_stages;
	//	bool  is_tree;
	//	bool  isStumpBased;
	//} CvHidHaarClassifierCascade;

#define calc_sumf(rect,offset) \
    static_cast<float>((rect).p0[offset] - (rect).p1[offset] - (rect).p2[offset] + (rect).p3[offset])

	namespace cv_haar_avx
	{
#if 0 /*CV_TRY_AVX*/
#define CV_HAAR_USE_AVX 1
#else
#define CV_HAAR_USE_AVX 0
#endif

#if CV_HAAR_USE_AVX
		// AVX version icvEvalHidHaarClassifier.  Process 8 CvHidHaarClassifiers per call. Check AVX support before invocation!!
		double icvEvalHidHaarClassifierAVX(CvHidHaarClassifier* classifier, double variance_norm_factor, size_t p_offset);
		double icvEvalHidHaarStumpClassifierAVX(CvHidHaarClassifier* classifier, double variance_norm_factor, size_t p_offset);
		double icvEvalHidHaarStumpClassifierTwoRectAVX(CvHidHaarClassifier* classifier, double variance_norm_factor, size_t p_offset);
#endif
	}
}

#endif

/* End of file. */
