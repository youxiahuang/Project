
#ifndef OPENCV_OBJDETECT_C_H
#define OPENCV_OBJDETECT_C_H

#include "opencv2/core/core_c.h"
#include "BNImgProc.h"

//#include "core/types_c.h"
//#include "core/core_c.h"
//#include<opencv.hpp>

#ifdef __cplusplus
#include <deque>
#include <vector>

extern "C" {
#endif

	/** @addtogroup objdetect_c
	  @{
	  */

	/****************************************************************************************\
	*                         Haar-like Object Detection functions                           *
	\****************************************************************************************/
	namespace faceDetect
	{

#define CV_HAAR_MAGIC_VAL    0x42500000
#define CV_TYPE_NAME_HAAR    "opencv-haar-classifier"

#define CV_IS_HAAR_CLASSIFIER( haar )                                                    \
    ((haar) != NULL &&                                                                   \
    (((const CvHaarClassifierCascade*)(haar))->flags & CV_MAGIC_MASK)==CV_HAAR_MAGIC_VAL)

#define CV_HAAR_FEATURE_MAX  3
#define CV_HAAR_STAGE_MAX 1000

		typedef struct CvHaarFeature
		{
			int tilted;
			struct
			{
				CvRect r;
				float weight;
			} rect[CV_HAAR_FEATURE_MAX];
		} CvHaarFeature;

		typedef struct CvHaarClassifier
		{
			int count;
			CvHaarFeature* haar_feature;
			float* threshold;
			int* left;
			int* right;
			float* alpha;
		} CvHaarClassifier;

		typedef struct CvHaarStageClassifier
		{
			int  count;
			float threshold;
			CvHaarClassifier* classifier;

			int next;
			int child;
			int parent;
		} CvHaarStageClassifier;


		typedef struct CvHidHaarClassifierCascade CvHidHaarClassifierCascade;

		typedef struct CvHaarClassifierCascade
		{
			int  flags;
			int  count;
			CvSize orig_window_size;
			CvSize real_window_size;
			double scale;
			CvHaarStageClassifier* stage_classifier;
			CvHidHaarClassifierCascade* hid_cascade;
		} CvHaarClassifierCascade;

		typedef struct CvAvgComp
		{
			CvRect rect;
			int neighbors;
		} CvAvgComp;


		/* Loads haar classifier cascade from a directory.
		   It is obsolete: convert your cascade to xml and use cvLoad instead */
		CVAPI(CvHaarClassifierCascade*) cvLoadHaarClassifierCascade(
			const char* directory, CvSize orig_window_size);

		CVAPI(void) cvReleaseHaarClassifierCascade(CvHaarClassifierCascade** cascade);

#define CV_HAAR_DO_CANNY_PRUNING    1
#define CV_HAAR_SCALE_IMAGE         2
#define CV_HAAR_FIND_BIGGEST_OBJECT 4
#define CV_HAAR_DO_ROUGH_SEARCH     8

		CVAPI(CvSeq*) cvHaarDetectObjects(const CvArr* image,
			CvHaarClassifierCascade* cascade, CvMemStorage* storage,
			double scale_factor CV_DEFAULT(1.1),
			int min_neighbors CV_DEFAULT(3), int flags CV_DEFAULT(0),
			CvSize min_size CV_DEFAULT(cvSize(0, 0)), CvSize max_size CV_DEFAULT(cvSize(0, 0)));

		/* sets images for haar classifier cascade */
		CVAPI(void) cvSetImagesForHaarClassifierCascade(CvHaarClassifierCascade* cascade,
			const CvArr* sum, const CvArr* sqsum,
			const CvArr* tilted_sum, double scale);

		/* runs the cascade on the specified window */
		CVAPI(int) cvRunHaarClassifierCascade(const CvHaarClassifierCascade* cascade,
			CvPoint pt, int start_stage CV_DEFAULT(0));

		/** @} objdetect_c */
		//}

		}
#ifdef __cplusplus
	}

	CV_EXPORTS CvSeq* cvHaarDetectObjectsForROC_(const CvArr* image,
		faceDetect::CvHaarClassifierCascade* cascade, CvMemStorage* storage,
		std::vector<int>& rejectLevels, std::vector<double>& levelWeightds,
		double scale_factor = 1.1,
		int min_neighbors = 3, int flags = 0,
		CvSize min_size = cvSize(0, 0), CvSize max_size = cvSize(0, 0),
		bool outputRejectLevels = false);

#endif


#endif /* OPENCV_OBJDETECT_C_H */
