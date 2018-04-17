/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_OBJDETECT_DBT_HPP
#define OPENCV_OBJDETECT_DBT_HPP

//#include <opencv2/core.hpp>

#include "facetracker.h"
//#include "bnobjdetect.hpp"

//#define D_TRACK_PRO
#define D_USE_TRACKER 

#ifndef CV_CXX11
#  if __cplusplus >= 201103L || (defined(_MSC_VER) && _MSC_VER >= 1800)
#    define CV_CXX11 1
#  endif
#else
#  if CV_CXX11 == 0
#    undef CV_CXX11
#  endif
#endif

// After this condition removal update blacklist for bindings: modules/python/common.cmake
#if defined(__linux__) || defined(LINUX) || defined(__APPLE__) || defined(__ANDROID__) || \
  defined(CV_CXX11)

#include <vector>

using namespace cv;

namespace faceDetect
{

//! @addtogroup objdetect
//! @{

class CV_EXPORTS DetectionBasedTracker
{
    public:
        struct CV_EXPORTS Parameters
        {
            int maxTrackLifetime;
            int minDetectionPeriod; //the minimal time between run of the big object detector (on the whole frame) in ms (1000 mean 1 sec), default=0

            Parameters();
        };

        class IDetector
        {
            public:
                IDetector():
                    minObjSize(96, 96),
                    maxObjSize(INT_MAX, INT_MAX),
                    minNeighbours(2),
                    scaleFactor(1.1f)
                {}

				virtual void detect(const Mat& image, std::vector<Rect>& objects, bool firstflag = true) = 0;
				virtual void setOnlyOneFaceFlag(bool flag) = 0;

				
                void setMinObjectSize(const Size& min)
                {
                    minObjSize = min;
                }
                void setMaxObjectSize(const Size& max)
                {
                    maxObjSize = max;
                }
                Size getMinObjectSize() const
                {
                    return minObjSize;
                }
                Size getMaxObjectSize() const
                {
                    return maxObjSize;
                }
                float getScaleFactor()
                {
                    return scaleFactor;
                }
                void setScaleFactor(float value)
                {
                    scaleFactor = value;
                }
                int getMinNeighbours()
                {
                    return minNeighbours;
                }
                void setMinNeighbours(int value)
                {
                    minNeighbours = value;
                }
                virtual ~IDetector() {}

            protected:
                Size minObjSize;
                Size maxObjSize;
                int minNeighbours;
                float scaleFactor;
        };

        DetectionBasedTracker(Ptr<IDetector> mainDetector, Ptr<IDetector> trackingDetector, const Parameters& params);
        virtual ~DetectionBasedTracker();

        virtual bool run();
        virtual void stop();
        virtual void resetTracking();

        virtual void process(const Mat& imageGray);

        bool setParameters(const Parameters& params);
        const Parameters& getParameters() const;


        typedef std::pair<Rect, int> Object;
        virtual void getObjects(std::vector<Rect>& result) const;
        virtual void getObjects(std::vector<Object>& result) const;

		//目标检测状态
        enum ObjectStatus
        {
            DETECTED_NOT_SHOWN_YET,
            DETECTED,
            DETECTED_TEMPORARY_LOST,
            WRONG_OBJECT
        };

		//目标
        struct ExtObject
        {
            int id;
            Rect location;
            ObjectStatus status;	//目标状态
            ExtObject(int _id, Rect _location, ObjectStatus _status)
                :id(_id), location(_location), status(_status)
            {
            }
        };

        virtual void getObjects(std::vector<ExtObject>& result) const;
        virtual int addObject(const Rect& location); //returns id of the new object

    protected:
        class SeparateDetectionWork;
        Ptr<SeparateDetectionWork> separateDetectionWork;
        friend void* workcycleObjectDetectorFunction(void* p);

		//内部参数
        struct InnerParameters
        {
            int numLastPositionsToTrack;
            int numStepsToWaitBeforeFirstShow;
            int numStepsToTrackWithoutDetectingIfObjectHasNotBeenShown;
            int numStepsToShowWithoutDetecting;

            float coeffTrackingWindowSize;
            float coeffObjectSizeToTrack;
            float coeffObjectSpeedUsingInPrediction;

            InnerParameters();
        };
        Parameters parameters;
        InnerParameters innerParameters;

		//追踪目标
        struct TrackedObject
        {
            typedef std::vector<Rect> PositionsVector;
            PositionsVector lastPositions;

            int numDetectedFrames;
            int numFramesNotDetected;
            int id;

            TrackedObject(const Rect& rect):numDetectedFrames(1), numFramesNotDetected(0)
            {
                lastPositions.push_back(rect);
                id = getNextId();
            };

            static int getNextId()
            {
                static int _id=0;
                return _id++;
            }
        };

        int numTrackedSteps;
        std::vector<TrackedObject> trackedObjects;		//追踪到的目标
        std::vector<float> weightsPositionsSmoothing;
        std::vector<float> weightsSizesSmoothing;
        Ptr<IDetector> cascadeForTracking;
		Rect biggestFace(std::vector<Rect> &faces) const;
        void updateTrackedObjects(const std::vector<Rect>& detectedObjects);
        Rect calcTrackedObjectPositionToShow(int i) const;
        Rect calcTrackedObjectPositionToShow(int i, ObjectStatus& status) const;
        void detectInRegion(const Mat& img, const Rect& r, std::vector<Rect>& detectedObjectsInRegions);

		//lost ,template mathch
		Mat                 m_faceTemplate;
		Mat     getFaceTemplate(const Mat &frame, Rect face);
		void        detectFacesTemplateMatching(const Mat &frame, std::vector<Rect> &faces);
		int64                   m_templateMatchingStartTime = 0;
		int64                   m_templateMatchingCurrentTime = 0;
		double                  m_templateMatchingMaxDuration = 3;
		static const double     TICK_FREQUENCY;
		Rect    doubleRectSize(const Rect &inputRect, const Rect &frameSize) const;
		Rect                m_faceRoi;
		Mat                 m_matchingResult;
        
        
        FaceTracker* ptracker;
		Rect m_trackedFace;

};

//! @} objdetect

} //end of cv namespace
#endif

#endif
