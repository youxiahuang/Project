#ifndef FTDETECTION_BASED_TRACKER_H_
#define FTDETECTION_BASED_TRACKER_H_
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

#ifdef CV_CXX11
#define USE_STD_THREADS
#endif

// After this condition removal update blacklist for bindings: modules/python/common.cmake
#if defined(__linux__) || defined(LINUX) || defined(__APPLE__) || defined(__ANDROID__) || \
	defined(CV_CXX11)

#ifdef USE_STD_THREADS
#include <thread>
#include <mutex>
#include <condition_variable>
#else
#include <pthread.h>
#endif

#include <opencv2/opencv.hpp>
#include <vector>
#include "IMDetector.h"

using namespace cv;
using namespace std;

struct CV_EXPORTS Parameters
{
	int maxTrackLifetime;
	int minDetectionPeriod; //the minimal time between run of the big object detector (on the whole frame) in ms (1000 mean 1 sec), default=0

	Parameters();
};
class CV_EXPORTS FTDetectionBaseTracker
{
public:
	FTDetectionBaseTracker(Ptr<IMDetector> mainDetector, Ptr<IMDetector> trackingDetector, const Parameters& params);
	virtual ~FTDetectionBaseTracker();

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

		TrackedObject(const Rect& rect) :numDetectedFrames(1), numFramesNotDetected(0)
		{
			lastPositions.push_back(rect);
			id = getNextId();
		};

		static int getNextId()
		{
			static int _id = 0;
			return _id++;
		}
	};

	int numTrackedSteps;
	std::vector<TrackedObject> trackedObjects;		//追踪到的目标
	std::vector<float> weightsPositionsSmoothing;
	std::vector<float> weightsSizesSmoothing;
	Ptr<IMDetector> cascadeForTracking;
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


	//FaceTracker* ptracker;
	Rect m_trackedFace;

};

//class:SeparateDetectionWork
class FTDetectionBaseTracker::SeparateDetectionWork
{
public:
	SeparateDetectionWork(FTDetectionBaseTracker& _detectionBasedTracker, cv::Ptr<IMDetector> _detector,
		const Parameters& params);
	virtual ~SeparateDetectionWork();
	bool communicateWithDetectingThread(const Mat& imageGray, std::vector<Rect>& rectsWhereRegions);
	bool run();
	void stop();
	void resetTracking();

	inline bool isWorking()
	{
		return (stateThread == STATE_THREAD_WORKING_SLEEPING) || (stateThread == STATE_THREAD_WORKING_WITH_IMAGE);
	}
	void setParameters(const Parameters& params)
	{
#ifdef USE_STD_THREADS
		std::unique_lock<std::mutex> mtx_lock(mtx);
#else
		pthread_mutex_lock(&mutex);
#endif
		parameters = params;
#ifndef USE_STD_THREADS
		pthread_mutex_unlock(&mutex);
#endif
	}

	inline void init()
	{
#ifdef USE_STD_THREADS
		std::unique_lock<std::mutex> mtx_lock(mtx);
#else
		pthread_mutex_lock(&mutex);
#endif
		stateThread = STATE_THREAD_STOPPED;
		isObjectDetectingReady = false;
		shouldObjectDetectingResultsBeForgot = false;
#ifdef USE_STD_THREADS
		objectDetectorThreadStartStop.notify_one();
#else
		pthread_cond_signal(&(objectDetectorThreadStartStop));
		pthread_mutex_unlock(&mutex);
#endif
	}
protected:
	FTDetectionBaseTracker& detectionBasedTracker;
	cv::Ptr<IMDetector> cascadeInThread;
#ifdef USE_STD_THREADS
	std::thread second_workthread;
	std::mutex mtx;
	std::condition_variable objectDetectorRun;
	std::condition_variable objectDetectorThreadStartStop;
#else
	pthread_t second_workthread;
	pthread_mutex_t mutex;
	pthread_cond_t objectDetectorRun;
	pthread_cond_t objectDetectorThreadStartStop;
#endif
	std::vector<cv::Rect> resultDetect;
	volatile bool isObjectDetectingReady;
	volatile bool shouldObjectDetectingResultsBeForgot;

	enum StateSeparatedThread {
		STATE_THREAD_STOPPED = 0,
		STATE_THREAD_WORKING_SLEEPING,
		STATE_THREAD_WORKING_WITH_IMAGE,
		STATE_THREAD_WORKING,
		STATE_THREAD_STOPPING
	};
	volatile StateSeparatedThread stateThread;
	cv::Mat imageSeparateDetecting;
	void workcycleObjectDetector();
	friend void* workcycleObjectDetectorFunction(void* p);
	long long  timeWhenDetectingThreadStartedWork;
	Parameters parameters;
};
#endif

#endif