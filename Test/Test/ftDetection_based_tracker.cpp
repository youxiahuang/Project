#include "ftDetection_based_tracker.h"
#include "IMDetector.h"

#if DEBUGLOGS
#define LOGD(_str, ...) LOGD0(_str , ## __VA_ARGS__)
#define LOGI(_str, ...) LOGI0(_str , ## __VA_ARGS__)
#define LOGW(_str, ...) LOGW0(_str , ## __VA_ARGS__)
#define LOGE(_str, ...) LOGE0(_str , ## __VA_ARGS__)
#else
#define LOGD(...)
#define LOGI(...)
#define LOGW(...)
#define LOGE(...)
#endif //DEBUGLOGS

#define LOGD0(_str, ...) (printf(_str , ## __VA_ARGS__), printf("\n"), fflush(stdout))
#define LOGI0(_str, ...) (printf(_str , ## __VA_ARGS__), printf("\n"), fflush(stdout))
#define LOGW0(_str, ...) (printf(_str , ## __VA_ARGS__), printf("\n"), fflush(stdout))
#define LOGE0(_str, ...) (printf(_str , ## __VA_ARGS__), printf("\n"), fflush(stdout))

using namespace cv;

FTDetectionBaseTracker::FTDetectionBaseTracker(cv::Ptr<IMDetector>mainDetector, cv::Ptr<IMDetector> trackingDetector, const Parameters& params)
:separateDetectionWork(),
parameters(params),
innerParameters(),
numTrackedSteps(0),
cascadeForTracking(trackingDetector)
{
	CV_Assert((params.maxTrackLifetime >= 0)
		//&& mainDetector
		&& trackingDetector);

	if (mainDetector) {
		Ptr<SeparateDetectionWork> tmp(new SeparateDetectionWork(*this, mainDetector, params));
		separateDetectionWork.swap(tmp);
	}

	weightsPositionsSmoothing.push_back(1);
	weightsSizesSmoothing.push_back(0.5);
	weightsSizesSmoothing.push_back(0.3f);
	weightsSizesSmoothing.push_back(0.2f);
}

bool DetectionBasedTracker::run()
{
	if (separateDetectionWork) {
		return separateDetectionWork->run();
	}
	return false;
}

void DetectionBasedTracker::stop()
{
	if (separateDetectionWork) {
		separateDetectionWork->stop();
	}
}

//初始化构造函数SeparateDetectionWork
FTDetectionBaseTracker::SeparateDetectionWork::SeparateDetectionWork(FTDetectionBaseTracker& _detectionBasedTracker, cv::Ptr<IMDetector> _detector,
	const Parameters& params)
	:detectionBasedTracker(_detectionBasedTracker),
	cascadeInThread(),
	isObjectDetectingReady(false),
	shouldObjectDetectingResultsBeForgot(false),
	stateThread(STATE_THREAD_STOPPED),
	timeWhenDetectingThreadStartedWork(-1),
	parameters(params)
{
	CV_Assert(_detector);
	cascadeInThread = _detector;
#ifndef USE_STD_THREADS
	second_workthread = 0;
	int res = 0;
	res = pthread_mutex_init(&mutex, NULL);//TODO: should be attributes?
	if (res) {
		LOGE("ERROR in DetectionBasedTracker::SeparateDetectionWork::SeparateDetectionWork in pthread_mutex_init(&mutex, NULL) is %d", res);
		throw(std::exception());
	}
	res = pthread_cond_init(&objectDetectorRun, NULL);
	if (res) {
		LOGE("ERROR in DetectionBasedTracker::SeparateDetectionWork::SeparateDetectionWork in pthread_cond_init(&objectDetectorRun,, NULL) is %d", res);
		pthread_mutex_destroy(&mutex);
		throw(std::exception());
	}
	res = pthread_cond_init(&objectDetectorThreadStartStop, NULL);
	if (res) {
		LOGE("ERROR in DetectionBasedTracker::SeparateDetectionWork::SeparateDetectionWork in pthread_cond_init(&objectDetectorThreadStartStop,, NULL) is %d", res);
		pthread_cond_destroy(&objectDetectorRun);
		pthread_mutex_destroy(&mutex);
		throw(std::exception());
	}
#endif
}

FTDetectionBaseTracker::SeparateDetectionWork::~SeparateDetectionWork()
{
	if (stateThread != STATE_THREAD_STOPPED) {
		LOGE("\n\n\nATTENTION!!! dangerous algorithm error: destructor DetectionBasedTracker::DetectionBasedTracker::~SeparateDetectionWork is called before stopping the workthread");
	}
#ifndef USE_STD_THREADS
	pthread_cond_destroy(&objectDetectorThreadStartStop);
	pthread_cond_destroy(&objectDetectorRun);
	pthread_mutex_destroy(&mutex);
#else
	second_workthread.join();
#endif
}
bool FTDetectionBaseTracker::SeparateDetectionWork::run()
{
	LOGD("DetectionBasedTracker::SeparateDetectionWork::run() --- start");
#ifdef USE_STD_THREADS
	std::unique_lock<std::mutex> mtx_lock(mtx);
	// unlocked when leaving scope
#else
	pthread_mutex_lock(&mutex);
#endif
	if (stateThread != STATE_THREAD_STOPPED) {
		LOGE("DetectionBasedTracker::SeparateDetectionWork::run is called while the previous run is not stopped");
#ifndef USE_STD_THREADS
		pthread_mutex_unlock(&mutex);
#endif
		return false;
	}
	stateThread = STATE_THREAD_WORKING_SLEEPING;
#ifdef USE_STD_THREADS
	second_workthread = std::thread(workcycleObjectDetectorFunction, (void*)this); //TODO: add attributes?
	objectDetectorThreadStartStop.wait(mtx_lock);
#else
	pthread_create(&second_workthread, NULL, workcycleObjectDetectorFunction, (void*)this); //TODO: add attributes?
	pthread_cond_wait(&objectDetectorThreadStartStop, &mutex);
	pthread_mutex_unlock(&mutex);
#endif
	LOGD("DetectionBasedTracker::SeparateDetectionWork::run --- end");
	return true;
}


//检测线程函数
void* workcycleObjectDetectorFunction(void* p)
{
	//CATCH_ALL_AND_LOG({ ((faceDetect::DetectionBasedTracker::SeparateDetectionWork*)p)->workcycleObjectDetector(); });
	try{
		((FTDetectionBaseTracker::SeparateDetectionWork*)p)->init();
	}
	catch (...) {
		LOGE0("DetectionBasedTracker: workcycleObjectDetectorFunction: ERROR concerning pointer, received as the function parameter");
	}
	return NULL;
}