#ifndef __BNSHM_H__
#define __BNSHM_H__
#include <stdlib.h>  
#include <stdio.h>  
#include <string.h> 
//#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include <unistd.h>  
#include <sys/shm.h>

#include "SocketTCPSever.h"


#define SHARE_KEY		0x6789 
#define SLEEP(a)		{usleep((a) * 1000);}
#define EXIT(a)			{exit(a);}
#define SPRINTF(a, b, c){sprintf((a), (b), (c));}

using namespace cv;
using namespace std;
#define FRAME_WIDTH		640
#define FRAME_HEIGHT	480
#define DATA_SIZE		(FRAME_WIDTH * FRAME_HEIGHT * 3)
//#define D_SEND_BUFFER_SIZE 1024

typedef struct
{
	//int no;
	struct timeval now;
	unsigned char data[DATA_SIZE];
}frame_t;
typedef struct
{
	int width;
	int height;
	int size;
	int t;
	int h;
	//frame_t frames[32];
	frame_t frames[4];
}frames_t;

namespace bn 
{

int share_read(frames_t*fs, unsigned char* pbuffer);
int share_write( void* pbuffer,int dataSize);

int share_write( S_SOC_DATA & sendData_);

void tcp_connect();

void share_init(void** ppBuffer);
void share_release(void* shm_addr);

}//namespace bn
#endif //__BNSHM_H__
