
#include "BNShm.h"

//tcp 
BNTran_SocketTCPServer tcpSever;

namespace bn 
{
int share_read(frames_t*fs, unsigned char* pbuffer)
{
	if(fs->h != fs->t)
	{
		if(fs->h == 0)
			fs->t = 3;
		else
			fs->t = fs->h - 1;
		//frame_t* f = (frame_t*)&fs->frames[fs->t++];
		//fs->t &= 31;
		frame_t* f = (frame_t*)&fs->frames[fs->t];
		fs->t = fs->h;
		//memcpy(pbuffer, f->data, fs->size);
		memcpy(pbuffer, f, fs->size+sizeof(struct timeval));
		//memcpy(pbuffer, f, fs->size+sizeof(struct timeval)+sizeof(int));
		return 0;
	}
	return -1;
}

int share_write( void* pbuffer,int dataSize)
{
	tcpSever.sendCommandInfo((void*)pbuffer,dataSize);
	return 1;
}
int share_write( S_SOC_DATA & sendData_)
{
	tcpSever.sendCommandInfo(sendData_);
	return 1;
}

void tcp_connect()
{
	tcpSever.run();
}

void share_init(void** ppBuffer)
{
	int shm_id = shmget((key_t)SHARE_KEY, sizeof(frames_t), 0666|IPC_CREAT);
	if(shm_id == -1)  
    {  
		printf("ERROR: shmget failed\n");  
        EXIT(-1); 
    }
	void* shm_addr = shmat(shm_id, (void*)0, 0);  
    if(shm_addr == (void*)-1 || !shm_addr)  
    {  
        printf("ERROR: shmat failed\n");  
        EXIT(-1); 
    } 
	*ppBuffer = shm_addr;
	

}
void share_release(void* shm_addr)
{
	if(shm_addr)
	{
		if(shmdt(shm_addr) == -1)  
		{  
			printf("ERROR: shmdt failed\n");  
			EXIT(-1);  
		}  
	}
}

//void fps_show(Mat* pMat)
//{
//	static char strFPS[16]; 
//	double fps, tt;
//	static double t = (double)cv::getTickCount();
//	static double fs[4] = {0};
//	static int fsi = 0;
//	tt = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
//	t = (double)cv::getTickCount();
//	fps = 1.0 / tt;
//	fs[fsi++] = fps;
//	fsi &= 3;
//	fps = (fs[0] + fs[1] + fs[2] + fs[3]) / 4.0;
//	SPRINTF(strFPS, "FPS: %.2f", fps);
//	putText(*pMat, strFPS, Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
//}

}//namespace bn
