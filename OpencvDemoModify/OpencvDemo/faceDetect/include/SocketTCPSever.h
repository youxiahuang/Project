#ifndef _TRAN_SOCKETTCPSERVER_H_
#define _TRAN_SOCKETTCPSERVER_H_

#include <string.h>
//#include <Winsock2.h>
#include <netinet/in.h>    // for sockaddr_in
#include <sys/types.h>    // for socket
#include <sys/socket.h>    // for socket
#include <netinet/tcp.h>   
#include <sys/types.h>    // for socket
#include <arpa/inet.h>

#include "opencv2/core.hpp"


//#pragma  comment(lib,"ws2_32.lib")

#define D_SOCKET_TCP_PORT		6666		/* TCPServer Port */
#define D_BUFFER_SIZE	1024 * 64 -1		/* TCPServer Port */
#define D_SEND_BUFFER_SIZE	640*480+30		/* TCPServer Port */
//#define D_SEND_BUFFER_SIZE 1024


typedef struct 
{
	char faceNum;
	cv::Rect face[1];
	char featNum;
	cv::Point featP[68];
	cv::Vec6d rt_vec;
	float fx;
	float fy;
	float cx;
	float cy;
	int tired;
	uint tv_sec ;
	uint tv_usec ;
}S_SOC_DATA;

class BNTran_SocketTCPServer 
{
public:

	BNTran_SocketTCPServer();
	~BNTran_SocketTCPServer();
	void run();
	void closeRun();
	
	void sendCommandInfo(S_SOC_DATA & sendData_);
	void sendCommandInfo(void * sendData_,int dataSize_);
private:

	void _saveCommandRecvFromTarget(const char* p_commandData,int i4_dataSize);
	
	/* property */
	int sockSrv;					/* server socket */
	int sockConn;				/* connected socket */
	//char _m_recvCommandBuf[D_BUFFER_SIZE];			/* Buffer area to receive data */
	char _m_sendCommandBuf[D_SEND_BUFFER_SIZE];			/* Buffer area to receive data */					
	bool _m_isConnected;					/* client connect flg */
	bool _m_isRun;							/* thread run flg */

	struct sockaddr_in server_addr;
	
};

#endif //_TRAN_SOCKETTCPSERVER_H_
/*↑ ADD 14/02/18、BN LuoXueqiang ********************************************/
