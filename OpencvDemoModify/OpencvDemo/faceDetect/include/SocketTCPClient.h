#ifndef _TRAN_SOCKETTCPCLIENT_H_
#define _TRAN_SOCKETTCPCLIENT_H_

#include <string.h>
//#include <Winsock2.h>
#include <netinet/in.h>    // for sockaddr_in
#include <sys/types.h>    // for socket
#include <sys/socket.h>    // for socket
#include <arpa/inet.h>

//#pragma  comment(lib,"ws2_32.lib")

#define D_SOCKET_TCP_PORT		6666		/* TCPServer Port */
#define D_BUFFER_SIZE	1024 * 64-1		/* TCPServer Port */

class BNTran_SocketTCPClient 
{
public:

	BNTran_SocketTCPClient();
	~BNTran_SocketTCPClient();
	void run();
	void close();
	void sendCommandInfo();
	void sendCommandInfo(void * sendData_,int dataSize_);

	bool _m_isAbleConnected;

private:

	/* property */
	int sockClient;	/* client socket */
	//SOCKET sockConn; /* connected socket */
	char _m_sendCommandBuf[D_BUFFER_SIZE];	/* Buffer area to send data */
	bool _m_isConnected;					/* client connect flg */
	bool _m_isRun;							/* thread run flg */

};

#endif //_TRAN_SOCKETTCPCLIRNT_H_
