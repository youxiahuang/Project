#include <iostream>

//#include <opencv2/highgui.hpp>
//#include "opencv2/core/core_c.h"
#include "BNImgProc.h"
#include "BNImgcodecs.h"

#include "SocketTCPSever.h"

using namespace std;
using namespace cv;

#include <stdio.h>

BNTran_SocketTCPServer::BNTran_SocketTCPServer() 
{
	/* Initialization */
	_m_isConnected = false;
	_m_isRun = true;
	//_m_port = i4_port;
	
	//memset(_m_recvCommandBuf, 0, sizeof(_m_recvCommandBuf));
	memset(_m_sendCommandBuf, 0, sizeof(_m_sendCommandBuf));
}

BNTran_SocketTCPServer::~BNTran_SocketTCPServer()
{

}

void BNTran_SocketTCPServer::run()
{
	int i4_recv ;
	
	/* Bind a local port to the socket */
	/*WSADATA wsaData;
	printf("WSAStartup\n");
	int SocketStartRet = WSAStartup(0x0202, &wsaData);
    if (SocketStartRet != 0)
    {
		printf("WSAStartup error: %d\n", WSAGetLastError());
        return;
    }*/
	

    //struct sockaddr_in server_addr;
	memset(&server_addr, 0, sizeof(server_addr));
    //server_addr.sin_addr.S_un.S_addr=htonl(INADDR_ANY);
	server_addr.sin_addr.s_addr=htonl(INADDR_ANY);
	//server_addr.sin_addr.s_addr =inet_addr("192.168.0.101");
    server_addr.sin_family=AF_INET;
    server_addr.sin_port=htons(D_SOCKET_TCP_PORT);
    
	printf("Open sockSrv\n");
    sockSrv=socket(AF_INET,SOCK_STREAM,0);
    //sockSrv=socket(AF_INET,SOCK_DGRAM,0);
    if ( sockSrv < 0 ) 
    {
		printf("Open sockSrv failed: \n");
		return;
    }else{
		int opt =1;
	   setsockopt(sockSrv,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt));
	}

	//_m_serverSocket.bind(_m_port);  ///绑定端口
	printf("bind\n");
    if(bind(sockSrv,(sockaddr*)&server_addr,sizeof(sockaddr)))
	{
		printf("bind error: \n");
		return;
	}

	// Puts the socket into listening state 
	printf("listen\n");
	if(listen(sockSrv,5))
	{
		printf("listen error: \n");
		return;
	}

	socklen_t len=sizeof(server_addr);
		
	if (_m_isRun)
	{
		if(_m_isConnected == false)
		{
			printf("acceptConnection\n");
			sockConn=accept(sockSrv,(sockaddr*)&server_addr,&len);
			//sockConn=accept(sockSrv,(sockaddr*)&server_addr,&len);
			if ( sockConn  < 0  ) 
			{
				printf("acceptConnection failed: \n");
				return;
			}
			_m_isConnected = true;
		} 
	} 
	return;
}

void BNTran_SocketTCPServer::sendCommandInfo(S_SOC_DATA & sendData_)
{
	if(_m_isConnected == false)
	{
		//Client is not connected 
		_m_isRun = true;
		run();
		if(_m_isConnected == false)
		{
			printf("BNTran_SocketTCPClient::run() ERROR\n");
			return;
		}
	}
	char prex[4] = {'r','e','c','t'};
	memcpy(_m_sendCommandBuf,prex,4);
	int datasize = 4;
		
	memcpy(_m_sendCommandBuf+4,(uchar *)&sendData_,sizeof(sendData_));
	datasize += sizeof(sendData_);
	
	
    //int on = 1;
    ////setsockopt(sockConn,IPPROTO_TCP,TCP_QUICKACK,&on,sizeof(int));
    //setsockopt(sockConn,IPPROTO_TCP,TCP_NODELAY,&on,sizeof(int));
    
	//printf("sending message to server ..\n");  
    if( send(sockConn,_m_sendCommandBuf,datasize,0) == -1 )  
    //if( send(sockConn,_m_sendCommandBuf,datasize,MSG_DONTWAIT) == -1 )  
    //if( sendto(sockSrv,_m_sendCommandBuf,datasize,0,(sockaddr*)&server_addr,sizeof(sockaddr)) == -1 ) 
    {  
        perror("error in send \n");
        return;  
    }   
    
    
}

void BNTran_SocketTCPServer::sendCommandInfo(void * sendData_,int dataSize_)
{
   if(_m_isConnected == false)
	{
		/* Client is not connected */
		_m_isRun = true;
		run();
		if(_m_isConnected == false)
		{
			printf("BNTran_SocketTCPClient::run() ERROR\n");
			return;
		}
	}
	memcpy(_m_sendCommandBuf,sendData_,dataSize_);
	
    //int on = 1;
    ////setsockopt(sockConn,IPPROTO_TCP,TCP_QUICKACK,&on,sizeof(int));
    //setsockopt(sockConn,IPPROTO_TCP,TCP_NODELAY,&on,sizeof(int));
    
	//printf("sending message to server ..\n");  
    if( send(sockConn,_m_sendCommandBuf,dataSize_,0) == -1 )  
    {  
        perror("error in send \n");
        return;  
    }   
    
}

void BNTran_SocketTCPServer::_saveCommandRecvFromTarget(const char* p_commandData,int i4_dataSize)
{
	printf("%s\n",p_commandData);
	////TBD;
	FILE *fp;
	fp = fopen("111","wb");
	fwrite(p_commandData,sizeof(char),i4_dataSize,fp);
	fclose(fp);
}

void BNTran_SocketTCPServer::closeRun()
{
	_m_isRun = false;
	return;
}

/*int  main()
{
	//cv::Mat frame = cv::imread("smb://192.168.0.100/share/home/pi/Desktop/document/code/faceDetect/1.bmp");
	//if(frame.empty()){
	//	printf("frame is empty\n");
	//}
	
	//return 1;	
				
	BNTran_SocketTCPServer a;
	a.run();
	return true;	
}*/
