#include "SocketTCPClient.h"

#include <stdio.h>

#define INVALID_SOCKET -1
BNTran_SocketTCPClient::BNTran_SocketTCPClient()
{
	/* Initialization */
	_m_isConnected = false;
	_m_isRun = true;

	memset(_m_sendCommandBuf, 0, sizeof(_m_sendCommandBuf));

}

BNTran_SocketTCPClient::~BNTran_SocketTCPClient()
{

}

void BNTran_SocketTCPClient::run()
{	
	/*WSADATA wsaData;
	int SocketStartRet = WSAStartup(0x0202, &wsaData);
    if (SocketStartRet != 0)
    {
		printf("WSAStartup error: %d\n", WSAGetLastError());
        return;
    }

	char hostname[256];
	int iRet = 0;
	memset(hostname, 0, 256);
	iRet = gethostname(hostname, sizeof(hostname));
	if(iRet != 0 )
	{
	 printf( "get hostname error\n");
	}
	else
	{
		printf("hostname = %s\n", hostname);
		//获得本机IP
		struct hostent *phostent;

		//gethostbyname(hostname);
		if( (phostent = gethostbyname(hostname) ) == NULL )
		{
			printf("gethostbyname error for host:%s/n", ptr);
		   return 0; // 如果调用gethostbyname发生错误，返回1 
		}
		else
		{
			printf("name:%s\nalianses:%s\naddrtype:%d\nlength:%d\n",phostent->h_name,phostent->h_aliases,phostent->h_addrtype,phostent->h_length); 
		  struct sockaddr_in sa; 

		  for(int n=0;phostent->h_addr_list[n];n++) 
		  { 
			 memcpy(&sa.sin_addr.s_addr,phostent->h_addr_list[n],phostent->h_length); 
			 printf("address:%s\n",inet_ntoa(sa.sin_addr)); 
		  } 

		  protoent *pprotoent; 
		   pprotoent=getprotobyname("tcp"); 
		  if(pprotoent==NULL) 
		  { 
			 printf("getprotobyname() failed!!\n"); 
			 //return; 
		  } 
		  else
		  {
			  printf("name:%s\nproto:%d\n",pprotoent->p_name,pprotoent->p_proto); 
			  for(int n=0;pprotoent->p_aliases[n];n++) 
			  { 
				 printf("aliases:%s\n",pprotoent->p_aliases[n]); 
			  } 
		  }

		}

	}*/
	
    sockClient=socket(AF_INET,SOCK_STREAM,0);
	if ( sockClient == INVALID_SOCKET ) 
    {
		printf("Open sockClient failed\n");
		return;
    }
 
    struct sockaddr_in addrSrv;
	memset(&addrSrv, 0, sizeof(addrSrv));
    //addrSrv.sin_addr.S_un.S_addr=inet_addr("192.168.78.133");
    //addrSrv.sin_addr.S_un.S_addr=inet_addr("192.168.0.101");
    //char *hostName = "192.168.0.101";
    addrSrv.sin_addr.s_addr =inet_addr("192.168.0.101");
    //inet_pton(AF_INET,hostName,&addrSrv.sin_addr); 
    addrSrv.sin_family=AF_INET;
    addrSrv.sin_port=htons(D_SOCKET_TCP_PORT);
	printf("addrSrv.sin_port: %d\n",addrSrv.sin_port);
  
	try
	{
		if(_m_isRun)
		{
			while (!_m_isConnected)
			{
				if (_m_isAbleConnected == true && _m_isConnected == false)
				{
					printf("connect\n");
					if (connect(sockClient,(sockaddr*)&addrSrv,sizeof(sockaddr)) == -1)
					{
						printf("connect failed\n");
						//SPCOMTRC_DPRINT( clOutputStr1 );
						return;
					}
					else
					{
						_m_isConnected = true;
						printf("addrSrv.sin_port: %d\n",addrSrv.sin_port);
					}
				}
			}
		}
	} 
	catch (...)
	{
		printf("BNTran_SocketTCPClient::run() ERROR\n");
		_m_isConnected = false;
	} 

	return;
}

void BNTran_SocketTCPClient::sendCommandInfo(void * sendData_,int dataSize_)
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
	
	//printf("sending message to server ..\n");  
        if( send(sockClient,sendData_,dataSize_,0) == -1 )  
        {  
                perror("error in send \n");
                return;  
        }   
        
}
void BNTran_SocketTCPClient::sendCommandInfo()
{
	if(_m_isConnected == false)
	{
		/* Client is not connected */
		_m_isRun = true;
		run();
		if(_m_isConnected == false)
		{
			printf("BNTran_SocketTCPClient::run() ERROR\n");
			//return;
		}
	}
	/*char buffer[1024] = { 0 };
	
	unsigned short i4_bufSize = sizeof(_m_sendCommandBuf);
	buffer[0] = 'E';
	buffer[1] = 'S';
	buffer[2] = 'C';
	buffer[3] = 'M';

	buffer[4] = i4_bufSize;
	buffer[5] = i4_bufSize>>8;

	char j = 0;
	for(int i =0 ;i<255 ;i++)
	{
		memset(_m_sendCommandBuf+256*i, j, 256);
		j++;
	}

	int i4_result;
	try
	{
		printf("send\n");
		i4_result = send(sockClient,buffer,6,0);		
		if (i4_result == SOCKET_ERROR )
		{
			printf("send ERROR: %d\n",WSAGetLastError());
			//SPCOMTRC_DPRINT( clOutputStr1 );
			_m_isConnected = false;
			return ;
		}
		i4_result = recv(sockClient,buffer,3,0);
		if (i4_result == SOCKET_ERROR )
		{
			printf("recv ERROR: %d\n",WSAGetLastError());
			//SPCOMTRC_DPRINT( clOutputStr1 );
			_m_isConnected = false;
			return ;
		}
		else if (strncmp(buffer, "OK", 2) == 0)
		{
			i4_result = send(sockClient,_m_sendCommandBuf,i4_bufSize,0);	
			if (i4_result == SOCKET_ERROR )
			{
				printf("send ERROR: %d\n",WSAGetLastError());
				//SPCOMTRC_DPRINT( clOutputStr1 );
				_m_isConnected = false;
				return ;
			}
		}
	} 
	catch (...)
	{
		printf("send error");
	}*/

	return ;
}

void BNTran_SocketTCPClient::close()
{
	_m_isRun = false;
	//closesocket(sockClient);
	_m_isConnected = false;
	//WSACleanup();
	return;
}

/*void main__()
{
	BNTran_SocketTCPClient a;
	a._m_isAbleConnected = true;
	a.sendCommandInfo();
}*/
