/***************************************************************************
 
    file                 : client.cpp
    copyright            : (C) 2007 Daniele Loiacono
 
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
/* Uncomment the following lines under windows */
#define WIN32 // maybe not necessary because already define
#define __DRIVER_CLASS__ EpicMount     // put here the name of your driver class
#define __DRIVER_INCLUDE__ "EpicMount.h" // put here the filename of your driver h\\eader
//#define __DRIVER_CLASS__ SimpleDriver     // put here the name of your driver class
//#define __DRIVER_INCLUDE__ "SimpleDriver.h" // put here the filename of your driver h\\eader

#ifdef WIN32
#include <WinSock.h>
#else
#include <netdb.h>
#include <netinet/in.h>
#include <unistd.h>
#endif

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include __DRIVER_INCLUDE__

/*** defines for UDP *****/
#define UDP_MSGLEN 1000
#define UDP_CLIENT_TIMEUOT 1000000
//#define __UDP_CLIENT_VERBOSE__
/************************/

#ifdef WIN32
typedef sockaddr_in tSockAddrIn;
#define CLOSE(x) closesocket(x)
#define INVALID(x) x == INVALID_SOCKET
#else
typedef int SOCKET;
typedef struct sockaddr_in tSockAddrIn;
#define CLOSE(x) close(x)
#define INVALID(x) x < 0
#endif

class __DRIVER_CLASS__;
typedef __DRIVER_CLASS__ tDriver;


using namespace std;

int main(int argc, char *argv[])
{
    SOCKET socketDescriptor;
    int numRead;
    unsigned long int maxEpisodes;
    unsigned long int maxSteps;	
    unsigned short int serverPort;
    tSockAddrIn serverAddress;
    struct hostent *hostInfo;
    struct timeval timeVal;
    fd_set readSet;
    char buf[UDP_MSGLEN];


#ifdef WIN32 
     /* WinSock Startup */

     WSADATA wsaData={0};
     WORD wVer = MAKEWORD(2,2);
     int nRet = WSAStartup(wVer,&wsaData);

     if(nRet == SOCKET_ERROR)
     {
 	std::cout << "Failed to init WinSock library" << std::endl;
	exit(1);
     }
#endif

    if (argc < 4)
    {
        cout << "Usage " << argv[0] << " <ip> <port> <id> [<maxEpisodes> <maxSteps>]" << endl;
        exit(1);
    }

    if (argc==4)
    {
	maxEpisodes=0;
	maxSteps=0;
    }
	
    if (argc==5)
    {
	maxEpisodes=atoi(argv[4]);
	maxSteps=0;
    }

    if (argc==6)
    {
	maxEpisodes=atoi(argv[4]);
	maxSteps=atoi(argv[5]);
    }


    hostInfo = gethostbyname(argv[1]);
    if (hostInfo == NULL)
    {
        cout << "Error: problem interpreting host: " << argv[1] << "\n";
        exit(1);
    }

    serverPort = atoi(argv[2]);

    cout << "***********************************" << endl;
    cout << "IP: "   << hostInfo    << endl; 
    cout << "PORT: " << serverPort  << endl;
    cout << "ID: "   << argv[3]     << endl;
    cout << "MAX_STEPS: " << maxSteps << endl; 
    cout << "MAX_EPISODES: " << maxEpisodes << endl;
    cout << "***********************************" << endl;
    // Create a socket (UDP on IPv4 protocol)
    socketDescriptor = socket(AF_INET, SOCK_DGRAM, 0);
    if (INVALID(socketDescriptor))
    {
        cerr << "cannot create socket\n";
        exit(1);
    }

    // Set some fields in the serverAddress structure.
    serverAddress.sin_family = hostInfo->h_addrtype;
    memcpy((char *) &serverAddress.sin_addr.s_addr,
           hostInfo->h_addr_list[0], hostInfo->h_length);
    serverAddress.sin_port = htons(serverPort);

    tDriver d;

    bool shutdownClient=false;
    unsigned long curEpisode=0;
    do
    {
        /***********************************************************************************
        ************************* UDP client identification ********************************
        ***********************************************************************************/
        do
        {
            cout << "Sending id to server: " << argv[3] << endl;
            if (sendto(socketDescriptor, argv[3], strlen(argv[3]), 0,
                       (struct sockaddr *) &serverAddress,
                       sizeof(serverAddress)) < 0)
            {
                cerr << "cannot send data ";
                CLOSE(socketDescriptor);
                exit(1);
            }


            // wait until answer comes back, for up to UDP_CLIENT_TIMEUOT micro sec
            FD_ZERO(&readSet);
            FD_SET(socketDescriptor, &readSet);
            timeVal.tv_sec = 0;
            timeVal.tv_usec = UDP_CLIENT_TIMEUOT;
			TIMEVAL ptimeVal;
			ptimeVal.tv_sec = 0;
            ptimeVal.tv_usec = UDP_CLIENT_TIMEUOT;

            if (select(socketDescriptor+1, &readSet, NULL, NULL, &ptimeVal))
            {
                // Read data sent by the solorace server
                memset(buf, 0x0, UDP_MSGLEN);  // Zero out the buffer.
                numRead = recv(socketDescriptor, buf, UDP_MSGLEN, 0);
                if (numRead < 0)
                {
                    cerr << "didn't get response from server...";
                }
		else
		{
                	cout << "Received: " << buf << endl;

                	if (strcmp(buf,"***identified***")==0)
                    		break;
            	}
	      }

        }  while(1);

	unsigned long currentStep=0; 

        while(1)
        {
            // wait until answer comes back, for up to UDP_CLIENT_TIMEUOT micro sec
            FD_ZERO(&readSet);
            FD_SET(socketDescriptor, &readSet);
            timeVal.tv_sec = 0;
            timeVal.tv_usec = UDP_CLIENT_TIMEUOT;
			TIMEVAL ptimeVal;
			ptimeVal.tv_sec = 0;
            ptimeVal.tv_usec = UDP_CLIENT_TIMEUOT;

            if (select(socketDescriptor+1, &readSet, NULL, NULL, &ptimeVal))
            {
                // Read data sent by the solorace server
                memset(buf, 0x0, UDP_MSGLEN);  // Zero out the buffer.
                numRead = recv(socketDescriptor, buf, UDP_MSGLEN, 0);
                if (numRead < 0)
                {
                    cerr << "didn't get response from server?";
                    CLOSE(socketDescriptor);
                    exit(1);
                }

#ifdef __UDP_CLIENT_VERBOSE__
                cout << "Received: " << buf << endl;
#endif

                if (strcmp(buf,"***shutdown***")==0)
                {
                    d.onShutdown();
                    shutdownClient = true;
                    cout << "Client Shutdown" << endl;
                    break;
                }

                if (strcmp(buf,"***restart***")==0)
                {
                    d.onRestart();
                    cout << "Client Restart" << endl;
                    break;
                }
                /**************************************************
                 * Compute The Action to send to the solorace sever
                 **************************************************/

		if ( (++currentStep) != maxSteps)
		{
                	string action = d.drive(string(buf));
                	memset(buf, 0x0, UDP_MSGLEN);
			sprintf(buf,"%s",action.c_str());
		}
		else
			sprintf (buf, "(meta 1)");

                if (sendto(socketDescriptor, buf, strlen(buf)+1, 0,
                           (struct sockaddr *) &serverAddress,
                           sizeof(serverAddress)) < 0)
                {
                    cerr << "cannot send data ";
                    CLOSE(socketDescriptor);
                    exit(1);
                }
#ifdef __UDP_CLIENT_VERBOSE__
                else
                    cout << "Sending " << buf << endl;
#endif
            }
            else
            {
                cout << "** Server did not respond in 1 second.\n";
            }
        }
    } while(shutdownClient==false && ( (++curEpisode) != maxEpisodes) );

    if (shutdownClient==false)
	d.onShutdown();
    CLOSE(socketDescriptor);
#ifdef WIN32
    WSACleanup();
#endif
    return 0;

}
