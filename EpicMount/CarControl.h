/***************************************************************************
 
    file                 : CarControl.h
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
#ifndef CARCONTROL_H_
#define CARCONTROL_H_

#include <iostream>
#include <sstream>
#include <cstring>
#include <cassert>
#include "SimpleParser.h"

using namespace std;

class CarControl
{
private:

        // Accelerate command [0,1]
        float accel;

        // Brake command [
        float brake;

        // Gear command
        int gear;
        
        // Steering command [-1,1]
        float steer;
        
        // meta-command
        int meta;
        

public:

        CarControl(){};

        CarControl(string sensors);

        CarControl(float accel, float brake, int gear, float steer,int meta);
        CarControl(float accel, float brake, int gear, float steer);

        string toString();

        void fromString(string sensors);

        /* Getter and setter methods */

        float getAccel();
        
        void setAccel (float accel);
        
        float getBrake();
        
        void setBrake (float brake);
        
        int getGear();
        
        void setGear(int gear);
        
        float getSteer();
        
        void setSteer(float steer);  
        
        int getMeta();
        
        void setMeta(int gear);
        
	// meta-command value for race restart 
        static int META_RESTART;
};

#endif /*CARCONTROL_H_*/
