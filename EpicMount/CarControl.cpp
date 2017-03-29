/***************************************************************************
 
    file                 : CarControl.cpp
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
#include "CarControl.h"

// meta-command value for race restart
int CarControl::META_RESTART=1;

CarControl::CarControl(float accel, float brake, int gear, float steer, int meta)
{
	this->accel = accel;
	this->brake = brake;
	this->gear  = gear;
	this->steer = steer;
	this->meta = meta;
}

CarControl::CarControl(float accel, float brake, int gear, float steer)
{
	this->accel = accel;
	this->brake = brake;
	this->gear  = gear;
	this->steer = steer;
	this->meta = 0;
}

CarControl::CarControl(string sensors)
{
        fromString(sensors);
}

string 
CarControl::toString()
{
	string str;
	
	str  = SimpleParser::stringify("accel", accel);
	str += SimpleParser::stringify("brake", brake);
	str += SimpleParser::stringify("gear",  gear);
	str += SimpleParser::stringify("steer", steer);
	str += SimpleParser::stringify("meta", meta);
	
	return str;	
}

void 
CarControl::fromString(string sensors)
{
	if (SimpleParser::parse(sensors, "accel", accel)==false)
		accel=0.0;
	if (SimpleParser::parse(sensors, "brake", brake)==false)
		brake=0.0;
	if (SimpleParser::parse(sensors, "gear",  gear)==false)
		gear=1;
	if (SimpleParser::parse(sensors, "steer", steer)==false)
		steer=0.0;
	if (SimpleParser::parse(sensors, "meta", meta)==false)
		meta=0;
	
}

float 
CarControl::getAccel()
{
        return this->accel;
};

void 
CarControl::setAccel (float accel)
{
        this->accel = accel;
};

float 
CarControl::getBrake()
{
        return this->brake;
};

void 
CarControl::setBrake (float brake)
{
        this->brake = brake;
};

int
CarControl::getGear()
{
        return this->gear;
};

void 
CarControl::setGear(int gear)
{
        this->gear = gear;
};

float 
CarControl::getSteer()
{
        return this->steer;
};

void 
CarControl::setSteer(float steer)
{
        this->steer = steer;
};

int
CarControl::getMeta()
{
        return this->meta;
};

void 
CarControl::setMeta(int meta)
{
        this->meta = meta;
};
