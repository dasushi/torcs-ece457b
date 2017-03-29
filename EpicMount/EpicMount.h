/* This is a very fast mount.

Written by Shogoth and Xylae.

*/

#ifndef	EPICMOUNT_H
#define EPICMOUNT_H

#include "WrapperBaseDriver.h"
#include "CarState.h"

class EpicMount: public WrapperBaseDriver // SimpleDriver 
{
public:
	// Information about the previous steps and constants that the program might change.
	int buffAngle;
	float prevDamage, prevLapseTime, prevLapseDamage;
	float safeSpeed;
	float sundayDriver;
	float sharpTurn;
	
	// counter of stuck steps
	int stuck;

	EpicMount();

	float distanceAhead(CarState &cs, float angle);
	int   getGear(CarState &cs);
	float getAccel(CarState &cs, float targetAngle);
	float getTargetAngle(CarState &cs);
	float getSteer(CarState &cs, float targetAngle);
	void positionTrack(CarState &cs, float trackPos, float &targetAngle);
	void learnSpeed(CarState cs);
	void computeSharpTurn(CarState &cs, float angle);

	CarControl wDrive(CarState cs);
	
	// Apply an ABS filter to brake command
	float filterABS(CarState &cs,float brake);
	
	// Print a restart message and reset class variables.
	virtual void onRestart();
};

#endif
