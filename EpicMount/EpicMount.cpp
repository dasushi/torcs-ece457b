/* This is a very fast mount.

Written by Shogoth and Xylae.

*/
#include <cmath>
#include <iostream>
using namespace std;

#include "EpicMount.h"
#include "NNDataLog.h"

//Constant constants
const float PI=3.1415926536;
const float PI12 = PI/2;
const float deg10 = PI/18.0;
const float maxTurn=0.785398;
const float ourSteerSensitivityOffset=80.0;
const float ourWheelSensitivityCoeff=1;

/* Stuck constants*/
const int ourStuckTime = 25;
const float ourStuckAngle = .523598775; //PI/6

/* Constants copied from SimpleDriver */
/* Stuck constants*/
const int stuckTime = 25;
const float stuckAngle = .523598775; //PI/6

/* Accel and Brake Constants*/
const float maxSpeedDist=70;
const float maxSpeed=150;
const float sin10 = 0.17365;
const float cos10 = 0.98481;

/* Steering constants*/
const float steerLock=0.785398;
const float steerSensitivityOffset=80.0;
const float wheelSensitivityCoeff=1;

/* ABS Filter Constants */
const float wheelRadius[4]={0.3179,0.3179,0.3276,0.3276};
const float absSlip=2.0;
const float absRange=3.0;
const float absMinSpeed=3.0;

// Variable constants
const float almostStraight = deg10;//0.05236; // approximately 3 degrees.
const float pedalToTheMetal = 5000;
const float floorItDist = 80; 
const float floorItAngle = deg10;
const float safelyInsideTrack = 0.9;
const float maxSensor = 100;
const float steerOutsideCurve = 0.5*deg10;
const float steerCoef = 0.2;
const int ourGearUp[6]= {8000,7500,7000,7000,7000,0};
const int ourGearDown[6]= {0,2500,3000,3000,3500,3500};
const float tooClose = 10; // meters
const int minSafeSpeed = 50;
const float steerAmp = 1.2;
const float safeSharpTurn = 0.4;

EpicMount::EpicMount()
	: WrapperBaseDriver() // SimpleDriver()
{
	NNDataLogger::InitResetDataFile();
	onRestart();
}

// Print a restart message and reset class variables.
void EpicMount::onRestart()
{
	buffAngle = 0;
	prevDamage = 0;
	prevLapseTime=-1;
	prevLapseDamage = 0;
	safeSpeed = 130;
	sundayDriver = 500;
}

// This is basically the same function as in SimpleDriver, except that we redefined the 
// RPM values for changing the gear up. We're waiting longer to be able to accelerate faster.s
int EpicMount::getGear(CarState &cs)
{
    int gear = cs.getGear();
    int rpm  = cs.getRpm();

    // if gear is 0 (N) or -1 (R) just return 1 
    if (gear<1)
        return 1;
    // check if the RPM value of car is greater than the one suggested 
    // to shift up the gear from the current one     
    if (gear <6 && rpm >= ourGearUp[gear-1])
        return gear + 1;
    else
    	// check if the RPM value of car is lower than the one suggested 
    	// to shift down the gear from the current one
        if (gear > 1 && rpm <= ourGearDown[gear-1])
            return gear - 1;
        else // otherwhise keep current gear
            return gear;
}

// Converts an angle from degrees to radians.
float deg2rad(float angle)
{
	return angle*PI/180.0;
}

// Converts an angle from radians to degrees.
float rad2deg(float angle)
{
	return angle*180.0/PI;
}

// Finds out which index in the trackPos array of the sensors corresponds
// to a given angle. The angle is relative to the car orientation.
int indexTrack(float angle)
{
	return int((angle + PI12)*18/PI);
}

// Finds out which index in the opponents array of the sensors corresponds
// to a given angle. The angle is relative to the car orientation.
int indexOpponent(float angle)
{
	return int((angle + PI)*35/(2*PI));
}

// Find the minimal distance between adjacent sensors as an indication of
// a sharp turn coming up ahead. The angle is the target angle by which
// we expect to be turning.
void EpicMount::computeSharpTurn(CarState &cs, float angle)
{
	float localSharpTurn, dist;
	sharpTurn = 0;
	for (float i=-2*deg10 + angle; i< 2*deg10 + angle; i+= deg10)
	{
		float first, second;
		first = distanceAhead(cs, i);
		second = distanceAhead(cs, i+deg10);
		if (first != -1 && second != -1)
		{
			// Find the absolute value of the difference between the distance ahead in two directions.
			// The coefficient of sharp turn is 1/(1+dist). We're looking for the minimal such
			// difference indicating that the road goes in a new direction. So we're finding the
			// maximum of this coefficient and storing it.
			dist = fabs(first - second);
			if (first < 90 && second < 90) 
			{
				localSharpTurn = 1.0/(1.0+dist);
				if (localSharpTurn > sharpTurn)
					sharpTurn = localSharpTurn;
			}
		}
	}
	//cout << "sharp turn" << sharpTurn << endl;
}

// Gives us the available distance ahead in the direction given by this angle.
float EpicMount::distanceAhead(CarState &cs, float angle)
{
	float currentAngle = cs.getAngle();
	float relAngle = currentAngle-angle;
	if (fabs(relAngle) > PI12) // outside of the track sensors, don't know
		return -1;
	float low, high, distTrack, distOpponent, ratio;
	// Approximate the distance ahead in the direction of the angle as given by the track array.
	low = indexTrack(relAngle);
	if (low == 18)
		high = low;
	else 
		high = low+1;
	ratio = (relAngle + PI12 - deg10*low)/deg10;
	distTrack = (1-ratio) * cs.getTrack(low) + ratio * cs.getTrack(high);
	return distTrack;

	// Compute the same thing in the opponent array.
	/*
	low = indexOpponent(relAngle);
	if (low == 35)
		high = 0;
	else 
		high = low+1;
	ratio = (relAngle + PI - deg10*low)/deg10;
	distOpponent = (1-ratio) * cs.getOpponents(low) + ratio * cs.getOpponents(high);
	return distTrack <= distOpponent ? distTrack : distOpponent;
	*/
}

// Deciding on the speed based on the direction in which we want the car to go. 
// That is given as the parameter targetAngle, and is decided on elsewhere.
float EpicMount::getAccel(CarState &cs, float targetAngle)
{
    // checks if car is out of track
    if (cs.getTrackPos() < 1 && cs.getTrackPos() > -1)
    {
        float targetSpeed, currentAngle;

		//currentAngle = cs.getAngle();

        // track is straight and enough far from a turn so goes to max speed
        if (fabs(targetAngle) < almostStraight && distanceAhead(cs, 0)>= floorItDist && 
			sharpTurn < safeSharpTurn)
            targetSpeed = pedalToTheMetal;
        else
        {
            // computing approximately the "angle" of turn
            float sinAngle = sin(fabs(targetAngle));
			float freeSpace = distanceAhead(cs, targetAngle);
			//float freeStraight = distanceAhead(cs, 0); // free space straight ahead of us
			//float aveSpace = (freeSpace+freeStraight)/2.0;
			float aveSpace = freeSpace;
            // estimate the target speed depending on turn and on how close it is
            targetSpeed = safeSpeed + 
				         (sundayDriver-safeSpeed)*sinAngle* (25+aveSpace)/(25+maxSensor);
			// Mitigate the target speed by the sharp turn factor
			targetSpeed = targetSpeed * (1 - sharpTurn);
			//cout << "speed " << cs.getSpeedX() << " ts " << targetSpeed << endl;
        }
		//return targetSpeed;
        // accel/brake command is expontially scaled w.r.t. the difference between target speed and current one
        return 2.0/(1+exp(cs.getSpeedX() - targetSpeed)) - 1;
    }
    else
        return 0.3; // when out of track returns a moderate acceleration command
}

// This function will adjust the target angle based on the desired position with respect to the 
// center of the track.
void EpicMount::positionTrack(CarState &cs, float trackPos, float &targetAngle)
{
	float centerAngle;
	// if the car is too far to the left
	if (trackPos >= safelyInsideTrack) 
	{
		centerAngle = -10*(trackPos-safelyInsideTrack) * deg10;
		//cout << "cr " << centerAngle << endl;
		if (centerAngle < targetAngle)
			targetAngle = centerAngle;
	}
	// if the car is too far to the right
	else if (trackPos <= -safelyInsideTrack) 
	{
		centerAngle = -10*(trackPos+safelyInsideTrack) * deg10;
		//cout << "cl " << centerAngle << endl;
		if (centerAngle > targetAngle)
			targetAngle = centerAngle;
	}
	// The next code was supposed to 
	//else if (targetAngle < 0) 
	//{
	//	if (targetAngle > -steerOutsideCurve && trackPos < safelyInsideTrack) // steer the car towards the outside of the curve
	//	{
	//		centerAngle = steerCoef*(safelyInsideTrack - trackPos)* deg10;
	//		targetAngle += centerAngle;
	//	}
	//	else if (targetAngle < -steerOutsideCurve && trackPos > -safelyInsideTrack)
	//	{
	//		centerAngle = steerCoef*(trackPos-safelyInsideTrack)* deg10;
	//		targetAngle += centerAngle;
	//	}
	//}
	//else // targetAngle > 0
	//{
	//	if (targetAngle < steerOutsideCurve && trackPos > -safelyInsideTrack) // steer the car towards the outside of the curve
	//	{
	//		centerAngle = steerCoef*(trackPos - safelyInsideTrack)* deg10;
	//		targetAngle += centerAngle;
	//	}
	//	else if (targetAngle < -steerOutsideCurve && trackPos < safelyInsideTrack)
	//	{
	//		centerAngle = steerCoef*(safelyInsideTrack - trackPos)* deg10;
	//		targetAngle += centerAngle;
	//	}
	//}
}

float EpicMount::getTargetAngle(CarState &cs)
{
	register float targetAngle, centerAngle;
	float dist, dist1, dist2;
	float currentAngle = cs.getAngle();	
	float trackPos = cs.getTrackPos();
	if (fabs(currentAngle) <= floorItAngle && distanceAhead(cs, 0)>= floorItDist && fabs(trackPos) < safelyInsideTrack)
		return 0;
	targetAngle = currentAngle;
	if (targetAngle > maxTurn)
		targetAngle = maxTurn;
	else if (targetAngle < -maxTurn)
		targetAngle = -maxTurn;
	dist = distanceAhead(cs, targetAngle);
	dist1 = distanceAhead(cs, targetAngle + deg10);
	dist2 = distanceAhead(cs, targetAngle - deg10);
	if (dist >= dist1 && dist >= dist2)
		;
	else if (dist1 > dist2) 
		while (targetAngle+deg10 <= maxTurn && dist1 >= dist2) 
		{
			targetAngle += deg10;
			dist2 = dist1;
			dist1 = distanceAhead(cs, targetAngle+deg10);
		}
	else 
		while (targetAngle-deg10 >= -maxTurn && dist2 >= dist1) 
		{
			targetAngle -= deg10;
			dist1 = dist2;
			dist2 = distanceAhead(cs, targetAngle - deg10);
		}
	//cout << "ta " << rad2deg(targetAngle) << " ca " << rad2deg(currentAngle) << endl;
	//cout << "ta " << rad2deg(targetAngle) << " tp " << trackPos << endl;
	// If the car is too close to the border of the road then make it move to the inside.
	positionTrack(cs, trackPos, targetAngle);
	computeSharpTurn(cs, targetAngle);

	return steerAmp*targetAngle;
	//targetAngle += buffAngle;
	//if (fabs(targetAngle) > 0.8*deg10) {
	//	buffAngle = 0;
	//	return targetAngle;
	//}
	//else {
	//	buffAngle = targetAngle;
	//	return 0;
	//}
}
		
float EpicMount::getSteer(CarState &cs, float targetAngle)
{
	 // at high speed reduce the steering command to avoid loosing the control
	//return targetAngle;
    if (cs.getSpeedX() > ourSteerSensitivityOffset)
	{
		//cout << "if " << (targetAngle)/maxTurn << endl;
		return (targetAngle)/maxTurn;
        //return targetAngle/(maxTurn*(cs.getSpeedX()-ourSteerSensitivityOffset)*ourWheelSensitivityCoeff);
	}
	else
	{
		//cout << "el " << targetAngle << endl;
        return targetAngle/maxTurn;
	}
}

// Adapts the reference speeds of the car to the situation on the road so far. 
// 1. If we notice that the car was dammaged and there is no other car in the
// close vicinity, it means we bumped against the shoulder and we're going too fast.
// 2. If we managed to complete an entire lapse without new dammage, maybe we can increase 
// the speed a little bit.
// 3. If the car is stuck and there is no other car close, maybe we're going too fast again.
// We might need to check this because we might skid without hitting a wall and that can 
// also mean that we're going too fast.
void EpicMount::learnSpeed(CarState cs)
{
	static bool damageLast = false;
	float damage = cs.getDamage();
	float lapseTime = cs.getCurLapTime();
	
	if (damage > prevDamage || stuck == 1) 
	{
		bool carClose = false;
		for (int i=0; i<36 && !carClose; i++)
		{
			//cout << "c" << i << " " << cs.getOpponents(i) << endl;
			if (fabs(cs.getOpponents(i)) < tooClose)
				carClose = true;
		}
		if (!carClose)
		{
			if (!damageLast) 
				safeSpeed -= 10;
			damageLast = true;
			//cout << "ss " << safeSpeed << endl;
		}
	}
	else if (prevLapseTime > lapseTime) 
	{
		if (prevLapseDamage == damage)
			safeSpeed += 5;
	}
	else
		damageLast = false;
	// Update the globals after we're done dealing with them.
	if (prevLapseTime > lapseTime) 
		prevLapseDamage = damage;
	prevDamage = damage;
	prevLapseTime = lapseTime;

	if (safeSpeed < minSafeSpeed)
		safeSpeed = minSafeSpeed;
}

float EpicMount::filterABS(CarState &cs,float brake)
{
	// convert speed to m/s
	float speed = cs.getSpeedX() / 3.6;
	// when spedd lower than min speed for abs do nothing
    if (speed < absMinSpeed)
        return brake;
    
    // compute the speed of wheels in m/s
    float slip = 0.0f;
    for (int i = 0; i < 4; i++)
    {
        slip += cs.getWheelSpinVel(i) * wheelRadius[i];
    }
    // slip is the difference between actual speed of car and average speed of wheels
    slip = speed - slip/4.0f;
    // when slip too high applu ABS
    if (slip > absSlip)
    {
        brake = brake - (slip - absSlip)/absRange;
    }
    
    // check brake is not negative, otherwise set it to zero
    if (brake<0)
    	return 0;
    else
    	return brake;
}

CarControl EpicMount::wDrive(CarState cs)
{
	static int starting = 0;
	//learnSpeed(cs);
	sharpTurn = 0;
	//computeSharpTurn(cs);
	// check if car is currently stuck
	if ( fabs(cs.getAngle()) > ourStuckAngle )
    {
		// update stuck counter
        stuck++;
    }
    else
    {
    	// if not stuck reset stuck counter
        stuck = 0;
    }

	// after car is stuck for a while apply recovering policy
    if (stuck > ourStuckTime)
    {
    	/* set gear and sterring command assuming car is 
    	 * pointing in a direction out of track */
    	
    	// to bring car parallel to track axis
        float steer = - cs.getAngle() / steerLock; 
        int gear=-1; // gear R
        
        // if car is pointing in the correct direction revert gear and steer  
        if (cs.getAngle()*cs.getTrackPos()>0)
        {
            gear = 1;
            steer = -steer;
        }
        // build a CarControl variable and return it
        CarControl cc (1.0,0.0,gear,steer);
        return cc;
    }

    else // car is not stuck
    {		
		int gear;
		float targetAngle;
		float accel_and_brake;
		if (starting < 10) 
		{
			//cout << "starting" << endl;
			gear = 1;
			targetAngle = 0;
			accel_and_brake = 1;
			starting++;
		}
		else {
			gear = getGear(cs);
			targetAngle = getTargetAngle(cs);
			//cout << "ta " << targetAngle << endl;
			accel_and_brake = getAccel(cs, targetAngle);
		}
		//cout << "acc " << accel_and_brake << endl;
        //int gear = getGear(cs);
        // compute steering

        float steer = getSteer(cs, targetAngle);

        // normalize steering
        if (steer < -1)
            steer = -1;
        if (steer > 1)
            steer = 1;
        
        // set accel and brake from the joint accel/brake command 
        float accel,brake;
        if (accel_and_brake>0)
        {
            accel = accel_and_brake;
            brake = 0;
        }
        else
        {
            accel = 0;
            // apply ABS to brake
            brake = filterABS(cs,-accel_and_brake);
        }
		if ((steer == 1 || steer == -1) && brake == 0)
		{
			brake = .1515;
		}
        //cout << "steer " << steer << " brake " << brake << endl;
        // build a CarControl variable and return it
        CarControl cc(accel,brake,gear,steer);
		DataLogEntry logEntry;
		logEntry.time = cs.getCurLapTime();
		for(int i = 0; i < 19; i++){
			logEntry.carTrackSensors[i] = cs.getTrack(i) / 200.0f;
		}
		logEntry.carXSpeed = cs.getSpeedX() / 150; // max observed speed is 252
		//fuse the brake [0,0.5] and accel [0.5,1] signal from [0,1] brake and [0,1] accel
		logEntry.carBrakeAccel = brake == 0 ? (accel + 1) / 2 : (-brake + 1) / 2; 
		// logEntry.carClutch = car->ctrl.clutchCmd; // always zero
		logEntry.carRpm = cs.getRpm() / 8200;
		logEntry.carGear = gear / 8.0; // 10 - neutral - reverse
		logEntry.carSteer = (steer + 1) / 2; //from -1, 1 to [0,2] to [0,1]
		logEntry.carAngleToTrack = ((cs.getAngle()/ PI) + 1) / 2; //from [-pi,pi] to [-1,1] to [0,2] to [0,1]
		
		//normally roughly from [-1, 1]
		logEntry.carTrackOffset = cs.getTrackPos();
	
		//logEntry.carTargetAngle = targetAngle / maxAngleRad; // already calculated
		
	
		NNDataLogger::AppendDataRecord(logEntry);
        return cc;
    }
}