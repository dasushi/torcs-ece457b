#include "NNDataLog.h"
#include <fstream>
#include <iostream>

using namespace std;

void NNDataLogger::InitResetDataFile() {
	remove("NNDataLogOutput.csv");
	
	fstream file;

	file.open("NNDataLogOutput.csv", fstream::out);
	file << "time,";
	for(int i = 0; i < SENSOR_COUNT; i++){
		file << "carTrackSensors" << i << ", ";
	}
	file << "carTrackSensors0, ";
	file << "carAngleToTrack, ";
	file << "carTrackOffset, ";
	file << "carXSpeed, ";
	file << "carBrakeAccel, ";
	file << "carSteer, ";
	file << "carRpm, ";
	file << "carGear" << endl;
	file.close();
}

/*Context: "add some kind of a print statement to the files 
to dump the sensor inputs (~18 track sensors + wheel speed 
sensors etc) and control outputs (steering, brake, accel, 
clutch/gear) to a file for a whole race on one constant track*/

// probably very bad for performance
void NNDataLogger::AppendDataRecord(DataLogEntry dataRow) {
	printf("t: %4.2f, centerSensor: %s, angle: %5.2f, offset: %5.2f, speed: %5.2f, accBrake: %4.2f, steer: %3.2f, RPM: %4.2f, gear: %2.1f \r\n",
		dataRow.time,
		dataRow.carTrackSensors[9],
		//dataRow.carTargetAngle,
		dataRow.carAngleToTrack,
		dataRow.carTrackOffset,
		dataRow.carXSpeed,
		dataRow.carBrakeAccel,
		dataRow.carSteer,
		dataRow.carRpm,
		dataRow.carGear
	);

	fstream file;
	file.open("NNDataLogOutput.csv", fstream::app);
	
	// just write to the file
	file << dataRow.time << ',';
	for(int i = 0; i < SENSOR_COUNT; i++){
		file << dataRow.carTrackSensors[i] << ',';
	}
	file << dataRow.carAngleToTrack << ',';
	file << dataRow.carTrackLeftDist << ',';
	file << dataRow.carTrackRightDist << ',';
	//file << dataRow.carTargetAngle << ',';
	file << dataRow.carSpeed << ',';
	file << dataRow.carBrakeAccel << ',';
	file << dataRow.carSteer << ',';
	file << dataRow.carRpm << ',';
	file << dataRow.carGear;


	file << endl;
	file.close();
}