#ifndef _NNDATALOG_H_
#define _NNDATALOG_H_

#include <fstream>

#DEFINE SENSOR_COUNT 19

/*Context: "add some kind of a print statement to the files 
to dump the sensor inputs (~18 track sensors + wheel speed 
sensors etc) and control outputs (steering, brake, accel, 
clutch/gear) to a file for a whole race on one constant track*/

struct DataLogEntry {
	double time;
	double carXSpeed;
	double carBrakeAccel; // [-1,1]
	double carClutch; // [0,1]
	double carRpm;
	double carGear; // actually an integer
	double carSteer; // [-1,1]
	double carAngleToTrack;
	double carTrackOffset; // dist to center of track
	double carTrackSensors[ SENSOR_COUNT ]; // array of distance sensors
};

class NNDataLogger {
public:
	static void InitResetDataFile();
	static void AppendDataRecord(DataLogEntry dataRow);

private:
	NNDataLogger();
};

#endif