#include "stdafx.h"
#include "trace.h"

#include "LibNN\BPNeuralNetwork.h"
#include "LibNN\BPNeuron.h"


//accuracy structure for storing accuracy tallies
typedef struct _accuracy {
	float mse; 
	float acc;
} ACCURACY, *PACCURACY;

wchar_t normalization_type[][20] = {L"zscore",L"minmax",L"energy",L"sigmoidal"};
enum NORMALIZATION {NONE, ZSCORE, MINMAX, ENERGY, SIGMOIDAL};

class BPNeuralNetwork *neuralNet = 0;

int normalization_val = 0;
int validation_type = 0;
int vector_dim = 6;

//train, validate, and test methods
void train(int argc, wchar_t* argv[]);
void validate(PRECORD record, float TH, float *accuracy, PACCURACY paccuracy);
void set_validation(PRECORD valid, PRECORD train, float p);
void split_train(PRECORD newRecord, PRECORD train, float p);
void set_normalization(RECORD *record, BPNeuralNetwork *networkPtr);
void dump_sets(PRECORD train, PRECORD valid, PRECORD test);
void test(int argc, wchar_t* argv[]);
float gmean(float i, int j);

//IO methods
void get_file_name(wchar_t *path, wchar_t *name);
void read_file(FILE *filepointer, PRECORD record);
void msec_to_time(int msec, int &hour, int &minute, int &second, int &msOut); 
int read_line(wchar_t *buffer, FILE *file, int *c = 0);
int parse_path(wchar_t *path, wchar_t *directory, wchar_t *filename);


int _tmain(int argc, wchar_t* argv[]){
	srand((unsigned int)time(0));

	if (argc == 1) {
		
		wprintf(L" Usage:\n");
		wprintf(L"  ann1dn.exe t net.nn datafile 3000 [norm [0]] [err [0.05]] \n");
		wprintf(L"  ann1dn.exe r net.nn testcls [norm [0]]\n\n");
		wprintf(L"\n argv[1] t - train\n");
		wprintf(L" [2] network conf file\n");
		wprintf(L" [3] input.csv data file \n");
		//wprintf(L" [4] cls2 output data files [0.1]\n");
		wprintf(L" [4] epochs count\n");
		//wprintf(L" [5] opt [validation TH 0.5]\n");
		//wprintf(L" [6] opt [validation metric mse]\n");
		wprintf(L" [5] opt [norm]: [0-no], 1-zscore, 2-minmax, 3-energy, 4-softmax\n");
		wprintf(L" [6] opt [error tolerance cls]: default +- 0.05 \n\n");

		wprintf(L" argv[1] r - run\n");
		wprintf(L" [2] network conf file\n");
		wprintf(L" [3] input.csv file\n");
		//wprintf(L" [4] opt [validation TH 0.5]\n");
		wprintf(L" [4] opt [norm]: [0-no], 1-zscore, 2-minmax, 3-energy, 4-softmax\n\n");


		/*wprintf(L" metrics: [0 - mse, optional]\n");
		wprintf(L"           1 - AC\n");
		wprintf(L"           2 - sqrt(SE*SP)\n");
		wprintf(L"           3 - sqrt(SE*PP)\n");
		wprintf(L"           4 - sqrt(SE*SP*AC)\n");
		wprintf(L"           5 - sqrt(SE*SP*PP*NP*AC)\n");
		wprintf(L"           6 - F-measure b=1\n");
		wprintf(L"           7 - F-measure b=1.5\n");
		wprintf(L"           8 - F-measure b=3\n");*/
	} else if ( !wcscmp(argv[1], L"t") )
		//train flag supplied
	        train(argc, argv);
	else if ( !wcscmp(argv[1], L"r") )
		//test flag supplied
	        test(argc, argv);
	else
		//wrong flag used
	        wprintf(L"argv[1] t-train, r-run\n");

	return 0;
}


void train(int argc, wchar_t *argv[]){
	RECORD validateRecord;
	RECORD testRecord;
	RECORD trainRecord;

	//acceptable error
	float error = 0.05f;
	float TH = 0.5f;
	bool validation = false;
	bool test = false;

	//using optional arguments
	if(argc > 5){
		normalization_val = _wtoi(argv[5]);
		if(argc > 6){
			error = float(_wtof(argv[6]));
		}
	}

	wprintf(L"Loading CLS data\n");
	FILE *inputData;

	inputData = _wfopen(argv[3], L"rt");

	if(!(inputData)){
		wprintf(L"Loading Input data file failed: %s\n", argv[3]);
	} else {
		read_file(inputData, &trainRecord);
	}

	if(!(trainRecord.entries.size() > 0)){
		wprintf(L"Train Record is empty: %s %s\n", argv[3], argv[4]);
		exit(1);
	} 

	wprintf(L"Train Class file loaded. Count: %d samples\n",
		trainRecord.entries.size());

	//set test-train ratio depending on validation & test data
	//25% test, 75% train
	split_train(&testRecord, &trainRecord, 25.0f);
	//26.25% validate, 48.75% train, 25% test
	split_train(&validateRecord, &trainRecord, 35.0f);
	
	dump_sets(&trainRecord, &validateRecord, &testRecord);

	//create network with specified filename
	wchar_t* networkName = argv[2];
	neuralNet = new BPNeuralNetwork(networkName); 

	if(neuralNet->flag() < 0){
		wprintf(L"Loading network %s failed\n", argv[2]);
		exit(1);
	} else if (neuralNet->get_layer(0)->get_neuron_count()
					!= vector_dim){
			if(neuralNet->get_layer(0)->get_neuron_count() 
					> vector_dim){
				wprintf(L"Input layer dim %d neurons is bigger than data dim %d\n", 
					neuralNet->get_layer(0)->get_neuron_count(), 
					vector_dim);
				exit(1);
			} else {
				wprintf(L"Input layer dim %d neurons are smaller than data dim %d\n", 
					neuralNet->get_layer(0)->get_neuron_count(), 
					vector_dim);
			}
	}

	if(normalization_val != 0){
		//energy normalization exception
		//if(normalization_val != 3){
			wprintf(L"Starting normalization %d\n",normalization_val);
			set_normalization(&trainRecord, neuralNet);
		//}
	}

	wprintf(L"Training network\n");
	//initialize vars for training loop
	int x = 0;
	int y = 0;
	int i = 0;
	int index = 0;
	float val = 0.0f;
	int epoch = 0;
	int size = 0;
	int epoch_count = 0;
	int max_epoch = 0;
	int done = 0;
	bool previous = false;

	float desired_vec[1] = {0.0f};
	float *in_vec;
	float out_vec[1] = {0.0f};
	float out_vec1[1] = {0.0f};
	float accuracy = 0.0f;
	float tmp_accuracy = 0.0f;

	ACCURACY paccuracy;
	ACCURACY tempaccuracy;
	memset(&paccuracy, 0, sizeof(ACCURACY));
	memset(&tempaccuracy, 0, sizeof(ACCURACY));

	//int currentTime = GetTickCount();

	size = (int)trainRecord.entries.size();
	wchar_t maxName[_MAX_PATH] = L"max";
	wcscat(maxName, networkName); 

	epoch_count = _wtoi(argv[4]);
	int per_epoch = size / epoch_count;
	//epoch = epoch_count / size;
	for(int epoch = 0; epoch < epoch_count; epoch++){
		for(int run = 0; run < per_epoch; run++){
			in_vec = trainRecord.entries.at(epoch * per_epoch + run)->vec;
			val = trainRecord.entries.at(epoch * per_epoch + run)->val;
			desired_vec[0] = val;
			//wprintf(L"Epoch: %d record: %d val: %f\n", epoch, epoch * per_epoch + run, val);

			neuralNet->train(in_vec, out_vec, desired_vec, error);
			out_vec1[0] += out_vec[0];
		}

		float mul_out1 = out_vec1[0];
		out_vec1[0] = 0.0f;

		if(validateRecord.entries.size()){
			//if(mul_out1 > TH){// && mul_out2 < TH){
				validate(&validateRecord, TH, &tmp_accuracy, &tempaccuracy); 
				if(tmp_accuracy >= accuracy){
					accuracy = tmp_accuracy;
					max_epoch = epoch;
					memcpy(&paccuracy, &tempaccuracy, sizeof(ACCURACY));
					if(!neuralNet->save(maxName)){
						wprintf(L"Failed to save maxaccuracy.nn results\n");
					}
				}
				//Print max accuracy record
				wprintf(L"Max accuracy: %.2f (epoch %d) mse: %.2f\n", 
					accuracy, max_epoch, paccuracy.mse);//, paccuracy.sqe,
					//paccuracy.spec, paccuracy.acc);
			//} else {
			//	wprintf(L"\n");
			//}
		} else {
			wprintf(L"\n");
		}


		//for(int j = 0; j < (int)trainRecord.indices.size(); j++){
		//	random_shuffle(trainRecord.entries.begin(), trainRecord.entries.end());
		//}
	}

	//int hour;
	//int min;
	//int second;
	//int msecond;
	//msec_to_time(GetTickCount() - currentTime, hour, min, second, msecond);
	if(epoch){
		wprintf(L"Training completed!\n");
	}
	if(!neuralNet->save(argv[2])){
		wprintf(L"Failed to save network %s \n", argv[2]);
	}
	wprintf(L"maxNetwork: %s\n", maxName);
	BPNeuralNetwork* maxAccuracy = new BPNeuralNetwork(maxName);
	if(maxAccuracy->flag() == 0){
		wprintf(L"\n\nMaxAccuracy classification results:\n");
		validate(&trainRecord, TH, &accuracy, &paccuracy);
		wprintf(L"Training set: %d, mse: %.2f, acc: %.2f\n", 
			trainRecord.entries.size(),
			paccuracy.mse, paccuracy.acc); //paccuracy.spec, paccuracy.pp, 
			//paccuracy.np, paccuracy.acc);
		if(validateRecord.entries.size()){
			validate(&validateRecord, TH, &accuracy, &paccuracy);
			wprintf(L"Validation set: %d, mse: %.2f, acc: %.2f\n", 
			validateRecord.entries.size(),
			paccuracy.mse, paccuracy.acc);//, paccuracy.spec, paccuracy.pp, 
			//paccuracy.np, paccuracy.acc);
			
		}
		if(testRecord.entries.size()){
			validate(&testRecord, TH, &accuracy, &paccuracy);
			wprintf(L"Test set: %d, mse: %.2f, acc: %.2f\n", 
			testRecord.entries.size(),
			paccuracy.mse, paccuracy.acc);
			
		}
	} else {
		wprintf(L"Failed loading %s max network during classification with flag %d\n", maxName, maxAccuracy->flag());
	}
	neuralNet = new BPNeuralNetwork(networkName);
	if(neuralNet->flag() == 0){
		wprintf(L"\n\nBPNetwork classification results:\n");
		validate(&trainRecord, TH, &accuracy, &paccuracy);
		wprintf(L"Training set: %d, mse: %.2f, acc: %.2f\n", 
			trainRecord.entries.size(), 
			paccuracy.mse, paccuracy.acc);//, paccuracy.spec, paccuracy.pp, 
			//paccuracy.np, paccuracy.acc);
		if(validateRecord.entries.size()){
			validate(&validateRecord, TH, &accuracy, &paccuracy);
			wprintf(L"Validation set: %d, mse: %.2f, acc: %.2f\n", 
				validateRecord.entries.size(), 
				paccuracy.acc);//, paccuracy.spec, paccuracy.pp, 
			//paccuracy.np, paccuracy.acc);
			
		}
		if(testRecord.entries.size()){
			validate(&testRecord, TH, &accuracy, &paccuracy);
			wprintf(L"Test set: %d , mse: %.2f, acc: %.2f\n", 
				testRecord.entries.size(), 
				paccuracy.mse, paccuracy.acc);//, paccuracy.spec, paccuracy.pp, 
			//paccuracy.np, paccuracy.acc);
			
		}
	} else {
		wprintf(L"Failed loading network %s during classification with flag %d\n", argv[2], neuralNet->flag());
	}
}

void validate(PRECORD record, float TH, float *accuracy, PACCURACY paccuracy){
	float mse = 0.0f; //meansquared error
	float sqe = 0.0f; //squared error
	//float spec = 0.0f; //specificity
	//float pp = 0.0f; //positive predicition
	//float np = 0.0f; //neg preditiction
	float acc = 0.0f; //accuracy
	float runningAcc = 0.0f;
	//float b;
	float *in_vec; //input vector
	float out_vec[1] = {0.0f};
	int size;
	int TOTALPOS = 0;
	int UNDERSHOOT = 0;
	int OVERSHOOT = 0;
	//int FALSEPOS = 0;
	//int TOTALNEG = 0;
	//int FALSENEG = 0;
	size = (int)record->entries.size();
	//test loop
	for(int i = 0; i < (int)record->entries.size(); i++){
		if(record->entries[i] == 0){
			size--;
			continue;
		}
		in_vec = record->entries[i]->vec;
		neuralNet->classify(in_vec, out_vec);
		//this is a 2-class classification validation test
		//TODO: modify to using scalar/continuous output value, directly compare 
		//		to scalar validation value
		float pred_val = out_vec[0];
		float compare_val = record->entries.at(i)->val;

		if(compare_val){
			mse += (pred_val - compare_val) * (pred_val - compare_val);
			if(fabs(compare_val - pred_val) < 0.05f) { //bingo!	
				TOTALPOS++;
			} else { //error
				if(pred_val < compare_val){
					UNDERSHOOT++;
				} else if (pred_val > compare_val){
					OVERSHOOT++;
				}
			}
		}
	}
	mse = 100.0f * (mse / (float)size);
	//mean squared error
	*accuracy = 1.0f / mse;
	paccuracy->acc = 100.0f * (TOTALPOS / (UNDERSHOOT + OVERSHOOT));
	paccuracy->mse = mse;
	wprintf(L"MSE Accuracy: %.2f Close: %d Under: %d Over: %d \n", mse, TOTALPOS, UNDERSHOOT, OVERSHOOT);
	


	/*if(TOTALPOS){
		sqe = float (TOTALPOS) / float (TOTALPOS + FALSENEG);
		pp = float (TOTALPOS) / float (TOTALPOS + FALSEPOS);
	}
	if(TOTALNEG){
		spec = float (TOTALNEG) / float (TOTALNEG + FALSEPOS);
		np = float (TOTALNEG) / float (TOTALNEG + FALSENEG);
	}
	if(TOTALPOS || FALSEPOS || FALSENEG || TOTALNEG){
		acc = float (TOTALPOS + TOTALNEG) / 
			float (TOTALPOS + FALSENEG + TOTALNEG + FALSEPOS);
	}
	spec *= 100.0f;
	sqe *= 100.0f;
	pp *= 100.0f;
	np *= 100.0f;
	acc *= 100.0f;
	paccuracy->spec = spec;
	paccuracy->sqe = sqe;
	paccuracy->np = np;
	paccuracy->pp = pp;
	paccuracy->acc = acc;

	if(validation_type == 0){
	} else if (validation_type == 1){
		//accuracy
		*accuracy = acc;
	} else if (validation_type == 2){
		//geometric mean squared error, specificity
		*accuracy = gmean(sqe * spec, 2);
	} else if (validation_type == 3){
		//geometric mean squared error, positive pred
		*accuracy = gmean(sqe * pp, 2);
	} else if (validation_type == 4){
		//geometric mean squared error, specificity, accuracy
		*accuracy = gmean(sqe * spec * acc, 3);
	} else if (validation_type == 5){
		//geometric mean all terms
		*accuracy = gmean(sqe * spec * acc * np * pp, 5);
	} else if (validation_type == 6){
		b = 1.0f;
	} else if (validation_type == 7){
		b = 1.5f;
	} else if (validation_type == 8){
		b = 3.0f;
	}
	if (validation_type >= 6){
		if(sqe && pp){
			*accuracy = ((b * b + 1) * pp * sqe) / (sqe * pp * b * b);
		} else {
			*accuracy = 0.0;
		}
	}*/
}
void test(int argc, wchar_t* argv[]){
	RECORD testRecord;
	
	wprintf(L"Starting to load test data...\n");

	float TH = 0.5f;
	if(argc >= 5){
		TH = (float)_wtof(argv[4]);
	}
	if(argc >= 6){
		normalization_val = _wtoi(argv[5]);
	}
	FILE *inputData = _wfopen(argv[3], L"rt");
	if(inputData){
		read_file(inputData, &testRecord);
	} else {
		wprintf(L"Error loading test record %s\n", argv[3]);
		exit(1);
	}
	if(testRecord.entries.size()){
		wprintf(L"Loaded %d test entries, size of %d samples\n", 
			testRecord.entries.size(), testRecord.entries[0]->size);
	} else {
		wprintf(L"Error, no files loaded from %s\n", argv[3]);
		exit(1);
	}
	neuralNet = new BPNeuralNetwork(argv[2]);

	//check network status
	if(neuralNet->flag()){
		wprintf(L"Loading network %s failed\n", argv[2]);
		exit(1);
	} else if (neuralNet->get_layer(0)->get_neuron_count()
					!= vector_dim){
			if(neuralNet->get_layer(0)->get_neuron_count() 
					> vector_dim){
				wprintf(L"Input layer dim %d neurons is bigger than data dim %d\n", 
					neuralNet->get_layer(0)->get_neuron_count(), 
					vector_dim);
				exit(1);
			} else {
				wprintf(L"Input layer dim %d neurons are smaller than data dim %d\n", 
					neuralNet->get_layer(0)->get_neuron_count(), 
					vector_dim);
			}
	} else {
		wprintf(L"Finished loading network %s\n", argv[2]);
	}

	float *in_vec;
	float out_vec[1] = {0.0f};
	wchar_t filename[_MAX_PATH] = L"";
	wchar_t dirname[_MAX_PATH] = L"";

	float mse = 0.0f; //meansquared error
	//float sqe = 0.0f; //squared error
	//float spec = 0.0f; //specificity
	//float pp = 0.0f; //positive predicition
	//float np = 0.0f; //neg preditiction
	float acc = 0.0f; //accuracy
	
	int TOTALPOS = 0;
	int UNDERSHOOT = 0;
	int OVERSHOOT = 0;
	//int TOTALNEG = 0;
	//int FALSEPOS = 0;
	//int FALSENEG = 0;
	int size = (int)testRecord.entries.size();
	//test loop
	for(int i = 0; i < (int)testRecord.entries.size(); i++){
		
		in_vec = testRecord.entries[i]->vec;
		neuralNet->classify(in_vec, out_vec);
		if(parse_path(testRecord.entries[i]->filename, dirname, filename)){
			wprintf(L"dir: %s  ", dirname);
		}
		//this is a 2-class classification validation test
		//TODO: modify to using scalar/continuous output value, directly compare 
		//		to scalar validation value
		float pred_val = out_vec[0];
		wprintf(L"name: %s  val %f\n", filename, pred_val);

		float test_val = testRecord.entries[i]->val;

		if(test_val){
			mse += (pred_val - test_val) * (pred_val - test_val);
			if(fabs(test_val - pred_val) < 0.05f) { //bingo!	
				TOTALPOS++;
			} else { //error
				if(pred_val < test_val){
					UNDERSHOOT++;
				} else if (pred_val > test_val){
					OVERSHOOT++;
				}
			}
		}
	}
	mse = 100.0f * mse / (float)size;
	acc = mse;
	wprintf(L"MSE Accuracy: %.2f Close: %d Under: %d Over: %d \n", mse, TOTALPOS, UNDERSHOOT, OVERSHOOT);

	/*if(TOTALPOS){
		sqe = float (TOTALPOS) / float (TOTALPOS + FALSENEG);
		sqe *= 100.0f;
		pp = float (TOTALPOS) / float (TOTALPOS + FALSEPOS);
		pp *= 100.0f;
		wprintf(L"Sensitivity: %.2f\n", sqe);
		wprintf(L"Pos Predictions: %.2f\n", pp);
	}
	if(TOTALNEG){
		spec = float (TOTALNEG) / float (TOTALNEG + FALSEPOS);
		spec *= 100.0f;
		np = float (TOTALNEG) / float (TOTALNEG + FALSENEG);
		np *= 100.0f;
		wprintf(L"Specificity: %.2f\n", spec);
		wprintf(L"Neg Predictions: %.2f\n", np);
	}
	if(TOTALPOS || FALSEPOS || FALSENEG || TOTALNEG){
		acc = float (TOTALPOS + TOTALNEG) / 
			float (TOTALPOS + FALSENEG + TOTALNEG + FALSEPOS);
		acc *= 100.0f;
		wprintf(L"Accuracy: %.2f\n", acc);
	}*/

}

void split_train(PRECORD newRecord, PRECORD trainRecord, float p){
	int newSize = int((p/100.0f) * (float)trainRecord->entries.size());
	wprintf(L"Train set size: %d new Size: %d\n", trainRecord->entries.size(), newSize);
	if(newSize < 1){
		wprintf(L"Error: one of the validation sets is of 0 length\n");
		return;
	}
	PENTRY swap;
	newRecord->entries.resize(newSize);
	for(int i = 0; i < newSize; i++){
		//move i'th element from train record to new record
		swap = trainRecord->entries.at(i);
		newRecord->entries.push_back(swap);
		//delete ith element
		trainRecord->entries.erase(trainRecord->entries.begin() + i);
	}
	//trainRecord->entries.resize(trainRecord->entries.size() - newSize);
}

void set_validation(PRECORD valRecord, PRECORD trainRecord, float p){
	int clas1 = int((p/100.0f) * (float)trainRecord->entries.size());
	wprintf(L"Validation set size: clas1 %d \n", clas1);
	if(clas1 < 1){
		wprintf(L"Error: one of the validation sets is of 0 length\n");
		return;
	}
	valRecord->entries.resize(clas1);
	//valRecord->clasnum.push_back(trainRecord->clasnum[0]);
	//valRecord->clasnum.push_back(trainRecord->clasnum[1]);

	vector<int> indices;
	indices.resize(clas1);

	//shuffle both class vectors
	random_shuffle(trainRecord->entries.begin(), trainRecord->entries.end());
	//random_shuffle(trainRecord->indices[1].begin(), trainRecord->indices[1].end());

	/*valRecord->indices.push_back(indices);
	for(int i = 0; i < clas1; i++){
		int index = trainRecord->indices[0].at(i);
		valRecord->indices[0].at(i) = i;
		valRecord->entries[i] = trainRecord->entries[index];
		valRecord->entries[index] = 0;
	}	
	trainRecord->indices[0].erase(trainRecord->indices[0].begin(),
			trainRecord->indices[0].begin() + clas1);*/

	/*indices.resize(clas2);
	valRecord->indices.push_back(indices);
	for(int i = 0; i < clas2; i++){
		int index = trainRecord->indices[1].at(i);
		valRecord->indices[1].at(i) = i + clas1;
		valRecord->entries[i + clas1] = trainRecord->entries[index];
		valRecord->entries[index] = 0;
	}	
	trainRecord->indices[1].erase(trainRecord->indices[1].begin(),
			trainRecord->indices[1].begin() + clas2);
	*/
}

float gmean(float m, int n){
	float power = 1.0f / (float) n;
	return pow(m, power);
}

void dump_sets(PRECORD trainRecord, PRECORD valRecord, PRECORD testRecord){
	wchar_t filename[_MAX_PATH] = L"";
	wchar_t dirname[_MAX_PATH] = L"";

	FILE *filepointer = _wfopen(L"dbgsets.txt", L"wt");

	if(trainRecord){
		size_t size = 0;

		for(size_t iter = 0; iter < trainRecord->entries.size(); iter++){
			if(trainRecord->entries[iter] != 0){
				size++;
			}
		}
		fwprintf(filepointer, L"[Training Set]: %d\n", size);
		if(trainRecord->entries.size() < 1000){
			for(size_t i = 0; i < trainRecord->entries.size(); i++){
				if(trainRecord->entries[i] != 0){
					fwprintf(filepointer, L"n: %s  val: %d\n", 
						trainRecord->entries[i]->filename, 
						trainRecord->entries[i]->val);
				}
			}
		}
	}
	if(valRecord){
		fwprintf(filepointer, L"[Validation Set]: %d\n", valRecord->entries.size());
		if(valRecord->entries.size() < 1000){
			for(size_t i = 0; i < valRecord->entries.size(); i++){
				if(valRecord->entries[i] != 0){
					fwprintf(filepointer, L"n: %s  val: %d\n", 
						valRecord->entries[i]->filename, 
						valRecord->entries[i]->val);
				}
			}
		}
	}
	if(testRecord){
		fwprintf(filepointer, L"[Test Set]: %d\n", testRecord->entries.size());
		if(testRecord->entries.size() < 1000){
			for(size_t i = 0; i < testRecord->entries.size(); i++){
				if(testRecord->entries[i] != 0){
					fwprintf(filepointer, L"n: %s  val: %d\n", 
						testRecord->entries[i]->filename, 
						testRecord->entries[i]->val);
				}
			}
		}
	}
	fclose(filepointer);
}

////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////data loading routines/////////////////////////////////////////////////////////
int read_line(FILE *f, wchar_t *buff, int *c)
{
        wint_t res = 0;
        wchar_t *pbuff = buff;

        while ((short)res != EOF) {
                res = fgetwc(f);
                if (res == 0xD || res == 0xA) {
                        if (pbuff == buff) continue;

                        *pbuff = 0;
                        if (!c) {
                                return 1;
                        } else {
                                int ptr = (int)wcslen(buff) - 1;

                                while (ptr > 0) {  //skip thru 'spaces'      dir/asds.f ___1__ \n
                                        if (buff[ptr] != 0x20) break;
                                        else ptr--;
                                }
                                while (ptr > 0) {  //skip thru 'clas type'
                                        if (buff[ptr] == 0x20) break;
                                        else ptr--;
                                }

                                if (ptr) {
                                        *c = _wtoi(&buff[ptr+1]);
                                        while (buff[ptr] == 0x20)  //remove blanks from end of string
                                                buff[ptr--] = 0;
                                } else
                                        *c = 0;

                                return 1;
                        }
                }
                if ((short)res != EOF) {
                        *pbuff++ = (char)res;
                }
        }

        return (short)res;
}

/*
    format         //data stored in this file - [name] [val]
													[vec1] [vec2] ...
     file1 [val]
      [vec1] [vec2] ...
     file2 [val]
      [vec1] [vec2]...
     ..
     read class data to PRECORD struct
                                        */
void read_file(FILE *filepointer, PRECORD record)
{
        //wchar_t ustr[_MAX_PATH], *pstr;
        int res = 1;
		int vector_dim = 6;
		//int clas;

		std::ifstream data(filepointer);
		std::string line;
		//throw away first title line
		std::getline(data, line);
		while(std::getline(data, line)){
			std::stringstream linestream(line);
			std::string cellContents;
            PENTRY entry = new ENTRY;
			int cnt = 0;
			float *fdata = new float[vector_dim];
			while(std::getline(linestream, cellContents, ',')){
				if(cnt == vector_dim){
					entry->val = (float)atof(cellContents.c_str());
				} else {
					fdata[cnt] = (float)atof(cellContents.c_str());
				}
				cnt++;

			}
            entry->vec = fdata;
            entry->size = vector_dim;
            record->entries.push_back(entry);
		}

        //int entrsize = (int)record->entries.size();   //size from previous read iteration
        fclose(filepointer);
}

//////////////////data loading routines//////////////////

/////////////////////get normalization params////////////////////////////////////////////////////////////
void set_normalization(RECORD *record, BPNeuralNetwork *network)
{
        int N = int(record->entries.size());
        int I = -1;  //first nonzero entry
        for (int i = 0; i < (int)record->entries.size(); i++) {
                if (record->entries[i] == 0)
                        N--;
                else if (I == -1)
                        I = i;
        }

        float tmp, min, max;
        vector<float> vmean, vdisp, vmin, vmax;

        //////////get mean, max,min of every feature////////////
        for (int x = 0; x < vector_dim; x++) {
                float *ivec = record->entries[I]->vec;
                tmp = 0.0f;
                min = ivec[x];
                max = ivec[x];
                for (int y = 0; y < (int)record->entries.size(); y++) {
                        if (record->entries[y] == 0) continue;

                        ivec = record->entries[y]->vec;
                        tmp += ivec[x];

                        if (ivec[x] > max) max = ivec[x];
                        if (ivec[x] < min) min = ivec[x];
                }

                vmean.push_back(tmp / float(N));
                vmax.push_back(max);
                vmin.push_back(min);
        }

        ///////get std of every feature////////////////////////
        for (int x = 0; x < vector_dim; x++) {
                float *ivec;
                tmp = 0.0f;
                for (int y = 0; y < (int)record->entries.size(); y++) {
                        if (record->entries[y] == 0) continue;

                        ivec = record->entries[y]->vec;
                        tmp += (ivec[x] - vmean[x]) * (ivec[x] - vmean[x]);
                }

                tmp = sqrt(tmp / float(N - 1));
                vdisp.push_back(tmp);
        }



        ///////write normalization coeffs to input layer///////
        for (int n = 0; n < network->get_layer(0)->get_neuron_count(); n++) {
                network->get_layer(0)->get_neuron(n)->get_input_link(0)->set_add_val(-vmin[n]);
                switch (normalization_val) {
                case MINMAX:
                        network->get_layer(0)->get_neuron(n)->set_function_type(BPNeuron::LINEAR);                        
                        if (fabs(float(vmax[n] - vmin[n])) != 0.0f)
                                network->get_layer(0)->get_neuron(n)->get_input_link(0)->set_weight(1.0f / (vmax[n] - vmin[n]));                                
                        break;

                default:
                case ZSCORE:  //zscore
                        network->get_layer(0)->get_neuron(n)->set_function_type(BPNeuron::LINEAR);                                                
                        if (vdisp[n] != 0.0f)
                                network->get_layer(0)->get_neuron(n)->get_input_link(0)->set_weight(1.0f / vdisp[n]);                                
                        break;

                case SIGMOIDAL:  //sigmoidal
                        network->get_layer(0)->get_neuron(n)->set_function_type(BPNeuron::SIGMOID);                                                
                        if (vdisp[n] != 0.0f)
                                network->get_layer(0)->get_neuron(n)->get_input_link(0)->set_weight(1.0f / vdisp[n]);                                
                        break;
                }
        }

}
void get_file_name(wchar_t *path, wchar_t *name)
{
        int sl = 0, dot = (int)wcslen(path);
        int i;
        for (i = 0; i < (int)wcslen(path); i++) {
                if (path[i] == '.') break;
                if (path[i] == '\\') break;
        }
        if (i >= (int)wcslen(path)) {
                wcscpy(name, path);
                return;
        }

        for (i = (int)wcslen(path) - 1; i >= 0; i--) {
                if (path[i] == '.')
                        dot = i;
                if (path[i] == '\\') {
                        sl = i + 1;
                        break;
                }
        }

        memcpy(name, &path[sl], (dot - sl)*2);
        name[dot-sl] = 0;
}

int parse_path(wchar_t *path, wchar_t *dir, wchar_t *name)   //true if dirs equal
{
        int res;
        int i;
        for (i = (int)wcslen(path) - 1; i > 0; i--) {
                if (path[i] == '\\')
                        break;
        }

        if (i) { //path + name
                wcscpy(name, &path[i+1]);
                path[i] = 0;
                res = wcscmp(dir, path);
                wcscpy(dir, path);
        } else { //no path
                res = wcscmp(dir, L"");
                wcscpy(dir, L"");
                wcscpy(name, path);
        }
        return res;   //res=0 if dir and path\filename are equal
}

void msec_to_time(int msec, int& hour, int& minute, int& second, int& msOut)
{
        msOut = msec % 1000;
        msec /= 1000;

        if (msec < 60) {
                hour = 0;
                minute = 0;
                second = msec;                 //sec to final
        } else {
                float tmp;
                tmp = (float)(msec % 60) / 60;
                tmp *= 60;
                second = int(tmp);
                msec /= 60;

                if (msec < 60) {
                        hour = 0;
                        minute = msec;
                } else {
                        hour = msec / 60;
                        tmp = (float)(msec % 60) / 60;
                        tmp *= 60;
                        minute = int(tmp);
                }
        }
}