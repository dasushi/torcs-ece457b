#pragma once
#pragma warning(disable : 4996)

#include <iostream>
#include <tchar.h>
#include <math.h>
#include <windows.h>
#include <stdio.h>
#include <time.h>
#include <wchar.h>
#include <conio.h>

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

//2D type array
// size, filename, class type
// N = record.size()
// getting a point: record.entries[i].vec[j]
// [entry, ... , vec, ..] 
// y axis of each vector at each entry
// vector<int> x0;  vector<int> x1; ... vector<int> xClassCount
//
typedef struct _entry {
	float *vec;
	float val;
	int size;
	wchar_t filename[_MAX_PATH];
} ENTRY, *PENTRY;

typedef struct _record {
	vector<PENTRY> entries;
	//vector< vector<int> > indices;
	//vector<float> clasnum;
} RECORD, *PRECORD;

