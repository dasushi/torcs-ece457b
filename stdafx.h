//#pragma once
//#pragma warning(disable : 4996)

#include <WinSock.h>
#include <iostream>
#include <tchar.h>
#include <math.h>
#include <windows.h>
//#include <time.h>
#include <wchar.h>
//#include <conio.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <vector>
#include <string>
#include <cstring>
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
	wchar_t filename[260]; //_MAX_PATH = 260
} ENTRY, *PENTRY;

typedef struct _record {
	vector<PENTRY> entries;
	//vector< vector<int> > indices;
	//vector<float> clasnum;
} RECORD, *PRECORD;

