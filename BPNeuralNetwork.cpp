#include "stdafx.h"
#include "BPNeuralNetwork.h"
#include "BPNeuron.h"

#include <wchar.h>
#include <stdio.h>


BPNeuralLayer::BPNeuralLayer(int neurons_count): m_neuron_count(neurons_count)
{
	for (int i = 0; i < m_neuron_count; i++){
		neurons.push_back(new BPNeuron());
	}
}

BPNeuralLayer::~BPNeuralLayer(){
	for (int i = 0; i < m_neuron_count; i++){
		delete neurons[i];
	}
}

BPNeuralNetwork::BPNeuralNetwork(int layer_count, int *neurons_per_layer): m_flag(0), m_nval(0.2f), m_alpha(0.7f){
	
	m_layer_count = layer_count;

	for(int i = 0; i < m_layer_count; i++){
		layers.push_back(new BPNeuralLayer(neurons_per_layer[i]));
	}

}

BPNeuralNetwork::BPNeuralNetwork(const wchar_t *filename): m_flag(-1), m_nval(0.2f), m_alpha(0.7f){
	int response = 0;
	int nval = 0;
	int in_func = 0;
	int h_func = 0;
	float w = 0.0f;
	size_t charSize = sizeof(char) * wcslen(filename);
	char *charName = (char *) malloc(charSize);
	wcstombs(charName, filename, charSize);
	FILE *filepointer = fopen(charName, "rt");
	
	//if file loaded successfully
	if (filepointer) {
		//scan layer count
		if((response = fwscanf(filepointer, L"%d", &m_layer_count)) != 1){
			fclose(filepointer);
			//wprintf(L"Failed loading network during layer count\n");
			m_flag = -1; //loading failed
			return;
		}
		for(int i = 0; i < m_layer_count; i++){
			//scan layer types
			if((response = fwscanf(filepointer, L"%d", &nval)) != 1){
				fclose(filepointer);
				//wprintf(L"Failed loading network during layer types\n");
				m_flag = -1; //loading failed
				return;
			} else {
				//push input layer with same type
				layers.push_back(new BPNeuralLayer(nval));
			}
		}
		//scan input and hidden function types
		if((response = fwscanf(filepointer, L"%d %d", &in_func, &h_func)) != 2){
			//default linear input function
			in_func = 0;
			//default sigmoid hidden function
			h_func = 1;
		} 

		std::vector<float> add_layers, mult_layers;
		for(int i = 0; i < layers[0]->get_neuron_count(); i++){
			float add, mult;
			//scan add and multiplier values
			if((response = fwscanf(filepointer, L"%f %f", &add, &mult)) != 2){
				for(int j = 0; j < layers[0]->get_neuron_count(); j++){
					//default add and multi values
					add_layers.push_back(0.0f);
					mult_layers.push_back(1.0f);
				}
			}else{
				add_layers.push_back(add);
				mult_layers.push_back(mult);
			}
		}
		//initialize loaded values
		init_links(&add_layers[0], &mult_layers[0], in_func, h_func);

		for(int i = 1; i < m_layer_count; i++){
			for(int j = 0; j < layers[i]->get_neuron_count(); j++){
				for(int k = 0; k < layers[i]->neurons[j]->get_input_link_count(); k++){
					if((response = fwscanf(filepointer, L"%f", &w)) != 1){
						m_flag = 1; //flag = 1 for randomized weights
						//wprintf(L"Randomizing weights\n");
						randomize_weights((unsigned int) time(0));
						return; //init random values since loading failed
					} else {
						//set properly loaded value
						//wprintf(L"Loaded weight\n");
						layers[i]->neurons[j]->inputs[k]->w = w;
					}
				}
			}
		}
		
		fclose(filepointer);
		m_flag = 0;
	} else {
		m_flag = -1;
	}
}

BPNeuralNetwork::~BPNeuralNetwork(){
	for(int i = 0; i < m_layer_count; i++){
		delete layers[i];
	}
}

//set weights on all neurons to randomized values based on seed value
void BPNeuralNetwork::randomize_weights(unsigned int random_seed){
	int weight;

	srand(random_seed);

	for(int i = 0; i < get_layer_count(); i++){
		for(int j = 0; j < layers[i]->get_neuron_count(); j++){
			for(int k = 0; k < layers[i]->neurons[j]->get_input_link_count(); k++){
				weight = 0xFFF & rand();
				weight -= 0x800;
				layers[i]->neurons[j]->inputs[k]->w = (float) weight / 2048.0f;
			}
		}
	}
}


//default values:
// in_func = 0, h_func = 1
//input layer: func = linear, add = 0, w = 1
//hidden & output layers: func = sigmoid, in_val = 1, w = 0

void BPNeuralNetwork::init_links(const float *add_vec, 
								 const float *mul_vec, 
								 int in_func, int h_func){
	BPNeuralLayer *layer;
	BPNeuralLayer *prev_layer;
	BPNeuron *neuronPtr;

	int i = 0;

	layer = layers[i++];
	//swprintf(layer->layer_name, L"input layer");

	//iterate over input layers neurons and set in, add, mul functions
	for(int j = 0; j < layer->get_neuron_count(); j++){
		neuronPtr = layer->neurons[j];
		
		//set & add input function with default value
		neuronPtr->function_type = in_func;
		neuronPtr->add_input();

		//set add and multiply (weight) functions if present
		if(add_vec){
			neuronPtr->inputs[0]->in_add = add_vec[j];
		}
		if(mul_vec){
			neuronPtr->inputs[0]->w = mul_vec[j];
		} else {
			neuronPtr->inputs[0]->w = 1.0f; //default weight
		}
	}

	//iterate over hidden layer neurons and set inputs and bias
	for(int j = 0; j < m_layer_count - 2; j++){
		prev_layer = layer;
		layer = layers[i++];

		//swprintf(layer->layer_name, L"hidden layer %d", j + 1);

		for(int k = 0; k < layer->get_neuron_count(); k++){
			//set hidden function and add default bias
			neuronPtr = layer->neurons[k];
			neuronPtr->function_type = h_func;
			neuronPtr->add_bias();

			for(int m = 0; m < prev_layer->get_neuron_count(); m++){
				neuronPtr->add_input(prev_layer->neurons[m]);
			}
		}
	}

	prev_layer = layer;
	layer = layers[i++];
	//swprintf(layer->layer_name, L"output layer");

	for(int j = 0; j < layer->get_neuron_count(); j++){
		neuronPtr = layer->neurons[j];
		neuronPtr->function_type = h_func;
		neuronPtr->add_bias();
		
		for(int m = 0; m < prev_layer->get_neuron_count(); m++){
			neuronPtr->add_input(prev_layer->neurons[m]);
		}
	}
}

//run back propagation
//calculates error gradient by comparing with desired vector
//uses specified learning rule and learning rate (momentum)
void BPNeuralNetwork::backpropagation_train(const float *desired_vec){
	float nval = m_nval; //learning rule
	float alpha = m_alpha; //learning rate
	float delta;
	float deltaw;
	float out_val;

	for(int i = 0; i < layers[get_layer_count()-1]->get_neuron_count(); i++){
		//calculate deltas for output layer
		//delta = out_val * (1 - out_val) * (out_val - pred_val)
		out_val = layers[get_layer_count()-1]->neurons[i]->out_val;
		layers[get_layer_count()-1]->neurons[i]->delta = out_val * (desired_vec[i] - out_val) * (1.0f - out_val);
	}

	for(int i = m_layer_count - 2; i > 0; i--){
		//calculate delta for hidden layer
		for(int j = 0; j < layers[i]->get_neuron_count(); j++){
			//delta = sum (weight * delta) for all connected to output layre
			delta = 0.0f;
			for(int k = 0; k < layers[i]->neurons[j]->get_output_link_count(); k++){
				delta += layers[i]->neurons[j]->outputs[k]->w * layers[i]->neurons[j]->outputs[k]->input_neuron->delta;
			}
			out_val = layers[i]->neurons[j]->out_val;
			//backpropagated error term = out * (1 - out) * delta
			layers[i]->neurons[j]->delta = out_val * delta * (1 - out_val);
		}
	}

	//set weights for all layers
	for(int i = 1; i < get_layer_count(); i++){
		for(int j = 0; j < layers[i]->get_neuron_count(); j++){
			for(int k = 0; k < layers[i]->neurons[j]->get_input_link_count(); k++){
				//change: delta = learning_rule * x_in * delta + learning_rate * delta_previous
				deltaw = nval * layers[i]->neurons[j]->inputs[k]->in_val * layers[i]->neurons[j]->delta;
				deltaw += alpha * layers[i]->neurons[j]->inputs[k]->deltaw_prev;
				//set weights
				layers[i]->neurons[j]->inputs[k]->deltaw_prev = deltaw;
				layers[i]->neurons[j]->inputs[k]->w += deltaw;
			}
		}
	}
}

bool BPNeuralNetwork::train(const float *in_vec, float *out_vec, const float *desired_vec, float error){
	
	float deviation = 0.0f;
	
	classify(in_vec, out_vec);
	for(int i = 0; i < layers[get_layer_count() - 1]->get_neuron_count(); i++){
		deviation = fabs(out_vec[i] - desired_vec[i]);
		if (deviation > error){
			//we have hit error margin, time for backpropagation
			break;
		}
	}

	if(deviation > error){
		backpropagation_train(desired_vec);
		return true;
	} else {
		//within error margin after all runs, not trained
		return false;
	}
}

void BPNeuralNetwork::classify(const float *in_vec, float *out_vec){
	
	for(int i = 0; i < layers[0]->get_neuron_count(); i++){
		layers[0]->neurons[i]->inputs[0]->in_val = in_vec[i];
		layers[0]->neurons[i]->input_fire();
	}

	for(int i = 1; i < get_layer_count(); i++){
		for(int j = 0; j < layers[i]->get_neuron_count(); j++){
			layers[i]->neurons[j]->fire();
		}
	}

	network_output(out_vec);
}

void BPNeuralNetwork::network_output(float *out_vec) const {
	
	for(int i = 0; i < layers[get_layer_count() - 1]->get_neuron_count(); i++){
		out_vec[i] = layers[get_layer_count() - 1]->neurons[i]->out_val;
	}
}

bool BPNeuralNetwork::save(const wchar_t *filename) const {
	
	size_t charSize = sizeof(char) * wcslen(filename);
	char *charName = (char *) malloc(charSize);
	wcstombs(charName, filename, charSize);
	FILE *filepointer = fopen(charName, "wt");

	if(filepointer){
		//layer count
		fwprintf(filepointer, L"%d\n", m_layer_count);
		//each layer count
		for(int i = 0; i < get_layer_count(); i++){
			fwprintf(filepointer, L"%d ", layers[i]->get_neuron_count());
		}

		fwprintf(filepointer, L"\n\n");
		//input and hidden layer type
		fwprintf(filepointer, L"%d\n%d\n\n", layers[0]->neurons[0]->function_type, layers[1]->neurons[0]->function_type);

		for(int i = 0; i < get_layer_count(); i++){
			//print weights & bias for input layer
			fwprintf(filepointer, L"%f ", layers[0]->neurons[i]->inputs[0]->in_add);
			fwprintf(filepointer, L"%f\n", layers[0]->neurons[i]->inputs[0]->w);
		}

		fwprintf(filepointer, L"\n");

		//every layer except input layer
		for(int i = 1; i < get_layer_count(); i++){
			for(int j = 0; j < layers[i]->get_neuron_count(); j++){
				for(int k = 0; k < layers[i]->neurons[j]->get_input_link_count(); k++){
					//print weights
					fwprintf(filepointer, L"%f\n", layers[i]->neurons[j]->inputs[k]->w);
				}
			}
			//fwprintf(filepointer, L"\n");
		}

		fclose(filepointer);
		return true;
	} else {
		return false;
	}
}