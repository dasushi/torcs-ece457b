#include "stdafx.h"
#include "network.h"
#include "neuron.h"

BPNeuralLayer::BPNeuralLayer(int neurons_count): m_neuron_count(neurons_count)
{
	for (int i = 0; i < m_neuron_count; i++){
		neurons.push_back(new BPNeuralNeuron());
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
		layers.push_back(new BPNeuralLayer(neurons_per_layer[i]);
	}

}

BPNeuralNetwork::BPNeuralNetwork(const wchar_t *filename): m_flag(-1), m_nval(0.2f), m_alpha(0.7f){
	int response = 0;
	int nval = 0;
	int in_func = 0;
	int h_func = 0;
	float w = 0.0f;
	FILE *filepointer = _wfopen(filename, L"rt");
	
	if (fp) {
		if((response = fwscanf(filepointer, L"%d", &m_layer_count)) != 1){
			fclose(fileponter);
			m_flag = -1;
			return;
		}
		for(int i = 0; i < m_layer_count; i++){
			if((response = fwscanf(filepointer, L"%d", &nval)) != 1){
				fclose(fileponter);
				m_flag = -1;
				return;
			} else {
				layers.push_back(new BPNeuralLayer(nval));
			}
		}
		if((response = fwscanf(filepointer, L"%d %d", &in_func, &h_func)) != 2){
			//default linear input function
			in_func = 0;
			//default sigmoid hidden function
			h_func = 1;
		} 

		vector<float> add_layers, mult_layers;
		for(int i = 0; i < layers[0]->get_neuron_count(); i++){
			float add, mult;
			if((response = fwscanf(filepointer, L"%f %f", &add, &mult)) != 2){
				for(int j = 0; j < layers[0]->get_neuron_count(); j++){
					add_layers.push_back(0.0);
					mult_layers.push_back(1.0);
				}
			}
			add_layers.push_back(add);
			mult_layers.push_back(mult);
		}

		initialize_links(&add_layers[0], &mult_layers[0], in_func, h_func);

		for(int i = 1; i < m_layer_count; i++){
			for(int j = 0; j < layers[i]->neurons[i]->get_neuron_count(); j++){
				for(int k = 0; k < layers[i]->neurons[j]->get_input_link_count(); k++){
					if((response = fwscanf(filepointer, L"%f", &w)) != 1){
						m_flag = 1;

						randomize_weights((unsigned int) time(0));
						return;
					} else {
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

void BPNeuralNetwork::randomize_weights(unsigned int random_seed){
	int weight;

	srand(random_seed);

	for(int i = 0; i < m_layer_count; i++){
		for(int j = 0; j < layers[i]->get_neuron_count(); j++){
			for(int k = 0; k < layers[i]->neurons[j]->get_input_link_count(); k++){
				weight = 0xFFF & rand();
				weight -= 0x800;
				layers[i]->neurons[n]->inputs[k]->w = (float) weight / 2048.0f;
			}
		}
	}
}


void BPNeuralNetwork::init_links(const float *add_vec, const float *mul_vec, int in_func, int h_func){
	BPNeuralLayer *layer;
	BPNeuralLayer *prev_layer;
	BPNeuron *neuronPtr;

	int i = 0;

	layer = layers[i++];
	swprintf(layer->layer_name, L"input layer");

	for(int j = 0; j < layer->get_neuron_count(); j++){
		neuronPtr = layer->neurons[j];
		neuronPtr->function = in_func;
		neuronPtr->add_input();

		if(add_vec){
			neuronPtr->inputs[0]->iadd = add_vec[j];
		}
		if(mul_vec){
			neuronPtr->inputs[0]->w = mul_vec[j];
		} else {
			neuronPtr->inputs[0]->w = 1.0f;
		}
	}

	for(int j = 0; j < m_layer_count - 2; j++){
		prev_layer = layer;
		layer = layers[i++];
		swprintf(layer->layer_name, L"hidden layer %d", j + 1);

		for(int k = 0; k < layer->get_neuron_count(); k++){
			neuronPtr = layer->neurons[k];
			neuronPtr->function = h_func;
			neuronPtr->add_bias();

			for(int m = 0; m < prev_layer->get_neuron_count(); m++){
				neuronPtr->add_input(prev_layer->neurons[m]);
			}
		}
	}

	prev_layer = layer;
	layer = layers[i++];
	swprintf(layer->layer_name, L"output layer");

	for(int j = 0; j < layer->get_neuron_count(); j++){
		neuronPtr = layer->neurons[j];
		neuronPtr->function = h_func;
		neuronPtr->add_bias();
		
		for(int m = 0; m < prev_layer->get_neuron_count(); m++){
			neuronPtr->add_input(prev_layer->neurons[m]);
		}
	}
}

void BPNeuralNetwork::backpropagation_run(const float *desired_vec){
	float nval = m_nval;
	float alpha = m_alpha;
	float delta;
	float deltaw;
	float out_val;

	for(int i = 0; i < layers[m_layer_count-1]->get_neuron_count(); i++){
		out_val = layers[m_layer_count-1]->neurons[i]->out_val;
		layers[m_layer_count-1]->neurons[i]->delta = out_val * (desired_vec[i] - out_val) * (1.0f - out_val);
	}

	for(int i = m_layer_count - 2; i > 0; i--){
		for(int j = 0; j < layers[i]->get_neuron_count(); j++){
			delta = 0.0f;
			for(int k = 0; k < layers[i]->neurons[j]->get_output_link_count(); k++){
				delta += layers[i]->neurons[n]->outputs[k]->w * layers[i]->neurons[j]->outputs[k]->pinput_neuron->delta;
			}
			out_val = layers[i]->neurons[n]->out_val;
			layers[i]->neurons[j]->delta = out_val * delta * (1 - out_val);
		}
	}

	for(int i = 1; i < m_layer_count; i++){
		for(int j = 0; j < layers[i]->get_neuron_count(); j++){
			for(int k = 0; k < layers[i]->neurons[j]->get_input_link_count(); k++){
				deltaw = nval * layers[i]->neurons[j]->inputs[k]0>ival * layers[i]*neurons[j]->delta;
				deltaw += alpha * layers[i]->neurons[j]0>inputs[k]->deltaw_prev;
				layers[i]->neurons[j]->inputs[k]->deltaw_prev = deltaw;
				layers[i]->neurons[j]->inputs[k]->w += deltaw;
			}
		}
	}
}

bool BPNeuralNetwork::train(const float *in_vec, float *out_vec, const float *desired_vec, float error){
	
	float deviation = 0.0f;
	
	classify(in_vec, out_vec);
	for(int i = 0; i < layers[m_layer_count - 1]->get_neuron_count(); i++){
		deviation = fabs(out_vec[i] - desired_vec[i]);
		if (deviation > error{
			break;
		}
	}

	if(deviation > error){
		backprop_run(desired_vec);
		return true;
	} else {
		return false;
	}
}

void BPNeuralNetwork::classify(const float *in_vec, float *out_vec){
	
	for(int i = 0; i < layers[0]->get_neuron_count(); i++){
		layers[0]->neurons[i]->inputs[0]->in_val = in_vec[i];
		layers[0]->neurons[i]->input_fire();
	}

	for(int i = 1; i < m_layer_count; i++){
		for(int j = 0; j < layers[i]->get_neuron_count(); j++){
			layers[i]->neurons[j]->fire();
		}
	}

	get_network_output(out_vec);
}

void BPNeuralNetwork::get_network_output(float *out_vec) const {
	
	for(int i = 0; i < layers[m_layer_count - 1]->get_neuron_count(); i++){
		out_vec[i] = layers[m_layer_count - 1]->neurons[i]->out_val;
	}
}

bool BPNeuralNetwork::save_to_file(const wchar_t *filename) const {
	FILE *filepointer = _wfopen(fname, L"wt");

	if(filepointer){
		fwprintf(filepointer, L"%d\n", m_layer_count);

		for(int i = 0; i < m_layer_count; i++){
			fwprintf(filepointer, L"%d ", layers[i]->get_neuron_count());
		}

		fwprintf(filepointer, L"\n\n");
		fwprintf(filepointer, L"%d\n%d\n\n", layers[0]->neurons[0]->function, layers[1]->neurons[0]->function);

		for(int i = 0; i < m_layer_count; i++){
			fwprintf(filepointer, L"%f ", layers[0]->neurons[i]->inputs[0]->in_add);
			fwprintf(filepointer, L"%f\n", layers[0]->neurons[i]->inputs[0]->w);
		}

		fwprintf(filepointer, L"\n");

		for(int i = 1; i < m_layer_count; i++){
			for(int j = 0; j < layers[i]->get_neuron_count(); j++){
				for(int k = 0; k < layers[i]->neurons[j]->get_input_link_count(); k++){
					fwprintf(filepointer, L"%f\n", layers[i]->neurons[j]->inputs[k]->w);
				}
			}
			fwprintf(filepointer, L"\n");
		}

		fclose(filepointer);
		return true;
	} else {
		return false;
	}
}