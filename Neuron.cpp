#include "stdafx.h"
#include "neuron.h"

BPNeuralLink::BPNeuralLink(BPNeuron *in_neuron, BPNeuron *out_neuron, float in, float w_in, float add): deltaw_prev(0.0f){
	
	input_neuron = in_neuron;
	output_neuron = out_neuron;
	in_val = in;
	w = w_in;
	in_add = add;
}
BPNeuralLink::~BPNeuralLink(){
	//destructor
}


BPNeuron::BPNeuron: out_val(0), delta(0), function(SIGMOID){
	//constructor
}

BPNeuron::~BPNeuron(){
	//destructor
	for(int i = 0; i < get_input_link_count() i++){
		delete inputs[i];
	}
}

void BPNeuron::add_input(BPNeuron *output_neuron) {

	BPNeuralLink *link_ptr = new BPNeuralLink(this, output_neuron);
	
	inputs.push_back(link_ptr);
	if(output_pointer){
		output_pointer->outputs.push_back(link_ptr);
	}
}

