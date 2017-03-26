#ifndef BPNeuron_h
#define BPNeuron_h

class BPNeuron;


////BPNeuralLink////
//connection between in and out neurons
class BPNeuralLink{
	friend class BPNeuron;
	friend class BPNeuralNetwork;
public:
	BPNeuralLink(BPNeuron *in_neuron, BPNeuron *out_neuron = 0, float in = 1.0f, float w = 0.0f, float add = 0.0f);
	~BPNeuralLink();
	
	//set the add term with a float value
	inline void set_add_val(float add);
	//set the weight with a float value
	inline void set_weight(float w);
private:
	//input neuron connected to N inputs
	BPNeuron *input_neuron;
	//output neuron connected to N outputs
	BPNeuron *output_neuron;

	//input neuron add value
	float in_add;
	//previous iteration weight change value
	float deltaw_prev;
	//weight value
	float w;
	//input value for input_neuron
	float in_val;
};

inline void BPNeuralLink::set_add_val(float add){
	in_add = add;
}

inline void BPNeuralLink::set_weight(float new_w){
	w = new_w;
}

////BPNeuron////
//A single BackPropagation Neuron with Sigmoid and Linear activation functions

class BPNeuron{
	friend class BPNeuralNetwork;
public:
	//option for linear and sigmoid activation functions
	enum FUNCTION_TYPE {LINEAR, SIGMOID};

	BPNeuron();
	~BPNeuron();

	void add_bias();
	//create an input with the designated output
	void add_input(BPNeuron *output_neuron = 0);
	//for output/hidden: iterate over inputs, 
	//  calculate out_val, send to output neuron
	void fire();
	//for input layer: take input, normalize, send to hidden neuron
	void input_fire();

	//accessors
	inline int get_input_link_count() const;
	inline int get_output_link_count() const;
	inline void set_function_type(enum FUNCTION_TYPE function);
	inline BPNeuralLink *get_input_link(int index) const;
	inline BPNeuralLink *get_output_link(int index) const;
private:
	//enum for FUNCTION_TYPE
	int function;
	//delta change
	float delta;
	float out_val;

	//output links, each with connections to their input neurons
	vector<BPNeuralLink *> outputs;
	//input and bias links 
	vector<BPNeuralLink *> inputs;
};


inline int BPNeuron::get_input_link_count() const{
	return inputs.size();
}

inline int BPNeuron::get_output_link_count() const{
	return outputs.size();
}

inline void BPNeuron::set_function_type(enum FUNCTION_TYPE function_type){
	function = function_type;
}

inline BPNeuralLink *BPNeuron::get_input_link(int index) const {
	if (index < 0 || index > get_input_link_count() - 1){
		return 0;
	}
	
	return inputs[index];
}

inline BPNeuralLink *BPNeuron::get_output_link(int index) const {
	//check index is within bounds
	if(index < 0 || index > get_output_link_count() - 1 ){
		return 0;
	}

	return outputs[index];
}


inline void BPNeuron::input_fire(){
	//apply input normalization
	out_val = inputs[0]->w * (inputs[0]->in_val + inputs[0]->in_add);

	//apply sigmoid function, if linear nothing required
	if(function == SIGMOID){
		out_val = 1.0f / (1.0f + exp(float(out_val * (-1.0f)));
	}

	//connect outputs to other input links
	for(int i = 0; i < get_output_link_count(); i++){
		outputs[i]->in_val = out_val;
	}
}

inline void BPNeuron::fire(){

	out_val = 0.0f;
	//sum the output for this neuron
	for(int i = 0; i < get_input_link_count(); i++){
		out_val += inputs[i]->w * inputs[i]->in_val;
	}
	//apply sigmoid function, if linear nothing required
	if(function == SIGMOID){
		out_val = 1.0f / (exp(float((-1.0f) * out_val)) + 1.0f);
	}

	//connect outputs to other input links
	for(int i = 0; i < get_output_link_count(); i++){
		outputs[i]->in_val = out_val;
	}
}

#endif