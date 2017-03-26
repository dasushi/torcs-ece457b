#ifndef BPNeuron_h
#define BPNeuron_h

class BPNeuron;

class BPNeuralLink{
	friend class BPNeuron;
	friend class BPNeuralNetwork;
public:
	BPNeuralLink(BPNeuron *in_neuron, BPNeuron *out_neuron = 0, float in = 1.0f, float w = 0.0f, float add = 0.0f);
	~BPNeuralLink();

	inline void set_add_val(float add);
	inline void set_weight(float w);
private:
	BPNeuron *input_neuron;
	BPNeuron *output_neuron;

	float in_add;
	float deltaw_prev;
	float w;
	float in_val;
};

inline void BPNeuralLink::set_add_val(float add){
	in_add = add;
}

inline void BPNeuralLink::set_weight(float new_w){
	w = new_w;
}


class BPNeuron{
	friend class BPNeuralNetwork;
	BPNeuron();
	~BPNeuron();

	void add_bias();
	void add_input(BPNeuron *output_neuron = 0);
	void fire();
	void input_fire();

	inline int get_input_link_count() const;
	inline int get_output_link_count() const;
	inline void set_function_type(enum FUNCTION_TYPE function);
	inline BPNeuralLink *get_input_link(int index) const;
	inline BPNeuralLink *get_output_link(int index) const;
private:
	int function;
	float delta;
	float out_val;

	vector<BPNeuralLink *> outputs;
	vector<BPNeuralLink *> inputs;
};

inline int BPNeuron::get_input_link_count() const{
	return intputs.size();
}

inline int BPNeuron::get_output_link_count() const{
	return outputs.size();
}

inline void BPNeuron::set_function_type(enum FUNCTION_TYPE function_type){
	function = function_type;
}

inline BPNeuralLink *BPNeuron::get_input_link(int index) const {
	if (i < 0 || i > get_input_link_count() - 1){
		return 0;
	}
	
	return inputs[i];
}

inline BPNeuralLink *BPNeuron::get_output_link(int index) const {
	if(i < 0 || i > get_output_link_count() - 1 ){
		return 0;
	}

	return outputs[i];
}


inline void BPNeuron::input_fire(){
	out_val = inputs[0]->w * (inputs[0]->in_val + inputs[0]->in_add);

	if(function == SIGMOID){
		out_val = 1.0f / (1.0f + exp(float(out_val * (-1.0f)));
	}

	for(int i = 0; i < get_output_link_count(); i++){
		outputs[i]->in_val = out_val;
	}
}

#endif