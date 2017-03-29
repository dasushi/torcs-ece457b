#ifndef BPNeuralNetwork_h
#define BPNeuralNetwork_h

class BPNeuron;

class BPNeuralLayer{
	
	friend class BPNeuralNetwork;

public:
	BPNeuralLayer(int neuron_count);
	~BPNeuralLayer();
	
	inline BPNeuron * get_neuron(int index) const;
	inline int get_neuron_count() const;

private:
	int m_neuron_count;
	wchar_t layer_name[260]; //_MAX_PATH = 260
	std::vector<BPNeuron *> neurons;

};

inline BPNeuron *BPNeuralLayer::get_neuron(int index) const{

	if((index < 0) | ((index > get_neuron_count()) - 1)){
		return 0;
	}

	return neurons[index];
}

inline int BPNeuralLayer::get_neuron_count() const{
	return m_neuron_count;
}

class BPNeuralNetwork{

public:
	BPNeuralNetwork(int layer_count, int *neuron_count_per_layer);
	BPNeuralNetwork(const wchar_t *filename);
	~BPNeuralNetwork();

	void randomize_weights(unsigned int rand_seed = 0);
	void init_links(const float *acc_vec = 0, const float *mul_vec = 0,
		int in_func = 0, int h_func = 1);
	void classify(const float *in_vec, float *out_vec);

	bool train(const float *in_vec, float *out_vec, const float *desired_vec,
		float error = 0.05);
	bool save(const wchar_t *filename) const; 

	inline int flag() const;
	inline int get_layer_count() const;
	inline BPNeuralLayer *get_layer(int index) const;

private:
	int m_flag;
	int m_layer_count;

	std::vector<BPNeuralLayer *> layers;
	float m_nval;
	float m_alpha;

	void backpropagation_train(const float *desired_vec);
	void network_output(float *out_vec) const;
};

inline int BPNeuralNetwork::get_layer_count() const{
	return m_layer_count;
}

inline BPNeuralLayer *BPNeuralNetwork::get_layer(int index) const{
	if ((index < 0) | (index > get_layer_count() - 1)){
		return 0;
	}
	return layers[index];
}

inline int BPNeuralNetwork::flag() const{
	return m_flag;
}
#endif