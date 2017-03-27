

#ifndef ANNetwork_h
#define ANNetwork_h



class ANeuron;

////////////////////////////////////ANN layer///////////////////////////////////////////////////
class ANNLayer
{        
        friend class ANNetwork;
public:
        ANNLayer(int neurons_number);    
        ~ANNLayer();
        
        inline int get_neurons_number() const;
        inline ANeuron *get_neuron(int i) const;

private:
        wchar_t layer_name[_MAX_PATH];          //layer name
        int m_neurons_number;                   //number of neurons in layer
        vector<ANeuron *> neurons;              //array of Neuron classes
        
};

inline int ANNLayer::get_neurons_number() const
{
        return m_neurons_number;
}

inline ANeuron *ANNLayer::get_neuron(int i) const
{
        if (i > get_neurons_number() - 1 || i < 0)
                return 0;
        return neurons[i];
}
////////////////////////////////////////////////////////////////////////////////////////////////





///////////////////////////////////ANN network//////////////////////////////////////////////////
class ANNetwork
{
public:
        ANNetwork(int layers_number, int *neurons_per_layer);                   //number of layers, and neurons per layer
        ANNetwork(const wchar_t *fname);                                        //custom config from file
        ~ANNetwork();

        //init links for network [default SIGMOID activation function]
        void init_links(const float *avec = 0, const float *mvec = 0, int ifunc = 0, int hfunc = 1);               
        void randomize_weights(unsigned int rseed = 0);
                        
        bool train(const float *ivec, float *ovec, 
                            const float *dsrdvec, float error = 0.05);          //train network with input vector and desired output vector
        void classify(const float *ivec, float *ovec);                          //run network with in vector [ivec],  return in [ovec] network out        
        bool save(const wchar_t *fname) const;                          

        inline int status() const;        
        inline int get_layers_number() const;         
        inline ANNLayer *get_layer(int i) const;        


private:        
        int m_status;                   //status -1 err, 0-OK, 1-random weights
        int m_layers_number;            //number of layers in network
        vector<ANNLayer *> layers;      //array of layers

        float m_nrule;                  //learning rule  0.2
        float m_alpha;                  //momentum  0.7
        
        void backprop_run(const float *dsrdvec);
        void network_output(float *ovec) const;                               //get network out
};

inline int ANNetwork::status() const
{
        return m_status;
}

inline int ANNetwork::get_layers_number() const
{
        return m_layers_number;
} 

inline ANNLayer *ANNetwork::get_layer(int i) const
{
        if (i > get_layers_number() - 1 || i < 0)
                return 0;
        return layers[i];
}
////////////////////////////////////////////////////////////////////////////////////////////////


#endif ANNetwork_h

/*
     ANNetwork

     1 2 3 4           layers,  ANNLayer
    ------------
    |N|N|N|N| | ....   neurons N,  ANeuron
    ------------
    |N|N|N| | | ....
    ------------
    |N| | | | |
    ...
    ...
    ...

                    */







/*
  file format
  3
  40 10 1
  
  0
  1                activation function 0-linear, 1-sigmoid
  
  [input norm]

  weights





 */

