#include <deque>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <functional>
#include <ctime>
#include <numeric>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <array>
#define endl '\n'
using namespace std;

unordered_map<string, _Gen> PGensRegister;
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

enum NType {
    Sensor,
    Output,
    Hidden
};

enum GenType {
    Weight,
    bias,
    HiddenNode,
    Input,
    Output
};

class Neuron;
class NLink {
public:
    Neuron& from;// Source neuron index
    Neuron& to;// Recipient neuron index
    double weight;// Weight of the link
    int innovation; // Innovation number
    bool active;
    NLink(Neuron& f, Neuron& t, double w) : from(f), to(t), weight(w), active(true)  {}
    void  disable()
    {
        active = false;
    }
};

typedef struct t_Gen
{
    string gen_id;
    GenType gen_type;
    double value; // Used for weights and biases

    bool active = true; // Activation status of the gene

    int indexFrom; // Index of the 'from' neuron within its layer
    int layerFrom; // Layer of the 'from' neuron
    int indexTo;   // Index of the 'to' neuron within its layer
    int layerTo;   // Layer of the 'to' neuron
    Neuron *fromNeuron; // Use smart pointers for better memory management
    Neuron *toNeuron;// Use smart pointers for better memory management
    // Constructor for weight and bias genes
    t_Gen(const string& geneId, GenType geneType, double val,  int iFrom, int lFrom, int iTo, int lTo)
        : gen_id(geneId), gen_type(geneType), value(val), indexFrom(iFrom),layerFrom(lFrom),   indexTo(iTo), layerTo(lTo) {}

    // Constructor for node genes
    t_Gen(const string& geneId, GenType geneType, int _value, int index, int layer)
        : gen_id(geneId), gen_type(geneType), value(_value), indexFrom(index), layerFrom(layer) {}
    // assigne operator
    t_Gen &operator=(const t_Gen& copy)
    {
        if (this != &copy)
        {
            gen_id = copy.gen_id;
            gen_type = copy.gen_type;
            value = copy.value;
            active = copy.active;
            indexFrom = copy.indexFrom;
            layerFrom = copy.layerFrom;
            indexTo = copy.indexTo;
            layerTo = copy.layerTo;
            fromNeuron = NULL;
            toNeuron = NULL;
        }
        return *this;
    }

    bool operator==(const t_Gen& copy)
    {
        return (this->gen_id == copy.gen_id);
    }
    // set From and To
    void setNeurons(Neuron *From, Neuron *To = NULL)
    {
        fromNeuron = From;
        toNeuron = From;
    }
} _Gen;

class Neuron {
public:
    
    NType type;
    double bias;
    double (*activation_function)(double);
    double state;
    int index;
    int layer;
    string gen_id;
    double activation; // this is the value of the input that was tested 
    deque<NLink*> in;// Incoming connections
    deque<NLink*> out;// Outgoing connections
    bool active;// this can determine if it in the equation or not (deleted)

    Neuron(NType t, double b, double (*act)(double), int id, int l, string &gen) : type(t), bias(b), activation_function(act), index(id), layer(l), gen_id(gen), active(true) {}
    /* @brief: activate neuron
    *   this function will calculate the input value passed to the neuron
    *   after activating the network
    */
    void activate(void) {
        state = bias; // Start with the bias
        for (const auto& link : in) {
            state += link->from.activation * link->weight;
        }
        // cerr << "Calculated State : " << state << endl;
        activation = activation_function(state); // Apply activation function
    }

};

class Network {
public:
    static constexpr size_t MAX_LAYER = 1000;
    array<int, MAX_LAYER> layers{};
    array<deque<Neuron*>, MAX_LAYER> layers_Neurons{};
    unordered_map<string, _Gen> gens;
    deque<Neuron> neurons;
    deque<NLink> links;
    int inputSize;
    int outputSize;
    deque<Neuron*> inputNeuron; // Store indices instead of references
    deque<Neuron*> outputNeuron;
    int score = 0;

    Network(int inSize, int outSize) : inputSize(inSize), outputSize(outSize) {
        createInitialNodes();
        connectInitialNodes();
    }

    Network(const vector<_Gen> &Gens)
    {
        for (auto& gen: Gens)
        {
            if (gen.gen_type != Weight)
            {
                neurons.emplace_back(static_cast<NType>(gen.gen_type), gen.value, sigmoid, gen.indexFrom , gen.layerFrom);
                if (gen.gen_type == Input)
                {
                    inputNeuron.push_back(&neurons.back());
                    layers_Neurons[gen.layerFrom].push_back(&neurons.back());
                    inputSize++;
                }
                if (gen.gen_type == Output)
                {
                    outputNeuron.push_back(&neurons.back());
                    layers_Neurons[gen.layerFrom].push_back(&neurons.back());
                    outputSize++;
                }
                gens[gen.gen_id] = gen;
                layers[gen.layerFrom]++;
            }
        }
        for (auto& gen: Gens)
        {
            if (gen.gen_type == Weight)
            {
                links.emplace_back(*(gen.fromNeuron), *(gen.toNeuron), gen.value);
                Neuron *tmp = gen.toNeuron;
                auto it1= find_if(neurons.begin(), neurons.end(), [tmp](Neuron &a){ return (tmp->index == a.index && tmp->layer == a.layer);});
                if (it1 != neurons.end())
                    it1->in.push_back(&links.back());
                tmp = gen.fromNeuron;
                auto it2 = find_if(neurons.begin(), neurons.end(), [tmp](Neuron &a){ return (tmp->index == a.index && tmp->layer == a.layer);});
                if (it2 != neurons.end())
                    it2->out.push_back(&links.back());
                gens[gen.gen_id] = gen;
            }
        }
    }

    deque<double> activate(deque<double> &input)
    {
      // check if the network dimension inputs match the input provided as paramiter
      if (input.size() != inputNeuron.size())
        throw runtime_error("Error: input size doesn't match inputNeuron size in Network::activate");
      // let activate the input first
      for (int i = 0; i < inputNeuron.size(); i++)
      {
          inputNeuron[i]->activation = input[i];
      }
      // activate the neuron that are not inputs
      for (auto& neu:neurons)
      {
        if (neu.type != Sensor)
        {
            // cou t << "whaaaaaaaaaaaa\n";
            neu.activate();
        }
      }
      // collect the ouput data
      deque<double> outputData;
      for(const auto& neuron : outputNeuron)
      {
          outputData.push_back(neuron->activation);
      }
      return outputData;
    }

    void addGen(_Gen &instance)
    {
        gens[instance.gen_id] = instance;
    }

    void displayNetworkInfo() {
        cout << "Network Information:" << endl;
        cout << "Total Neurons: " << neurons.size() << endl;
        cout << "Input Neurons: " << inputSize << ", Output Neurons: " << outputSize << endl;
        cout << "deque Input Neurons: " << inputNeuron.size() << ", deque Output Neurons: " << outputNeuron.size() << endl;
        cout << "Input Neuron:\n";
        for (auto& in : inputNeuron)
        {
          cout << "activation: "<< in->activation << " bias:" << in->bias << " state:" << in->state << endl;
        }
        cout << "Ouput Neuron:\n";
        for (auto& out : outputNeuron)
        {
          cout << "activation: "<< out->activation << " bias:" << out->bias << " state:" << out->state << endl;
        }
        cout << "Connections:" << endl;
        for (const auto& link : links) {
            cout << "From Neuron (Type " << link.from.type << ", Bias " << link.from.bias << ") ";
            cout << "to Neuron (Type " << link.to.type << ", Bias " << link.to.bias << ") ";
            cout << "with Weight: " << link.weight << endl;
        }
    }

    void displayNetworkInfoForAnalysis(const string& filename) {
        ofstream outFile(filename, ios::app);
        if (!outFile) {
            throw runtime_error("Error opening file for writing.");
        }

        stringstream ss;
        ss << "Network\n";
        ss << "Neuron Type,Bias,Activation,State\n";
        for (const auto& neuron : neurons) {
            ss << (neuron.type == Sensor ? "Input" : (neuron.type == Output ? "Output" : "Hidden")) << ","
            << neuron.bias << "," << neuron.activation << "," << neuron.state << "\n";
        }

        ss << "\nConnections\n";
        ss << "From Neuron Type,From Neuron Bias,To Neuron Type,To Neuron Bias,Weight\n";
        for (const auto& link : links) {
            ss << (link.from.type == Sensor ? "Input" : (link.from.type == Output ? "Output" : "Hidden")) << ","
            << link.from.bias << ","
            << (link.to.type == Sensor ? "Input" : (link.to.type == Output ? "Output" : "Hidden")) << ","
            << link.to.bias << "," << link.weight << "\n";
        }

        outFile << ss.str();
        cout << "Network information has been written to " << filename << endl;
    }

    void RegistrateGen(string &gen_id, GenType type, int layer, Neuron *from_neuron, Neuron *to_neuron, double value, int index)
    {
        if (PGensRegister.count(gen_id) == 0)
        {
            _Gen gen = {gen_id, type, value,  index, layer};
            gens[gen_id] = gen;
            gens[gen_id].setNeurons(from_neuron, to_neuron);
            PGensRegister[gen_id] = gen;
        }
        else
        {
            gens[gen_id] =  PGensRegister[gen_id];
            gens[gen_id].value = value;
        }
    }

    void GenIdGenerator(string &gen, GenType type, int arg1, int arg2=0, int arg3=0, int arg4=0) // N1 : layer or from_neurone 
    {
        // input I_index
        // Ouput O_index
        if (type == Input ||  type == Output)
        {
            gen += (type == Input) ? "I_":"O_";
            gen += to_string(arg1);
        }
        // Link L_From-index-Layer/To-index-Layer
        if (type == Weight)
        {
            gen += "L_";
            gen += to_string(arg1);gen+="-";
            gen += to_string(arg2);
        }
        // Hinden H_index-Layer
        if (type == HiddenNode)
        {
            gen += "H_";
            gen += to_string(arg1);gen+="-";
            gen += to_string(arg2);
            gen += "->";
            gen += to_string(arg3);gen+="-";
            gen += to_string(arg4);
        }
    }

    void addHiddenNode(int layer, double bias, double (*act)(double))
    {
        string gen_id;
        GenIdGenerator(gen_id, HiddenNode, layers[layer], layer);
        neurons.emplace_back(Hidden, bias, act, layers[layer], layer, gen_id);
        RegistrateGen(gen_id, HiddenNode, layer, NULL, NULL, bias, layers[layer]);
        layers[layer]++;
    }

    void DeleteHiddenNode(int layer, int index)
    {
        auto it = find_if(neurons.begin(), neurons.end(), [layer, index](Neuron &a){ return (index == a.index && layer == a.layer);});
        if (it != neurons.end())
        {
            it->active = false;

        }
    }

    void addLink();
    void DeleteLink();
private:
    void createInitialNodes() {
        mt19937 en(rand());
        uniform_real_distribution<> dis(-0.1, 0.1);
        for (int i = 0; i < inputSize; i++) {
            string gen_id;
            GenIdGenerator(gen_id, Input, i);
            neurons.emplace_back(Sensor, dis(en) * 0.2 - 0.1, sigmoid, i ,0, gen_id);
            inputNeuron.push_back(&neurons.back());
            layers_Neurons[0].push_back(&neurons.back());
            layers[0]++;

            RegistrateGen(gen_id, Input, 0, NULL, NULL, neurons.back().bias, neurons.back().index);
        }
        for (int i = 0; i < outputSize; i++) {
            string gen_id;
            GenIdGenerator(gen_id, Output,i);
            neurons.emplace_back(Output, dis(en) * 0.2 - 0.1, sigmoid, i , 1, gen_id);
            layers_Neurons[1].push_back(&neurons.back());
            outputNeuron.push_back(&neurons.back());
            layers[1]++;
            RegistrateGen(gen_id, Input, 1, NULL, NULL, neurons.back().bias, neurons.back().index);
        }
    }



    void connectInitialNodes() {
        mt19937 en(rand());
        uniform_real_distribution<> dis(-0.5, 0.5);

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                if (dis(en) > 0.0)
                {
                    float weight = static_cast<float>(dis(en)) / RAND_MAX * sqrt(2.0 / inputSize) * inputSize;
                    links.emplace_back(neurons[i], neurons[inputSize + j], weight);
                    neurons[inputSize + j].in.push_back(&links.back());
                    neurons[i].out.push_back(&links.back());
                    string gen_id;
                    GenIdGenerator(gen_id, Weight, neurons[i].index, neurons[i].layer, neurons[inputSize + j].index, neurons[inputSize + j].layer);
                    RegistrateGen(gen_id, Weight, 0, &neurons[i], &neurons[inputSize + j], weight, -1);
                }
            }
        }
    }

};

Network buildNetwork(int inputSize, int outputSize)
{
    if (inputSize== 0 || outputSize == 0)
    {
        throw "buildNetwork : inputSize || outputSize == 0\n";
    }
    return (Network {inputSize, outputSize});
}

tuple<unordered_map<string, _Gen>, unordered_map<string, _Gen>, unordered_map<string, _Gen>> gen_recognizer(const Network& dominantParent, const Network& weakParent)
{
    unordered_map<string, _Gen> matchedGen, unmatchedDominant, unmatchedWeak;
    for (const auto& [gene_id, gene] : dominantParent.gens) {
        auto it = weakParent.gens.find(gene_id);
        if (it != weakParent.gens.end()) {
            matchedGen[gene_id] = gene;
        } else {
            unmatchedDominant[gene_id] = gene;
        }
    }

    for (const auto& [gene_id, gene] : weakParent.gens) {
        if (dominantParent.gens.find(gene_id) == dominantParent.gens.end()) {
            unmatchedWeak[gene_id] = gene;
        }
    }
    return {matchedGen, unmatchedDominant, unmatchedWeak};
}

