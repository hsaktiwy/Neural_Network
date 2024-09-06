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
#include <unordered_set>
#include <array>
#define endl '\n'
using namespace std;


double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

string specie_Name(double e, double d, double c)
{
    char s[49];
    union {
        double d;
        uint64_t u;
    } converter;
    converter.d = e;
    sprintf(s, "%016llx", converter.u);
    converter.d = d;
    sprintf(&s[16], "%016llx", converter.u);
    converter.d = c;
    sprintf(&s[32], "%016llx", converter.u);
    s[48] = '\0';
    return s;
}

long simpleHash(int a, int b, int sequenceSize = 256)
{
    if (a == b)
        return -1;
    long mid = (a + b);
    long result = ((mid)*(mid + (abs(a - b) + a * b))) / 2 + ((a > b) ? a + sequenceSize: b);
    return result;
}

enum NType {
    SENSOR,
    OUTPUT,
    HIDDEN
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
    Neuron* from;// Source neuron index
    Neuron* to;// Recipient neuron index
    double weight;// Weight of the link
    string innovation; // Innovation number
    bool active;
    NLink(Neuron* f, Neuron* t, double w, string &ino) : from(f), to(t), weight(w), innovation(ino), active(true) {}
    NLink(const NLink& copy)
    {
        *this = copy;
    }

    NLink& operator=(const NLink& copy)
    {
        if (this != &copy)
        {
            from = copy.from;
            to = copy.to;
            weight = copy.weight;
            innovation = copy.innovation;
            active = copy.active;
        }
        return *this;
    }
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
    // Default constructor
    t_Gen() = default;
    t_Gen(const string& geneId, GenType geneType, double val,  int iFrom, int lFrom, int iTo, int lTo)
        : gen_id(geneId), gen_type(geneType), value(val), indexFrom(iFrom),layerFrom(lFrom),   indexTo(iTo), layerTo(lTo) {}

    // Constructor for node genes
    t_Gen(const string& geneId, GenType geneType, double _value, int index, int layer)
        : gen_id(geneId), gen_type(geneType), value(_value), indexFrom(index), layerFrom(layer) {}
    //copy constructor
    t_Gen(const t_Gen& copy)
    {
        *this = copy;
    }
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
unordered_map<string, _Gen> PGensRegister;
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
    void disable()
    {
        active = false;
    }

    bool operator<(const Neuron& op) const
    {
        if (layer != op.layer) {
            return layer < op.layer;
        }
        return index < op.index;
    }

    void activate(void) {
        if (in.size() > 0)
            state = bias; // Start with the bias
        else
            state = 0;
        for (const auto& link : in) {
            if (link->active)
                state += link->from->activation * link->weight;
        }
        // cerr << "Calculated State : " << state << endl;
        activation = activation_function(state); // Apply activation function
    }

};

class Network {
public:
    static int indexCounter;
    static constexpr size_t MAX_LAYER = 1000;
    array<int, MAX_LAYER> layers{};
    array<deque<Neuron*>, MAX_LAYER> layers_Neurons{};
    unordered_map<string, _Gen> gens;
    deque<Neuron> neurons;
    deque<NLink> links;
    int inputSize;
    int outputSize;
    int hiddenSize;
    deque<Neuron*> inputNeuron; // Store indices instead of references
    deque<Neuron*> outputNeuron;
    deque<Neuron*> hiddenNeuron;
    double adjustedFitness;
    double score = 0;
    vector<pair<string, _Gen>> SortedGens;
    int index;

    Network(int inSize, int outSize) : inputSize(inSize), outputSize(outSize), hiddenSize(0) {
        createInitialNodes();
        connectInitialNodes();
        index = indexCounter;
        indexCounter++;
    }

    Network(const Network& copy)
    {
        *this = copy;
    }

    Network& operator=(const Network& copy)
    {
        if (this != &copy)
        {
            inputSize = copy.inputSize;
            outputSize = copy.outputSize;
            hiddenSize = copy.hiddenSize;
            links = copy.links;

            neurons = copy.neurons;
            for(auto& neu: neurons)
            {
                if (neu.type == SENSOR)
                {
                    inputNeuron.push_back(&neu);
                }
                if (neu.type == OUTPUT)
                {
                    outputNeuron.push_back(&neu);
                }
                if (neu.type == HIDDEN)
                {
                    hiddenNeuron.push_back(&neu);
                }
            }
            adjustedFitness = copy.adjustedFitness;
            score = copy.score;
            SortedGens = copy.SortedGens;
            index = copy.index;
        }
        return *this;
    }

    static void indexCounterReset( void )
    {
        indexCounter = 0;
    }

    Network(vector<_Gen> &Gens)
    {
        for (auto& gen: Gens)
        {
            if (gen.gen_type != Weight)
            {
                NType NeuronType = (gen.gen_type == Output) ? OUTPUT:((gen.gen_type == Input) ? SENSOR: HIDDEN);
                neurons.emplace_back(NeuronType, gen.value, sigmoid, gen.indexFrom , gen.layerFrom, gen.gen_id);
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
                Neuron *from = gen.fromNeuron, *to = gen.toNeuron;
                if (from != NULL && to != NULL)
                {
                    auto it_from= find_if(neurons.begin(), neurons.end(), [from](Neuron &a){ return (from->index == a.index && from->layer == a.layer);});
                    auto it_to = find_if(neurons.begin(), neurons.end(), [to](Neuron &a){ return (to->index == a.index && to->layer == a.layer);});
                    if (it_from != neurons.end() && it_to != neurons.end())
                    {
                        links.emplace_back(&(*it_from), &(*it_to), gen.value, gen.gen_id);
                        it_to->in.push_back(&links.back());
                        it_from->out.push_back(&links.back());
                        gens[gen.gen_id] = gen;
                    }
                }
            }
        }
    }


    deque<double> activate(deque<double> &input)
    {
        // sort neurons before they will be surely sorted by there layer so no need to there index cause it want gave diffrence in the network to pology
        sort(neurons.begin(), neurons.end());
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
            if (neu.type != SENSOR)
            {
                if (neu.active)
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

    void display() {
        cout << "Network Information:" << endl;
        cout << "Total Neurons: " << neurons.size() << endl;
        cout << "Input Neurons: " << inputSize << ", Output Neurons: " << outputSize << ", Hidden Neurons: " << hiddenSize << endl;
        cout << "deque Input Neurons: " << inputNeuron.size() << ", deque Output Neurons: " << outputNeuron.size() << ", deque Hidden Neurons: " <<  neurons.size() - outputNeuron.size() - inputNeuron.size()<< endl;
        cout << "Input Neuron:\n";
        for (auto& in : inputNeuron)
        {
          cout << "activation: "<< in->activation << " bias:" << in->bias << " state:" << in->state << in->state << " active:"  << in->active << " layer: " << in->layer << " index:" << in->index << endl;
        }
        cout << "Ouput Neuron:\n";
        for (auto& out : outputNeuron)
        {
          cout << "activation: "<< out->activation << " bias:" << out->bias << " state:" << out->state << out->state << " active:"  << out->active << " layer: " << out->layer << " index:" << out->index << endl;
        }
        cout << "Hidden Neurons\n";
        for(auto& hidden: hiddenNeuron)
        {
            if (hidden->type == HIDDEN)
                cout << "activation: "<< hidden->activation << " bias:" << hidden->bias << " state:" << hidden->state << " active:"  << hidden->active << " layer: " << hidden->layer << " index:" << hidden->index << endl;
        }
        cout << "Connections:" << endl;
        for (const auto& link : links) {
            cout << "From Neuron (Type " << link.from->type << ", Bias " << link.from->bias << ") ";
            cout << "to Neuron (Type " << link.to->type << ", Bias " << link.to->bias << ") ";
            cout << "with Weight: " << link.weight << endl;
        }
    }

    void StreamAnalysis(const string& filename) {
        ofstream outFile(filename, ios::app);
        if (!outFile) {
            throw runtime_error("Error opening file for writing.");
        }

        stringstream ss;
        ss << "Network\n";
        ss << "Neuron Type,Bias,Activation,State\n";
        for (const auto& neuron : neurons) {
            ss << (neuron.type == SENSOR ? "Input" : (neuron.type == OUTPUT ? "OUTPUT" : "Hidden")) << ","
            << neuron.bias << "," << neuron.activation << "," << neuron.state << "\n";
        }

        ss << "\nConnections\n";
        ss << "From Neuron Type,From Neuron Bias,To Neuron Type,To Neuron Bias,Weight\n";
        for (const auto& link : links) {
            ss << (link.from->type == SENSOR ? "Input" : (link.from->type == OUTPUT ? "OUTPUT" : "Hidden")) << ","
            << link.from->bias << ","
            << (link.to->type == SENSOR ? "Input" : (link.to->type == OUTPUT ? "OUTPUT" : "Hidden")) << ","
            << link.to->bias << "," << link.weight << "\n";
        }

        outFile << ss.str();
        cout << "Network information has been written to " << filename << endl;
    }

    void RegistrateGen(string &gen_id, GenType type, double value, int layerFrom, int indexFrom,int layerTo=-1, int indexTo=-1,Neuron *from_neuron=NULL, Neuron *to_neuron=NULL)
    {
        if (PGensRegister.count(gen_id) == 0)
        {
            if (type != Weight)
            {
                _Gen gen = {gen_id, type, value,  indexFrom, layerFrom};
                gens[gen_id] = gen;
                gens[gen_id].setNeurons(from_neuron, to_neuron);
                PGensRegister[gen_id] = gen;
            }
            else
            {
                _Gen gen = {gen_id, type, value,  indexFrom, layerFrom, layerTo, indexTo};
                gens[gen_id] = gen;
                gens[gen_id].setNeurons(from_neuron, to_neuron);
                PGensRegister[gen_id] = gen;
            }
        }
        else
        {
            gens[gen_id] = PGensRegister[gen_id];
            gens[gen_id].setNeurons(from_neuron, to_neuron);
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
        neurons.emplace_back(HIDDEN, bias, act, layers[layer], layer, gen_id);
        RegistrateGen(gen_id, HiddenNode, bias, layer, layers[layer]);
        layers[layer]++;
        hiddenNeuron.push_back(&neurons.back());
        hiddenSize++;
    }

    void DeleteHiddenNode(int layer, int index)
    {
        auto it = find_if(neurons.begin(), neurons.end(), [layer, index](Neuron &a){ return (index == a.index && layer == a.layer);});
        if (it != neurons.end())
        {
            it->disable();
            for (auto in : it->in)
            {
                if (in)
                    in->disable();
            }
            for (auto in : it->in)
            {
                if (in)
                    in->disable();
            }
        }
        else
        {
            cout << "Hidden Node not found\n";
        }
    }

    bool isValidConnection(Neuron **f, Neuron **t)
    {
        if (((*f)->type == SENSOR && (*t)->type == SENSOR) || ((*f)->type == OUTPUT && (*t)->type == OUTPUT) || ((*f)->type == HIDDEN && (*t)->type == HIDDEN && (*f)->layer == (*t)->layer))
            return false;
        if (((*f)->type == HIDDEN && (*t)->type == SENSOR)
            || ((*f)->type == OUTPUT && (*t)->type == HIDDEN)
            || ((*f)->type == OUTPUT && (*t)->type == SENSOR)
            || ((*f)->type == HIDDEN && (*t)->type == HIDDEN && (*t)->layer < (*f)->layer))
        {
            Neuron *tmp;
            tmp = *f;
            (*f) = (*t);
            (*t) = tmp;
        }
        return true;
    }

    void addLink( void )
    {
        string gen_id;
        Neuron *n1;
        Neuron *n2;
        // this will check if that we did already create all possible links
        size_t PossibleLinksNumber = size_t((neurons.size() * ((neurons.size() - 1) / 2.0)));
        if (links.size() >= PossibleLinksNumber)
            return ;
        while (gens.count(gen_id) != 0 || gen_id == "")
        {
            gen_id.clear();
            int r1 = rand()% neurons.size();
            int r2 = rand()% neurons.size();

            n1 = &neurons[r1];
            n2 = &neurons[r2];
            if (isValidConnection(&n1, &n2)) {
                GenIdGenerator(gen_id, Weight, n1->index, n1->layer, n2->index, n2->layer);
            }
        }
        mt19937 en(rand());
        uniform_real_distribution<> dis(-0.1, 0.1);
        float weight = static_cast<float>(dis(en)) / RAND_MAX * sqrt(2.0 / inputSize) * inputSize;
        RegistrateGen(gen_id, Weight, weight, n1->layer, n1->index, n2->layer, n2->index, n1, n2);
        links.emplace_back(n1, n2, weight, gen_id);
        n2->in.push_back(&links.back());
        n1->out.push_back(&links.back());
    }

    void DeleteLink( void )
    {
        int r = static_cast<int>(rand() % links.size());

        auto it = links.begin() + r;
        if (it != links.end())
            it->disable();
    }

    void InitialSortedGens()
    {
        SortedGens.assign(gens.begin(),gens.end());
    }
private:
    void createInitialNodes() {
        mt19937 en(rand());
        uniform_real_distribution<> dis(-0.1, 0.1);
        for (int i = 0; i < inputSize; i++) {
            string gen_id;
            GenIdGenerator(gen_id, Input, i);
            neurons.emplace_back(SENSOR, dis(en) * 0.2 - 0.1, sigmoid, i ,0, gen_id);
            inputNeuron.push_back(&neurons.back());
            layers_Neurons[0].push_back(&neurons.back());
            layers[0]++;

            RegistrateGen(gen_id, Input, neurons.back().bias, 0, neurons.back().index);
        }
        for (int i = 0; i < outputSize; i++) {
            string gen_id;
            GenIdGenerator(gen_id, Output,i);
            neurons.emplace_back(OUTPUT, dis(en) * 0.2 - 0.1, sigmoid, i , 1, gen_id);
            layers_Neurons[1].push_back(&neurons.back());
            outputNeuron.push_back(&neurons.back());
            layers[1]++;
            RegistrateGen(gen_id, Output, neurons.back().bias, 1, neurons.back().index);
        }
    }



    void connectInitialNodes() {
        mt19937 en(rand());
        uniform_real_distribution<> dis(-0.5, 0.5);

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                // if (dis(en) > 0.0)
                // {
                    float weight = static_cast<float>(dis(en)) / RAND_MAX * sqrt(2.0 / inputSize) * inputSize;
                    string gen_id;
                    GenIdGenerator(gen_id, Weight, neurons[i].index, neurons[i].layer, neurons[inputSize + j].index, neurons[inputSize + j].layer);
                    links.emplace_back(&neurons[i], &neurons[inputSize + j], weight, gen_id);
                    neurons[inputSize + j].in.push_back(&links.back());
                    neurons[i].out.push_back(&links.back());
                    RegistrateGen(gen_id, Weight, weight, neurons[i].layer, neurons[i].index, neurons[inputSize + j].layer, neurons[inputSize + j].index, &neurons[i], &neurons[inputSize + j]);
                // }
            }
        }
    }

};

int Network::indexCounter = 0;
Network buildNetwork(int inputSize, int outputSize)
{
    if (inputSize== 0 || outputSize == 0)
    {
        throw "buildNetwork : inputSize || outputSize == 0\n";
    }
    return (Network {inputSize, outputSize});
}
/**
 * This function was create with the idea that we will need to keep unmatchedWeak parent list of gens
 * 
  tuple<unordered_map<string, _Gen>, unordered_map<string, _Gen>, unordered_map<string, _Gen>> gen_recognizer(const Network& dominantParent, const Network& weakParent)
  {
      unordered_map<string, _Gen> matchedGen, unmatchedDominant, unmatchedWeak;
      for (const auto& [gene_id, gene] : dominantParent.gens) {
          auto it = weakParent.gens.find(gene_id);
          if (it != weakParent.gens.end()) {
              (rand()%2 == 0) ? matchedGen[gene_id] = gene : matchedGen[gene_id] = it->second;
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
*/
tuple<unordered_map<string, _Gen>, unordered_map<string, _Gen> > gen_recognizer(const Network& dominantParent, const Network& weakParent)

{
    unordered_map<string, _Gen> matchedGen, unmatched;
    for (const auto& [gene_id, gene] : dominantParent.gens) {
        auto it = weakParent.gens.find(gene_id);
        if (it != weakParent.gens.end()) {
            (rand()%2 == 0) ? matchedGen[gene_id] = gene : matchedGen[gene_id] = it->second;
        } else {
            unmatched[gene_id] = gene;
        }
    }
    return {matchedGen, unmatched};
}

class SpeciesPerformanceManager {
    public:
        struct SpeciesData {
            vector<double> fitnessHistory;
            int currentSize;
            int bestFitness;
            int generationLastImproved;

            int currentEliteSize;
            bool isStagnant;
        };

        unordered_map<string, SpeciesData> speciesMap;
        int currentGeneration = 0;
        int maxGenerationsWithoutImprovement = 10;

        int getBestFitness(const string& speciesId) {
            return speciesMap[speciesId].bestFitness;
        }

        void updateSpecies(const string& speciesId, int size, int bestFitness) {
            auto& data = speciesMap[speciesId];
            data.fitnessHistory.push_back(bestFitness);
            data.currentSize = size;
            data.bestFitness = max(data.bestFitness, bestFitness);
            if (bestFitness > data.bestFitness) {
                data.generationLastImproved = currentGeneration;
            }
        }

        void evaluateSpecies() {
            for (auto& [speciesId, data] : speciesMap) {
                if (currentGeneration - data.generationLastImproved > maxGenerationsWithoutImprovement) {
                    data.isStagnant = true;
                } else {
                    data.isStagnant = false;
                }
            }
        }

        void adjustParameters() {
            for (auto& [speciesId, data] : speciesMap) {
                // Adjust elite size and mating rate based on performance
                if (data.currentSize > 10) {
                    cout << "Increasing elite size for species " << speciesId << endl;
                } else {
                    cout << "Decreasing elite size for species " << speciesId << endl;
                }
            }
        }

        void nextGeneration() {
            currentGeneration++;
        }

        void CalculateEliteSize(const string& speciesId) {
            auto& data = speciesMap[speciesId];
            data.currentEliteSize = (data.currentSize / 5);
        }
};


template <typename Gene>
class GeneticAlgorithm {
private:
    using Individual = Gene;
    using Population = vector<Individual>;
    function<Individual()> generateIndividual;
    function<double(Individual&)> fitnessFunction;
    function<Individual(const Individual&, const Individual&)> crossoverFunction;
    function<void(Individual&)> mutationFunction;

    int populationSize;
    double mutationRate;
    int eliteSize;
    double Threshold;
    class RandomGenerator {
    private:
        mt19937 rng;
        uniform_real_distribution<> dist;
    public:
        RandomGenerator() : rng(time(0)), dist(0, 1) {}
        double operator()() { return dist(rng); }
    };

    thread_local static RandomGenerator randomGenerator;

public:
    class Specie {
        public:
            vector<Gene*> members;
            vector<Gene*> elites;

            Network representative;
            double totalAdjustedFitness;
            int total_E,total_D;
            double total_W;
            double av_E, av_D, av_W;
            string speciesName;
            Specie() = default;
            Specie(Gene &initialMember):total_E(0),total_D(0),total_W(0), representative(initialMember)
            {
                members.reserve(50);
                members.push_back(&initialMember);
                totalAdjustedFitness = 0;
            }

            void addMember(Gene &network) {
                members.push_back(&network);
            }

            void calculateAdjustedFitness() {
                for (auto& member : members) {
                    double sharedFitness = member->score / members.size();
                    totalAdjustedFitness += sharedFitness;
                    member->adjustedFitness = sharedFitness;
                }
            }

            int countExcessGenes(const Gene& other) const {
                if (representative.SortedGens.empty() || other.SortedGens.empty()) return 0;

                const string& maxInnovThis = representative.SortedGens.back().first;
                const string& maxInnovOther = other.SortedGens.back().first;

                int excessCount = 0;
                for (const auto& [_, gene] : representative.SortedGens) {
                    if (gene.gen_id > maxInnovOther) excessCount++;
                }
                for (const auto& [_, gene] : other.SortedGens) {
                    if (gene.gen_id > maxInnovThis) excessCount++;
                }

                return excessCount;
            }

            int countDisjointGenes(const Gene& other) const {
                unordered_map<string, bool> thisGenes;
                cout << "Representative Sorted genes :" << representative.SortedGens.size() << endl;

                for (const auto& [_, gene] : representative.SortedGens) {
                    thisGenes[gene.gen_id] = true;
                }
                cout << "After Representative Sorted genes :" << representative.SortedGens.size() << endl;

                cout << "1\n";
                cout << "Sorted genes :" << other.SortedGens.size() << endl;
                int disjointCount = 0;
                for (const auto& [_, gene] : other.SortedGens) {
                    if (thisGenes.find(gene.gen_id) == thisGenes.end() &&
                        gene.gen_id < representative.SortedGens.back().first) {
                        disjointCount++;
                    }
                    cout << "> " << gene.gen_id << endl;
                }
                cout << "2\n";

                for (const auto& [gen_id, _] : representative.SortedGens) {
                    const string &ss = gen_id;
                    auto lambda = [&ss](const auto& pair) { return pair.first == ss; };
                    if (find_if(other.SortedGens.begin(), other.SortedGens.end(), lambda) == other.SortedGens.end() &&
                        gen_id < other.SortedGens.back().first) {
                        disjointCount++;
                    }
                }
                cout << "? end\n";
                return disjointCount;
            }

            double averageWeightDifference(const Gene& other) const {
                unordered_map<string, double> thisWeights;
                for (const auto& [_, gene] : representative.SortedGens) {
                    thisWeights[gene.gen_id] = gene.value;
                }

                double totalDiff = 0.0;
                int matchingGenes = 0;

                for (const auto& [_, gene] : other.SortedGens) {
                    auto it = thisWeights.find(gene.gen_id);
                    if (it != thisWeights.end()) {
                        totalDiff += abs(it->second - gene.value);
                        matchingGenes++;
                    }
                }

                return matchingGenes > 0 ? totalDiff / matchingGenes : 0.0;
            }

            double calculateCompatibility(Gene &other) {
                other.InitialSortedGens();
                int E = countExcessGenes(other);
                int D = countDisjointGenes(other);
                double W = averageWeightDifference(other);
                int N = max(representative.SortedGens.size(), other.SortedGens.size());
                N = (N < 20) ? 1 : N; // Normalize for small genomes
                total_E+=E,total_D+=D,total_W+=W;
                const double c1 = 1.0, c2 = 1.0, c3 = 0.4; // Example coefficients
                return (c1 * E / N) + (c2 * D / N) + (c3 * W);
            }
            bool isCompatible(Gene &net, double threshold)
            {
                double c = calculateCompatibility(net);
                if (threshold > c)
                    return true;
                return false;
            }

            void setAvrage()
            {
                av_E = static_cast<double>(total_E/ members.size()),
                av_D = static_cast<double>(total_D/members.size()),
                av_W = static_cast<double>(total_W/members.size());
                speciesName = specie_Name(av_E, av_D, av_W);
            }
    };
    using Species = vector<Specie>;
    GeneticAlgorithm(
        function<Individual()> genIndividual,
        function<double(Individual&)> fitFunc,
        function<Individual(const Individual&, const Individual&)> crossFunc,
        function<void(Individual&)> mutFunc,
        int popSize = 100,
        double mutRate = 0.01,
        int eliteCount = 2,
        int threshold = 1
    ) : generateIndividual(genIndividual),
        fitnessFunction(fitFunc),
        crossoverFunction(crossFunc),
        mutationFunction(mutFunc),
        populationSize(popSize),
        mutationRate(mutRate),
        eliteSize(eliteCount),
        Threshold(threshold)
    {}

    // Individual evolve(int generations) {
    //     Population population = initializePopulation();
    //     vector<double> fitnesses(populationSize);
    //     for (int gen = 0; gen < generations; ++gen) {
    //         updateFitnesses(population, fitnesses);
    //         Species Species = spliteToSpecies(population);
    //         Population newPopulation = selectElites(population, fitnesses);

    //         while (newPopulation.size() < populationSize) {
    //             auto& parent1 = *selectParent(population, fitnesses);
    //             auto& parent2 = *selectParent(population, fitnesses);
    //             auto child = make_unique<Individual>(crossoverFunction(*parent1, *parent2));
    //             if (randomGenerator() < mutationRate) {
    //                 mutationFunction(*child);
    //             }
    //             newPopulation.push_back(move(child));
    //         }

    //         population = move(newPopulation);
    //     }

    //     return getBestIndividual(population, fitnesses);
    // }

    Individual evolve(int generations) {
        Population population = initializePopulation();
        vector<double> fitnesses(populationSize);
        SpeciesPerformanceManager manager;
        Network::indexCounterReset();
        unordered_set<long> mates;
        for (int gen = 0; gen < generations; ++gen) {
            updateFitnesses(population, fitnesses);
            Species species = splitToSpecies(population);

            // Adjust fitnesses based on species
            for (auto& sp : species) {
                sp.calculateAdjustedFitness();
                sp.setAvrage();
                manager.updateSpecies(sp.speciesName, sp.members.size(), sp.totalAdjustedFitness);
                manager.CalculateEliteSize(sp.speciesName);
                int eliteSize = manager.speciesMap[sp.speciesName].currentEliteSize;
                sp.elites= selectElites(sp.members, fitnesses, eliteSize);
            }

            Population newPopulation;
            while (newPopulation.size() < populationSize) {
                Individual* parent1, *parent2;
                if (randomGenerator() > 0.05)
                {
                    Specie &sp = species[rand() % species.size()];
                    parent1 = selectParent(sp.members, fitnesses);
                    parent2 = selectParent(sp.members, fitnesses);
                }
                else
                {
                    auto& sp1 = species[rand() % species.size()];
                    auto& sp2 = species[rand() % species.size()];
                    parent1 = selectParent(sp1.members, fitnesses);
                    parent2 = selectParent(sp2.members, fitnesses);
                }
                long hashed = parent1->index + parent2->index;
                if (mates.count(hashed) == 0) {
                    mates.insert(hashed);    
                    auto child = crossoverFunction(*parent1, *parent2);
                    if (randomGenerator() < mutationRate) {
                        mutationFunction(child);
                    }
                    mutateWithSimulatedAnnealing(child, 100, 0.95, gen);
                    newPopulation.push_back(child);
                }
            }
            population = move(newPopulation);
    }
    return getBestIndividual(population, fitnesses);
}

private:

    void mutateWithSimulatedAnnealing(Network& network, double initialTemp, double coolingRate, int currentGeneration) {
        double temperature = initialTemp * pow(coolingRate, currentGeneration); // Exponential cooling

        mt19937 rng(random_device{}());
        normal_distribution<> perturbation(0, 1); // Small perturbation

        for (auto& link : network.links) {
            if (uniform_real_distribution<>(0, 1)(rng) < 0.1) { // 10% chance to mutate each weight
                double oldWeight = link.weight;
                double newWeight = oldWeight + perturbation(rng) * temperature; // Perturb weight based on temperature

                // Temporarily apply new weight
                link.weight = newWeight;

                // Evaluate change in fitness, here we assume a function that evaluates the entire network
                double oldFitness = network.score; // Assuming current score is stored
                double newFitness = fitnessFunction(network);

                double deltaE = newFitness - oldFitness;

                if (deltaE < 0 || exp(-deltaE / temperature) > uniform_real_distribution<>(0, 1)(rng)) {
                    // Accept new weight
                    network.score = newFitness; // Update network fitness
                } else {
                    // Revert to old weight
                    link.weight = oldWeight;
                }
            }
        }
    }

    void speciate(Species &species, Gene &network) {
        bool foundSpecies = false;
        cout << " int -- "<<  species.size() << endl;
        int i = 0;
        for (auto& sp : species) {
            cout << "inderxer : " << i << endl;
            if (sp.isCompatible(network, Threshold)) {
                sp.addMember(network);
                foundSpecies = true;
                break;
            }
            i++;
        }
        if (!foundSpecies) {
            species.emplace_back(network);
        }
        cout << " -- OUT "<< endl;
    }

    Species splitToSpecies(Population &population)
    {
        static Species sps;
        sps.clear();
        for (auto& member: population)
        {
            member.InitialSortedGens();
            speciate(sps, member);
        }
        return sps;
    }

    Population initializePopulation() {
        Population population;
        population.reserve(populationSize);// dropping the population vector time consimung in alocating and realocating
        for (int i = 0; i < populationSize; ++i) {
            population.emplace_back(generateIndividual());
        }
        return population;
    }

    void updateFitnesses(Population& population, vector<double>& fitnesses) {
        for (size_t i = 0; i < population.size(); ++i) {
            fitnesses[i] = fitnessFunction(population[i]);
        }
    }

     vector<Individual*> selectElites(const vector<Individual*>& population, const vector<double>& fitnesses, int Size) {
        static vector<Individual*> elites;
        if (elites.size() > 0)
            elites.clear();
        if (elites.capacity() == 0)
            elites.reserve(populationSize);
        vector<size_t> indices(population.size());
        iota(indices.begin(), indices.end(), 0);
        partial_sort(indices.begin(), indices.begin() + Size, indices.end(),
            [&fitnesses](size_t a, size_t b) { return fitnesses[a] > fitnesses[b]; });

        for (int i = 0; i < Size; ++i) {
            elites.push_back(population[indices[i]]);
        }
        return elites;
    }

    Individual* selectParent(vector<Individual*>& population, const vector<double>& fitnesses) {
        double totalFitness = accumulate(fitnesses.begin(), fitnesses.end(), 0.0);
        double pick = randomGenerator() * totalFitness;
        double current = 0;
        for (size_t i = 0; i < population.size(); ++i) {
            current += fitnesses[i];
            if (current > pick) {
                return population[i];
            }
        }
        return population.back();
    }

    Individual getBestIndividual(const Population& population, const vector<double>& fitnesses) {
        auto it = max_element(fitnesses.begin(), fitnesses.end());
        return population[distance(fitnesses.begin(), it)];
    }
};

template <typename Gene>
thread_local typename GeneticAlgorithm<Gene>::RandomGenerator GeneticAlgorithm<Gene>::randomGenerator;


/************************************************************************************************************************************************************************/
/************************************************************************************************************************************************************************/

deque<double> input;
// function<Individual()> genIndividual,
Network CreatIndividual(void)
{
    Network net(2, 1);
    net.activate(input);
    return (net);
}

// function<Individual(const Individual&, const Individual&)> crossFunc,
Network crossOver(const Network& p1, const Network& p2)
{
    const Network &DominantParent = (p1.score > p2.score) ? p1 : p2;
    const Network &WeakParent = (p1.score <= p2.score) ? p1 : p2;
    auto [matchedGen, unmatched] = gen_recognizer(DominantParent, WeakParent);
    vector<_Gen> gens;
    gens.reserve(matchedGen.size() + unmatched.size());
    // Add all matched genes
    for (const auto& [_, gene] : matchedGen) {
        gens.push_back(gene);
    }
    vector<_Gen> all_unmatched;
    all_unmatched.reserve(unmatched.size());
    for (const auto& [_, gene] : unmatched) all_unmatched.push_back(gene);\
    mt19937 en(rand());
    uniform_real_distribution<> dis(0, 1);
    for (const auto& gene : all_unmatched) {
        gens.push_back(gene);
    }
    return (Network (gens));
}
// function<void(Individual&)> mutFunc,

void mutFunc(Network& member)
{
    mt19937 en(rand());
    uniform_real_distribution<> dis(0, 1);
    bool mutate = false;
    mutate = dis(en) < 0.05;
    // add node (we can call this like a expensive evolution operation,
      //so could we drop it rate more like what happen in nature ?)
        // define the possiton  where we will add the node in case we wont to build new layer or not ?
    if (mutate)
    {
        int newLayerIndex =  1 + std::uniform_int_distribution<>(0, member.layers.size() - 2)(en);
        member.addHiddenNode(newLayerIndex, dis(en), sigmoid);
        // add the gens
    }
    mutate = dis(en) < 0.01;
    // extract node
        // define the node that we wont to delete plus we need to delete all connection related to it
    if (mutate)
    {
        int layer = std::uniform_int_distribution<>(1, member.hiddenNeuron.size())(en);
        Neuron *node = member.hiddenNeuron[layer];
        member.DeleteHiddenNode(node->layer, node->index);
    }
    // add link
    mutate = dis(en) < 0.02;
        // add link between 2 existing nodes
    if (mutate)
    {
        member.addLink();
    }
    mutate = dis(en) < 0.01;
    // extract link
        // extract link between 2 existing nodes
    if (mutate)
    {
        member.DeleteLink();
    }
}

// function<double(const Individual&)> fitFunc,
double fitFunc(Network& network)
{
    cout << "Activation : input size " << input.size() << endl;
    network.activate(input);
    int result =  (input[0] == input[1]) ? 0 : 1;
    return (result - network.outputNeuron[0]->activation > 0.1);
}
/************************************************************************************************************************************************************************/
/************************************************************************************************************************************************************************/
/**
 * ToDo: specie management
*/

void aggressiveTest() {
    // Test 1: Create a network with 3 inputs and 2 outputs
    Network net(3, 2);
    assert(net.inputSize == 3);
    assert(net.outputSize == 2);
    assert(net.neurons.size() == 5); // 3 inputs + 2 outputs
    assert(net.links.size() == 6); // Each input connected to each output

    // Test 2: Activate the network with valid input
    deque<double> input = {0.5, 0.3, 0.9};
    net.display();
    auto output = net.activate(input);
    assert(output.size() == 2); // Should return 2 outputs
    cout << "Output after activation: ";
    for (const auto& val : output) {
        cout << val << " ";
    }
    cout << endl;

    // Test 3: Attempt to activate with mismatched input size
    try {
        deque<double> invalidInput = {0.5, 0.3}; // Only 2 inputs
        net.activate(invalidInput);
        assert(false); // Should not reach here
    } catch (const runtime_error& e) {
        cout << "Caught expected error: " << e.what() << endl;
    }

    // Test 4: Add a hidden node and check the count
    net.addHiddenNode(2, 0.1, sigmoid);
    assert(net.hiddenSize == 1);
    assert(net.neurons.size() == 6); // 6 total neurons now

    // Test 5: Add a link and check the connections
    size_t initialLinkCount = net.links.size();
    net.addLink();
    assert(net.links.size() == initialLinkCount + 1);

    // Test 6: Delete a hidden node
    net.DeleteHiddenNode(2, 0);
    net.display();
    assert(net.hiddenNeuron.back()->active == false);
    assert(net.neurons.size() == 6); // Neuron still exists, just disabled

    // Test 7: Check display functionality
    net.display(); // This should print the network information

    // Test 8: Stream analysis to a file
    string filename = "network_analysis.txt";
    net.StreamAnalysis(filename);
    ifstream inFile(filename);
    assert(inFile.is_open());
    cout << "Network analysis written to " << filename << endl;

    // Test 9: Add and register genes
    _Gen gene1("I_0_0", Input, 0.5, 0, 0);
    int old  = net.gens.size();
    net.addGen(gene1);

    assert(net.gens.size() == 1 + old); // Should have 1 more gene

    // Test 10: Check for valid connections
    Neuron* n1 = &net.neurons[0]; // First input neuron
    Neuron* n2 = &net.neurons[3]; // First output neuron
    assert(net.isValidConnection(&n1, &n2) == true); // Should be valid

    // Test 11: Attempt invalid connection (input to input)
    Neuron* n3 = &net.neurons[1]; // Second input neuron
    assert(net.isValidConnection(&n1, &n3) == false); // Should be invalid

    // Test 12: Check the registration of genes
    string gen_id = "G2";
    net.RegistrateGen(gen_id, Weight, 0.3, 0, 0);
    assert(net.gens.count(gen_id) == 1); // Should be registered

    // Test 13: Check the initial sorted genes
    net.InitialSortedGens();
    assert(net.SortedGens.size() == net.gens.size()); // Should match the number of genes

    cout << "All tests passed!" << endl;
}

int main() {
    // aggressiveTest();
    input = {0.5, 0.5};
    GeneticAlgorithm<Network> ga(CreatIndividual, fitFunc, crossOver, mutFunc, 100, 0.01, 2, 1);
    Network bestNetwork = ga.evolve(100);
    return 0;
}