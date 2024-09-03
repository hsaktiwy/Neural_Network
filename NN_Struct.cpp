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

// Correct sigmoid function

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
    string gene_id;
    GenType gene_type;
    int layer; // this if it a hidden Node adding
    Neuron *from_neuron;
    Neuron *to_neuron;
    double value; // for bias or weight
    int index;
} _Gen;

class Neuron {
public:
    
    int active;// this can determine if it in the equation or not (deleted)
    double bias;
    double state;
    int index;
    int layer;
    double activation; // this is the value of the input that was tested 
    double (*activation_function)(double);
    NType type;
    deque<NLink*> in;// Incoming connections
    deque<NLink*> out;// Outgoing connections

    Neuron(NType t, double b, double (*act)(double), int id, int l) : type(t), bias(b), activation_function(act), index(id), layer(l), active(true) {}
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
            if (gen.gene_type != Weight)
            {
                neurons.emplace_back(static_cast<NType>(gen.gene_type), gen.value, sigmoid, gen.index , gen.layer);
                if (gen.gene_type == Input)
                {
                    inputNeuron.push_back(&neurons.back());
                    layers_Neurons[gen.layer].push_back(&neurons.back());
                    inputSize++;
                }
                if (gen.gene_type == Output)
                {
                    outputNeuron.push_back(&neurons.back());
                    layers_Neurons[gen.layer].push_back(&neurons.back());
                    outputSize++;
                }
                gens[gen.gene_id] = gen;
                layers[gen.layer]++;
            }
        }
        for (auto& gen: Gens)
        {
            if (gen.gene_type == Weight)
            {
                links.emplace_back(*(gen.from_neuron), *(gen.to_neuron), gen.value);
                Neuron *tmp = gen.to_neuron;
                auto it1= find_if(neurons.begin(), neurons.end(), [tmp](Neuron &a){ return (tmp->index == a.index && tmp->layer == a.layer);});
                if (it1 != neurons.end())
                    it1->in.push_back(&links.back());
                tmp = gen.from_neuron;
                auto it2 = find_if(neurons.begin(), neurons.end(), [tmp](Neuron &a){ return (tmp->index == a.index && tmp->layer == a.layer);});
                if (it2 != neurons.end())
                    it2->out.push_back(&links.back());
                gens[gen.gene_id] = gen;
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
        gens[instance.gene_id] = instance;
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

private:
    void createInitialNodes() {
        mt19937 en(rand());
        uniform_real_distribution<> dis(-0.1, 0.1);
        for (int i = 0; i < inputSize; i++) {
            neurons.emplace_back(Sensor, dis(en) * 0.2 - 0.1, sigmoid, i ,0);
            inputNeuron.push_back(&neurons.back());
            layers_Neurons[0].push_back(&neurons.back());
            layers[0]++;
            string gen_id = "I_";
            gen_id += (char(i + 48));
            _Gen gen = {gen_id, Input, 0, NULL, NULL, neurons.back().bias, neurons.back().index};
            gens[gen_id] = gen; 
        }
        for (int i = 0; i < outputSize; i++) {
            neurons.emplace_back(Output, dis(en) * 0.2 - 0.1, sigmoid, i , 1);
            layers_Neurons[1].push_back(&neurons.back());
            outputNeuron.push_back(&neurons.back());
            layers[1]++;
            string gen_id = "O_";
            gen_id += (char(i + 48));
            _Gen gen = {gen_id, Input, 1, NULL, NULL, neurons.back().bias, neurons.back().index};
            gens[gen_id] = gen;
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
                    string gen_id = "L_";
                    gen_id += (char(i + 48))+ '_' + (char(j + 48)) + "_1";
                    _Gen gen = {gen_id, Weight, 0, &neurons[i], &neurons[inputSize + j], weight, -1};
                    gens[gen_id] = gen;
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

// ##################################################################
// ##################################################################
// ##################################################################
// ##################################################################


template <typename Gene>
class GeneticAlgorithm {
private:
    using Individual = vector<Gene>;
    using Population = vector<unique_ptr<Individual>>;

    function<Individual()> generateIndividual;
    function<double(const Individual&)> fitnessFunction;
    function<Individual(const Individual&, const Individual&)> crossoverFunction;
    function<void(Individual&)> mutationFunction;

    int populationSize;
    double mutationRate;
    int eliteSize;

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
    GeneticAlgorithm(
        function<Individual()> genIndividual,
        function<double(const Individual&)> fitFunc,
        function<Individual(const Individual&, const Individual&)> crossFunc,
        function<void(Individual&)> mutFunc,
        int popSize = 100,
        double mutRate = 0.01,
        int eliteCount = 2
    ) : generateIndividual(move(genIndividual)),
        fitnessFunction(move(fitFunc)),
        crossoverFunction(move(crossFunc)),
        mutationFunction(move(mutFunc)),
        populationSize(popSize),
        mutationRate(mutRate),
        eliteSize(eliteCount)
    {}

    Individual evolve(int generations) {
        Population population = initializePopulation();
        vector<double> fitnesses(populationSize);

        for (int gen = 0; gen < generations; ++gen) {
            updateFitnesses(population, fitnesses);
            Population newPopulation = selectElites(population, fitnesses);

            while (newPopulation.size() < populationSize) {
                auto& parent1 = *selectParent(population, fitnesses);
                auto& parent2 = *selectParent(population, fitnesses);
                auto child = make_unique<Individual>(crossoverFunction(*parent1, *parent2));
                if (randomGenerator() < mutationRate) {
                    mutationFunction(*child);
                }
                newPopulation.push_back(move(child));
            }

            population = move(newPopulation);
        }

        return getBestIndividual(population, fitnesses);
    }

private:
    Population initializePopulation() {
        Population population;
        population.reserve(populationSize);// dropping the population vector time consimung in alocating and realocating
        for (int i = 0; i < populationSize; ++i) {
            population.push_back(make_unique<Individual>(generateIndividual()));
        }
        return population;
    }

    void updateFitnesses(const Population& population, vector<double>& fitnesses) {
        for (size_t i = 0; i < population.size(); ++i) {
            fitnesses[i] = fitnessFunction(*population[i]);
        }
    }

    Population selectElites(const Population& population, const vector<double>& fitnesses) {
        Population elites;
        elites.reserve(populationSize);

        vector<size_t> indices(population.size());
        iota(indices.begin(), indices.end(), 0);
        partial_sort(indices.begin(), indices.begin() + eliteSize, indices.end(),
            [&fitnesses](size_t a, size_t b) { return fitnesses[a] > fitnesses[b]; });

        for (int i = 0; i < eliteSize; ++i) {
            elites.push_back(make_unique<Individual>(*population[indices[i]]));
        }
        return elites;
    }

    const Individual* selectParent(const Population& population, const vector<double>& fitnesses) {
        double totalFitness = accumulate(fitnesses.begin(), fitnesses.end(), 0.0);
        double pick = randomGenerator() * totalFitness;
        double current = 0;
        for (size_t i = 0; i < population.size(); ++i) {
            current += fitnesses[i];
            if (current > pick) {
                return population[i].get();
            }
        }
        return population.back().get();
    }

    Individual getBestIndividual(const Population& population, const vector<double>& fitnesses) {
        auto it = max_element(fitnesses.begin(), fitnesses.end());
        return *population[distance(fitnesses.begin(), it)];
    }
};

template <typename Gene>
thread_local typename GeneticAlgorithm<Gene>::RandomGenerator GeneticAlgorithm<Gene>::randomGenerator;

// ##################################################################
// ##################################################################
// ##################################################################
// ##################################################################

// function<Individual()> genIndividual,
Network CreatIndividual(void)
{
    return (Network (3, 2));
}
// function<Individual(const Individual&, const Individual&)> crossFunc,
Network crossOver(const Network& p1, const Network& p2)
{
    const Network &DominantParent = (p1.score > p2.score) ? p1 : p2;
    const Network &WeakParent = (p1.score <= p2.score) ? p1 : p2;
    auto [matchedGen, unmatchedDominant, unmatchedWeak] = gen_recognizer(DominantParent, WeakParent);
    vector<_Gen> gens;
    gens.reserve(matchedGen.size() + unmatchedDominant.size() + unmatchedWeak.size());
    // Add all matched genes
    for (const auto& [_, gene] : matchedGen) {
        gens.push_back(gene);
    }
    vector<_Gen> all_unmatched;
    all_unmatched.reserve(unmatchedDominant.size() + unmatchedWeak.size());
    for (const auto& [_, gene] : unmatchedDominant) all_unmatched.push_back(gene);
    for (const auto& [_, gene] : unmatchedWeak) all_unmatched.push_back(gene);
    mt19937 en(rand());
    uniform_real_distribution<> dis(0, 1);
    shuffle(all_unmatched.begin(), all_unmatched.end(), en);
    for (const auto& gene : all_unmatched) {
        if (dis(en) > 0.5)
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
            member.neurons.emplace_back(Hidden, dis(en) * 2 - 1, sigmoid, member.layers[newLayerIndex], newLayerIndex);
            member.layers[newLayerIndex]++;
            // add the gens
        }
    mutate = dis(en) < 0.01;
    // extract node
        // define the node that we wont to delete plus we need to delete all connection related to it
        if (mutate)
        {

        }
    // add link
    mutate = dis(en) < 0.02;
        // add link between 2 existing nodes
    // extract link
        // extract link between 2 existing nodes
    // weight modification (we will apply simulation anealing),
    // i think about using a temperature?
}
// function<double(const Individual&)> fitFunc,

// ##################################################################
// ##################################################################
// ##################################################################
// ################################################################## 

int main() {
    srand(time(NULL));
    deque<Network> population;
    for (int i = 0; i < 25; i++)
    {
        Network net(3, 2);
        net.displayNetworkInfo();
        deque<double> inputs = {(random()%100) * 1.0, (random()%4000) * 1.0, (random()%5000) * 1.0};
        deque<double> outputs = net.activate(inputs);
        net.displayNetworkInfoForAnalysis("analyze.txt");
        for (double output : outputs) {
            cout << "Output: " << output << endl;
        }
        population.push_back(net);
    }
    return 0;
}