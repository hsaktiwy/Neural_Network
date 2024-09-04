#include <vector>
#include <algorithm>
#include <random>
#include <functional>
#include <ctime>
#include <numeric>
#include <memory>

template <typename Gene>
class GeneticAlgorithm {
private:
    using Individual = std::vector<Gene>;
    using Population = std::vector<std::unique_ptr<Individual>>;

    std::function<Individual()> generateIndividual;
    std::function<double(const Individual&)> fitnessFunction;
    std::function<Individual(const Individual&, const Individual&)> crossoverFunction;
    std::function<void(Individual&)> mutationFunction;

    int populationSize;
    double mutationRate;
    int eliteSize;

    class RandomGenerator {
    private:
        std::mt19937 rng;
        std::uniform_real_distribution<> dist;
    public:
        RandomGenerator() : rng(std::time(0)), dist(0, 1) {}
        double operator()() { return dist(rng); }
    };

    thread_local static RandomGenerator randomGenerator;

public:
    GeneticAlgorithm(
        std::function<Individual()> genIndividual,
        std::function<double(const Individual&)> fitFunc,
        std::function<Individual(const Individual&, const Individual&)> crossFunc,
        std::function<void(Individual&)> mutFunc,
        int popSize = 100,
        double mutRate = 0.01,
        int eliteCount = 2
    ) : generateIndividual(std::move(genIndividual)),
        fitnessFunction(std::move(fitFunc)),
        crossoverFunction(std::move(crossFunc)),
        mutationFunction(std::move(mutFunc)),
        populationSize(popSize),
        mutationRate(mutRate),
        eliteSize(eliteCount)
    {}

    Individual evolve(int generations) {
        Population population = initializePopulation();
        std::vector<double> fitnesses(populationSize);

        for (int gen = 0; gen < generations; ++gen) {
            updateFitnesses(population, fitnesses);
            Population newPopulation = selectElites(population, fitnesses);

            while (newPopulation.size() < populationSize) {
                auto& parent1 = *selectParent(population, fitnesses);
                auto& parent2 = *selectParent(population, fitnesses);
                auto child = std::make_unique<Individual>(crossoverFunction(*parent1, *parent2));
                if (randomGenerator() < mutationRate) {
                    mutationFunction(*child);
                }
                newPopulation.push_back(std::move(child));
            }

            population = std::move(newPopulation);
        }

        return getBestIndividual(population, fitnesses);
    }

private:
    Population initializePopulation() {
        Population population;
        population.reserve(populationSize);// dropping the population vector time consimung in alocating and realocating
        for (int i = 0; i < populationSize; ++i) {
            population.push_back(std::make_unique<Individual>(generateIndividual()));
        }
        return population;
    }

    void updateFitnesses(const Population& population, std::vector<double>& fitnesses) {
        for (size_t i = 0; i < population.size(); ++i) {
            fitnesses[i] = fitnessFunction(*population[i]);
        }
    }

    Population selectElites(const Population& population, const std::vector<double>& fitnesses) {
        Population elites;
        elites.reserve(populationSize);

        std::vector<size_t> indices(population.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + eliteSize, indices.end(),
            [&fitnesses](size_t a, size_t b) { return fitnesses[a] > fitnesses[b]; });

        for (int i = 0; i < eliteSize; ++i) {
            elites.push_back(std::make_unique<Individual>(*population[indices[i]]));
        }
        return elites;
    }

    const Individual* selectParent(const Population& population, const std::vector<double>& fitnesses) {
        double totalFitness = std::accumulate(fitnesses.begin(), fitnesses.end(), 0.0);
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

    Individual getBestIndividual(const Population& population, const std::vector<double>& fitnesses) {
        auto it = std::max_element(fitnesses.begin(), fitnesses.end());
        return *population[std::distance(fitnesses.begin(), it)];
    }
};

template <typename Gene>
thread_local typename GeneticAlgorithm<Gene>::RandomGenerator GeneticAlgorithm<Gene>::randomGenerator;

/**
 * ToDo: specie management
*/

