using HowlDev.AI.Structures.NeuralNetwork.Options;
using HowlDev.AI.Training.Saving;

namespace HowlDev.AI.Training.Genetic;

public class GeneticAlgorithmStrategy(GenerationStrategy generation, NetworkTopologyOptions topologyOptions, WeightInitializationOptions initializationOptions) {
    public NetworkSavingScheme SavingStrategy { get; set; } = new();
    public CullingStrategy CullingStrategy { get; set; } = new();
    public GenerationStrategy GenerationStrategy { get; set; } = generation;
    public NetworkTopologyOptions TopologyOptions { get; set; } = topologyOptions;
    public WeightInitializationOptions InitializationOptions { get; set; } = initializationOptions;
    public LearningSpeed LearningSpeed { get; set; } = new();
}