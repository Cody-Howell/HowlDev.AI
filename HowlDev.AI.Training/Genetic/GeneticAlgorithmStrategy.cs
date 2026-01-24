using HowlDev.AI.Core;
using HowlDev.AI.Structures.NeuralNetwork.Options;
using HowlDev.AI.Training.Saving;

namespace HowlDev.AI.Training.Genetic;

/// <summary>
/// Set the strategy for the entire algorithm. Some parameters are required (Generation, Topology, 
/// and Weight Initialization). 
/// </summary>
public class GeneticAlgorithmStrategy(GenerationStrategy generation, NetworkTopologyOptions topologyOptions, WeightInitializationOptions initializationOptions) {
    /// <summary>
    /// Determines which networks are sent to the <see cref="IResultReader"/>
    /// </summary>
    public NetworkSavingScheme SavingStrategy { get; set; } = new();
    /// <summary>
    /// Sets which networks are culled. 
    /// </summary>
    public CullingStrategy CullingStrategy { get; set; } = new();
    /// <summary>
    /// Sets the parameters for the algorithm to run. 
    /// </summary>
    public GenerationStrategy GenerationStrategy { get; set; } = generation;
    /// <summary>
    /// Sets the topology of the network.
    /// </summary>
    public NetworkTopologyOptions TopologyOptions { get; set; } = topologyOptions;
    /// <summary>
    /// Set the weight initialization for new networks. 
    /// </summary>
    public WeightInitializationOptions InitializationOptions { get; set; } = initializationOptions;
    /// <summary>
    /// Sets the learning speed for the algorithm, based on generation.
    /// </summary>
    public LearningSpeed LearningSpeed { get; set; } = new();
    /// <summary>
    /// Set the function activation for the neurons. Defaults to <see cref="ActivationFunctionKind.Identity"/>.
    /// </summary>
    public ActivationFunctionKind FunctionKind { get; set; } = ActivationFunctionKind.Identity;
}