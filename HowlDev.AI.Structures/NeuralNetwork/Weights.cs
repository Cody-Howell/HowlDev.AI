using HowlDev.AI.Structures.NeuralNetwork.Options;

namespace HowlDev.AI.Structures.NeuralNetwork;

/// <summary>
/// Sets weight initialization for new networks. 
/// </summary>
public static class WeightInitialization {
    /// <summary>
    /// Get a function that creates a weight based on the strategy given. 
    /// </summary>
    /// <param name="strategy"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    public static Func<double> GetWeightGenerationTechnique(WeightInitializationOptions strategy) {
        return strategy.Strategy switch {
            WeightInitializationStrategy.Constant => () => strategy.Max,
            WeightInitializationStrategy.Uniform => () => Random.Shared.NextDouble() * (strategy.Max - strategy.Min) + strategy.Min,
            _ => throw new Exception($"Unknown WeightInitializationStrategy {strategy.Strategy}")
        };
    }
}