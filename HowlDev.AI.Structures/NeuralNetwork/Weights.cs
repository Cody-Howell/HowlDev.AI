using HowlDev.AI.Structures.NeuralNetwork.Options;

namespace HowlDev.AI.Structures.NeuralNetwork;

public static class WeightInitialization {
    public static Func<double> GetWeightGenerationTechnique(WeightInitializationOptions strategy) {
        return strategy.Strategy switch {
            WeightInitializationStrategy.Constant => () => strategy.Max,
            WeightInitializationStrategy.Uniform => () => Random.Shared.NextDouble() * (strategy.Max - strategy.Min) + strategy.Min,
            _ => throw new Exception($"Unknown WeightInitializationStrategy {strategy.Strategy}")
        };
    }
}