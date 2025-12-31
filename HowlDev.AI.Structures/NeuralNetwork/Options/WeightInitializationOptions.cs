namespace HowlDev.AI.Structures.NeuralNetwork.Options;

public sealed class WeightInitializationOptions {
    public WeightInitializationStrategy Strategy { get; init; }
    public double Min { get; init; } = -1;
    public double Max { get; init; } = 1;
    public double InitialBias { get; init; } = 0.0;
}