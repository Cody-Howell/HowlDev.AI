namespace HowlDev.AI.Structures.NeuralNetwork.Options;

public sealed class WeightInitializationOptions {
    /// <summary>
    /// Set the strategy to use. Defaults to Uniform.
    /// </summary>
    public WeightInitializationStrategy Strategy { get; init; } = WeightInitializationStrategy.Uniform;
    public double Min { get; init; } = -1;
    public double Max { get; init; } = 1;
    public double InitialBias { get; init; } = 0.0;
}