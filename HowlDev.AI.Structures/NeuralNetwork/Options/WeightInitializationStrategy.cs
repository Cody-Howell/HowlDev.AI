namespace HowlDev.AI.Structures.NeuralNetwork.Options;

public enum WeightInitializationStrategy {
    /// <summary>
    /// Returns the Max.
    /// </summary>
    Constant,
    /// <summary>
    /// Returns a uniform distribution across the spectrum from the listed Min to the Max.
    /// </summary>
    Uniform
}