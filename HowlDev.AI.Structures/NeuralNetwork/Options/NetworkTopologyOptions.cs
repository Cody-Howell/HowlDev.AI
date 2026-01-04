namespace HowlDev.AI.Structures.NeuralNetwork.Options;

/// <summary>
/// Adjust Input sizes, hidden layer sizes, and output sizes.
/// </summary>
public sealed class NetworkTopologyOptions (int inputCount, int[] internalSizes, int outputCount) {
    /// <summary>
    /// Size of the input layer.
    /// </summary>
    public int InputCount { get; init; } = inputCount;
    /// <summary>
    /// Store the input layer inside the network. Defaults to True. <br/>
    /// Otherwise, pass it in every time to the <see cref="SimpleNeuralNetwork.CalculateLayers"/> function. 
    /// </summary>
    public bool CreateInputLayer { get; init; } = true;
    /// <summary>
    /// An array of all the internal hidden layer sizes.
    /// </summary>
    public int[] HiddenLayerSizes { get; init; } = internalSizes;
    /// <summary>
    /// Size of the output layer.
    /// </summary>
    public int OutputCount { get; init; } = outputCount;
}