namespace HowlDev.AI.Structures.NeuralNetwork.Options;

public sealed class NetworkTopologyOptions {
    public int InputCount { get; init; }
    public int[] HiddenLayerSizes { get; init; } = [];
    public int OutputCount { get; init; }

    public bool FullyConnected { get; init; } = true;
}