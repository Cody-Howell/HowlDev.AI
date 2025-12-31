using HowlDev.AI.Structures.NeuralNetwork.Options;

namespace HowlDev.AI.Structures.NeuralNetwork;

public class SimpleNeuralNetwork {
    private NeuronLayer[] layers;

    public NeuronLayer[] Layers => [.. layers];
    public NeuronLayer OutputLayer => layers[layers.Length -1];

    public SimpleNeuralNetwork(NeuronLayer[] _layers) {
        layers = _layers;
    }

    /// <summary>
    /// Calculates all the layers in the network. If provided with an input layer, it will 
    /// calculate based off that instead of the first layer of this network. 
    /// </summary>
    /// <param name="inputLayer"></param>
    public void CalculateLayers(NeuronLayer? inputLayer = null, ActivationFunctionKind kind = ActivationFunctionKind.Identity) {
        var func = NeuronActivations.GetFunction(kind);
        if (inputLayer is not null) {
            layers[0].CalculateLayer(inputLayer, func);
        }
        int index = 1;
        while (index < layers.Length) {
            layers[index].CalculateLayer(layers[index - 1], func);
            index++;
        }
    }
}