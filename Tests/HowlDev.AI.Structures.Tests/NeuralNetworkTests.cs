using HowlDev.AI.Structures.NeuralNetwork;
using HowlDev.AI.Structures.NeuralNetwork.Options;

namespace HowlDev.AI.Structures.Tests;

public class NeuralNetworkTests {
    [Test]
    public async Task Simple2LayerWorks() {
        SimpleNeuralNetwork n = new([
            new NeuronLayer([([], 1.0)]),
            new NeuronLayer([([1.0], 0.0)])
            ]);
        n.CalculateLayers();
        await Assert.That(n.OutputLayer.Values[0]).IsEqualTo(1.0);
    }

    [Test]
    public async Task Simple3LayerWorks() {
        SimpleNeuralNetwork n = new([
            new NeuronLayer([([], 1.0)]),
            new NeuronLayer([([1.0], 0.0), ([-1.0], 0.0)]),
            new NeuronLayer([([2.0, 1.0], 0.0)])
            ]);
        n.CalculateLayers();
        await Assert.That(n.OutputLayer.Values[0]).IsEqualTo(1.0);
    }

    [Test]
    public async Task Simple3LayerWorksWithInputLayer() {
        SimpleNeuralNetwork n = new([
            new NeuronLayer([([1.0], 0.0), ([-1.0], 0.0)]),
            new NeuronLayer([([2.0, 1.0], 0.0)])
            ]);
        n.CalculateLayers(inputLayer: new NeuronLayer([([], 1.0)]));
        await Assert.That(n.OutputLayer.Values[0]).IsEqualTo(1.0);
    }

    [Test]
    public async Task Simple3LayerWithReLU() {
        SimpleNeuralNetwork n = new([
            new NeuronLayer([([], 1.0)]),
            new NeuronLayer([([1.0], 0.0), ([-1.0], 0.0)]),
            new NeuronLayer([([2.0, 1.0], 0.0)])
            ]);
        n.CalculateLayers(kind: ActivationFunctionKind.ReLU);
        await Assert.That(n.OutputLayer.Values[0]).IsEqualTo(2.0);
    }
}
public class NeuralNetworkGenerationTests {
    [Test]
    [MethodDataSource(typeof(MyTestDataSources), nameof(MyTestDataSources.AdditionTestData))]
    public async Task CanGenerateAppropriateSize(NetworkCreation values) {
        NetworkTopologyOptions o1 = new() {
            InputCount = values.input,
            HiddenLayerSizes = values.hiddenLayers,
            OutputCount = values.output
        };
        WeightInitializationOptions o2 = new();
        SimpleNeuralNetwork n = new(o1, o2);
        await Assert.That(n.Layers.Length).IsEqualTo(2 + values.hiddenLayers.Length);
        await Assert.That(n.Layers[0].Neurons.Length).IsEqualTo(values.input);
        for (int i = 1; i < values.hiddenLayers.Length + 1; i++) {
            await Assert.That(n.Layers[i].Neurons.Length).IsEqualTo(values.hiddenLayers[i - 1]);
        }
        await Assert.That(n.OutputLayer.Neurons.Length).IsEqualTo(values.output);
    }

}

public static class MyTestDataSources {
    public static IEnumerable<Func<NetworkCreation>> AdditionTestData() {
        yield return () => new(2, [1], 1);
        yield return () => new(4, [2, 3], 3);
        yield return () => new(3, [10, 15, 20], 8);
    }
}

public struct NetworkCreation(int input, int[] hiddenLayers, int output) {
    public int input = input;
    public int[] hiddenLayers = hiddenLayers;
    public int output = output;
}