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
    public async Task Layer2WithShapeGeneration1() {
        SimpleNeuralNetwork n = new([
            new NeuronLayer([([], 1.0)]),
            new NeuronLayer([([1.0], 0.0), ([-1.0], 0.0)]),
            new NeuronLayer([([2.0, 1.0], 0.0)])
            ]);
        n.CalculateLayers(kind: ActivationFunctionKind.ReLU);
        await Assert.That(n.OutputLayer.Values[0]).IsEqualTo(2.0);
    }
}