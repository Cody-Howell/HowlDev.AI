using HowlDev.AI.Structures.NeuralNetwork;
using HowlDev.AI.Structures.NeuralNetwork.Options;

namespace HowlDev.AI.Structures.Tests;

public class NeuronLayerTests {
    [Test]
    public async Task NeuronLayerInitializesWeightsOfInputNeurons1() {
        NeuronLayer layer = new([([], 1.0)]);
        await Assert.That(layer.Values[0]).IsEqualTo(1.0);
        await Assert.That(layer.Values.Length).IsEqualTo(1);
    }

    [Test]
    public async Task NeuronLayerInitializesWeightsOfInputNeurons2() {
        NeuronLayer layer = new([([], 1.0), ([], 1.0)]);
        await Assert.That(layer.Values[0]).IsEqualTo(1.0);
        await Assert.That(layer.Values[1]).IsEqualTo(1.0);
        await Assert.That(layer.Values.Length).IsEqualTo(2);
    }

    [Test]
    public async Task MismatchedNeuronsInLayerThrowsError1() {
        await Assert.That(() => new NeuronLayer([([], 1.0), ([1.0], -1.0)]))
            .Throws<ArgumentException>()
            .WithMessage("Not all neurons are the same type (input/other).");
    }

    [Test]
    public async Task MismatchedNeuronsInLayerThrowsError2() {
        await Assert.That(() => new NeuronLayer([([1.0], 1.0), ([], -1.0)]))
            .Throws<ArgumentException>()
            .WithMessage("Not all neurons are the same type (input/other).");
    }
}
public class MultilayerNeuronLayerTests {
    [Test]
    public async Task ThreeLayerTests() {
        NeuronLayer layer1 = new([([], 1.0), ([], -1.0), ([], 3.0)]);
        NeuronLayer layer2 = new([([1.0, 1.0, 1.0], 1.0), ([1.0, 1.0, 1.0], 2.0)]);
        NeuronLayer layer3 = new([([1.0, 1.0], 1.0)]);
        layer2.CalculateLayer(layer1, NeuronActivations.GetFunction(ActivationFunctionKind.Identity));
        layer3.CalculateLayer(layer2, NeuronActivations.GetFunction(ActivationFunctionKind.Identity));
        await Assert.That(layer3.Values[0]).IsEqualTo(10.0);
    }

    [Test]
    public async Task ThreeLayersOutOfOrderThrowsError() {
        NeuronLayer layer1 = new([([], 1.0), ([], -1.0), ([], 3.0)]);
        NeuronLayer layer2 = new([([1.0, 1.0, 1.0], 1.0), ([1.0, 1.0, 1.0], 2.0)]);
        NeuronLayer layer3 = new([([1.0, 1.0], 1.0)]);
        await Assert.That(() => layer3.CalculateLayer(layer2, NeuronActivations.GetFunction(ActivationFunctionKind.Identity)))
            .Throws<InvalidOperationException>()
            .WithMessage("Layer has not yet been calculated/was cleared.");
    }
}