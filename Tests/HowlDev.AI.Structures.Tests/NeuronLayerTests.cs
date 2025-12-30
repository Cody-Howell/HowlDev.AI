using HowlDev.AI.Structures.NeuralNetwork;

namespace HowlDev.AI.Structures.Tests;

public class NeuronLayerTests {
    [Test]
    public async Task NeuronLayerInitializesWeightsOfInputNeurons1() {
        NeuronLayer layer = new([new Neuron([], 1.0)]);
        await Assert.That(layer.Values[0]).IsEqualTo(1.0);
        await Assert.That(layer.Values.Length).IsEqualTo(1);
    }

    [Test]
    public async Task NeuronLayerInitializesWeightsOfInputNeurons2() {
        NeuronLayer layer = new([new Neuron([], 1.0), new Neuron([], 1.0)]);
        await Assert.That(layer.Values[0]).IsEqualTo(1.0);
        await Assert.That(layer.Values[1]).IsEqualTo(1.0);
        await Assert.That(layer.Values.Length).IsEqualTo(2);
    }

    [Test]
    public async Task MismatchedNeuronsInLayerThrowsError1() {
        await Assert.That(() => new NeuronLayer([new Neuron([], 1.0), new Neuron([1.0], -1.0)]))
            .Throws<ArgumentException>()
            .WithMessage("Not all neurons are the same type (input/other).");
    }

    [Test]
    public async Task MismatchedNeuronsInLayerThrowsError2() {
        await Assert.That(() => new NeuronLayer([new Neuron([1.0], 1.0), new Neuron([], -1.0)]))
            .Throws<ArgumentException>()
            .WithMessage("Not all neurons are the same type (input/other).");
    }
}