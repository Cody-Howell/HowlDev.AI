using HowlDev.AI.Structures.NeuralNetwork;
using HowlDev.AI.Structures.NeuralNetwork.Options;

namespace HowlDev.AI.Structures.Tests;

public class NeuronTests {
    [Test]
    public async Task InputNeuronWorks() {
        Neuron n = new([], 1.0);
        await Assert.That(n.Bias).IsEqualTo(1.0);
        await Assert.That(n.IsInputNeuron).IsEqualTo(true);
        await Assert.That(n.CalculateValue(null, NeuronActivations.GetFunction(ActivationFunctionKind.Identity))).IsEqualTo(1.0);
    }

    [Test]
    public async Task NeuronCanReadInputLayerOfInputNeurons1() {
        Neuron n = new([1.0], 0.0);
        NeuronLayer layer = new([new Neuron([], 1.0)]);
        double result = n.CalculateValue(layer, NeuronActivations.GetFunction(ActivationFunctionKind.Identity));
        await Assert.That(result).IsEqualTo(1.0);
    }

    [Test]
    public async Task NeuronCanReadInputLayerOfInputNeurons2() {
        Neuron n = new([1.0, 1.0], 0.0);
        NeuronLayer layer = new([new Neuron([], 1.0), new Neuron([], 1.0)]);
        double result = n.CalculateValue(layer, NeuronActivations.GetFunction(ActivationFunctionKind.Identity));
        await Assert.That(result).IsEqualTo(2.0);
    }

    [Test]
    public async Task NeuronCanReadInputLayerOfInputNeurons3() {
        Neuron n = new([1.0, 1.0], 0.0);
        NeuronLayer layer = new([new Neuron([], 1.0), new Neuron([], -1.0)]);
        double result = n.CalculateValue(layer, NeuronActivations.GetFunction(ActivationFunctionKind.Identity));
        await Assert.That(result).IsEqualTo(0.0);
    }

    [Test]
    public async Task NeuronCanReadInputLayerOfInputNeurons4() {
        Neuron n = new([1.0, 1.0, 0.5], 0.0);
        NeuronLayer layer = new([new Neuron([], 1.0), new Neuron([], -1.0), new Neuron([], 3.0)]);
        double result = n.CalculateValue(layer, NeuronActivations.GetFunction(ActivationFunctionKind.Identity));
        await Assert.That(result).IsEqualTo(1.5);
    }

    [Test]
    public async Task NeuronThrowsErrorWhenWeightsMismatchLength() {
        Neuron n = new([1.0, 1.0, 0.5], 0.0);
        NeuronLayer layer = new([new Neuron([], 1.0), new Neuron([], -1.0)]);
        await Assert.That(() => n.CalculateValue(layer, NeuronActivations.GetFunction(ActivationFunctionKind.Identity)))
            .Throws<Exception>()
            .WithMessage("Input neuron count is not the same as weight initalization. Found lengths 2 and 3.");
    }

    [Test]
    public async Task NeuronThrowsErrorWhenNotAnInputNeuronAndLayerIsNull() {
        Neuron n = new([1.0, 1.0, 0.5], 0.0);
        await Assert.That(() => n.CalculateValue(null, NeuronActivations.GetFunction(ActivationFunctionKind.Identity)))
            .Throws<Exception>()
            .WithMessage("Input layer is null when neuron is not an input neuron.");
    }
}
public class ImplicitNeuronTests {
    [Test]
    public async Task CanImplicitlyMakeANeuron1() {
        Neuron n = ([1.0, 1.0, 0.5], 0.0);
        await Assert.That(n.Bias).IsEqualTo(0.0);
        await Assert.That(n.Weights[0]).IsEqualTo(1.0);
        await Assert.That(n.Weights[1]).IsEqualTo(1.0);
        await Assert.That(n.Weights[2]).IsEqualTo(0.5);
    }
}