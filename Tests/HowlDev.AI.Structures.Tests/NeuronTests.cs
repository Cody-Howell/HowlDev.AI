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
public class NeuronGenerationTests {
    [Test]
    public async Task CanGenerateANeuron1() {
        Neuron n = Neuron.MakeNeuron(0, () => 1.0, 1.0);
        await Assert.That(n.Bias).IsEqualTo(1.0);
        await Assert.That(n.Weights.Length).IsEqualTo(0);
        await Assert.That(n.IsInputNeuron).IsEqualTo(true);
    }

    [Test]
    public async Task CanGenerateANeuron2() {
        Neuron n = Neuron.MakeNeuron(2, () => 1.0, 1.0);
        await Assert.That(n.Bias).IsEqualTo(1.0);
        await Assert.That(n.Weights.Length).IsEqualTo(2);
        await Assert.That(n.Weights[0]).IsEqualTo(1.0);
        await Assert.That(n.Weights[1]).IsEqualTo(1.0);
        await Assert.That(n.IsInputNeuron).IsEqualTo(false);
    }
}
public class NeuronEqualsTests {
    [Test]
    public async Task IsEquals() {
        Neuron n1 = new(weights: [2.0, 3.0], bias: 2.0);
        Neuron n2 = new(weights: [2.0, 3.0], bias: 2.0);
        await Assert.That(n1.Equals(n2)).IsEqualTo(true);
    }

    [Test]
    public async Task BiasDoesntMatch() {
        Neuron n1 = new(weights: [2.0, 3.0], bias: 2.0);
        Neuron n2 = new(weights: [2.0, 3.0], bias: 1.0);
        await Assert.That(n1.Equals(n2)).IsEqualTo(false);
    }

    [Test]
    public async Task WeightLengthDoesntMatch() {
        Neuron n1 = new(weights: [2.0], bias: 2.0);
        Neuron n2 = new(weights: [2.0, 3.0], bias: 2.0);
        await Assert.That(n1.Equals(n2)).IsEqualTo(false);
    }

    [Test]
    public async Task WeightsDontMatch() {
        Neuron n1 = new(weights: [2.0, 1.0], bias: 2.0);
        Neuron n2 = new(weights: [2.0, 3.0], bias: 2.0);
        await Assert.That(n1.Equals(n2)).IsEqualTo(false);
    }
}
public class NeuronToTextTests {
    [Test]
    public async Task InputNeuronToText() {
        Neuron n = Neuron.MakeNeuron(0, () => 1.0, 2.0);
        await Assert.That(n.ToTextFormat()).IsEqualTo("2");
    }

    [Test]
    public async Task NeuronToText() {
        Neuron n = Neuron.MakeNeuron(2, () => 1.0, 2.0);
        await Assert.That(n.ToTextFormat()).IsEqualTo("2 1 1");
    }

    [Test]
    public async Task TextToInputNeuron() {
        Neuron n = Neuron.FromTextFormat("2.0");
        await Assert.That(n.Bias).IsEqualTo(2.0);
        await Assert.That(n.IsInputNeuron).IsEqualTo(true);
    }

    [Test]
    public async Task TextToNeuron() {
        Neuron n = Neuron.FromTextFormat("2.0 1.0 1.0");
        await Assert.That(n.Bias).IsEqualTo(2.0);
        await Assert.That(n.Weights.Length).IsEqualTo(2);
        await Assert.That(n.Weights[0]).IsEqualTo(1.0);
        await Assert.That(n.Weights[1]).IsEqualTo(1.0);
        await Assert.That(n.IsInputNeuron).IsEqualTo(false);
    }

    [Test]
    public async Task InputBackAndForth() {
        Neuron n = Neuron.MakeNeuron(0, () => 1.0, 2.0);
        Neuron recreation = Neuron.FromTextFormat(n.ToTextFormat());
        await Assert.That(recreation.Bias).IsEqualTo(2.0);
        await Assert.That(recreation.IsInputNeuron).IsEqualTo(true);
        await Assert.That(n.Equals(recreation)).IsEqualTo(true);
    }

    [Test]
    public async Task BackAndForth() {
        Neuron n = Neuron.MakeNeuron(2, () => 1.0, 2.0);
        Neuron recreation = Neuron.FromTextFormat(n.ToTextFormat());
        await Assert.That(recreation.Bias).IsEqualTo(2.0);
        await Assert.That(recreation.Weights.Length).IsEqualTo(2);
        await Assert.That(recreation.Weights[0]).IsEqualTo(1.0);
        await Assert.That(recreation.Weights[1]).IsEqualTo(1.0);
        await Assert.That(recreation.IsInputNeuron).IsEqualTo(false);
    }
}