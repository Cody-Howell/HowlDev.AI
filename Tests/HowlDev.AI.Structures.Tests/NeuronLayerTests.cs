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

    [Test]
    public async Task TwoLayersAfterClearingOneThrowsError() {
        NeuronLayer layer1 = new([([], 1.0), ([], -1.0), ([], 3.0)]);
        layer1.Clear();
        NeuronLayer layer2 = new([([1.0, 1.0, 1.0], 1.0), ([1.0, 1.0, 1.0], 2.0)]);
        await Assert.That(() => layer2.CalculateLayer(layer1, NeuronActivations.GetFunction(ActivationFunctionKind.Identity)))
            .Throws<InvalidOperationException>()
            .WithMessage("Layer has not yet been calculated/was cleared.");
    }
}
public class NeuronLayerGenerationTests {
    [Test]
    public async Task CanGenerateAnInputLayer() {
        NeuronLayer layer1 = NeuronLayer.MakeLayer(3, 0, () => 1.0, 2);
        await Assert.That(layer1.Values.Length).IsEqualTo(3);
        await Assert.That(layer1.Neurons[0].IsInputNeuron).IsEqualTo(true);
        await Assert.That(layer1.Neurons[1].IsInputNeuron).IsEqualTo(true);
        await Assert.That(layer1.Neurons[2].IsInputNeuron).IsEqualTo(true);
        await Assert.That(layer1.Neurons[0].Bias).IsEqualTo(2);
        await Assert.That(layer1.Neurons[1].Bias).IsEqualTo(2);
        await Assert.That(layer1.Neurons[2].Bias).IsEqualTo(2);
    }

    [Test]
    public async Task CanGenerateANormalLayer() {
        NeuronLayer layer1 = NeuronLayer.MakeLayer(3, 2, () => 1.0, 2);
        await Assert.That(layer1.Neurons.Length).IsEqualTo(3);
        await Assert.That(layer1.Neurons[0].Weights.Length).IsEqualTo(2);
        await Assert.That(layer1.Neurons[0].Weights[0]).IsEqualTo(1);
        await Assert.That(layer1.Neurons[0].Weights[1]).IsEqualTo(1);
        await Assert.That(layer1.Neurons[1].Weights.Length).IsEqualTo(2);
        await Assert.That(layer1.Neurons[2].Weights.Length).IsEqualTo(2);
    }
}
public class NeuronLayerToTextTests {
    [Test]
    public async Task CanMakeAnInputLayer() {
        NeuronLayer layer1 = NeuronLayer.MakeLayer(2, 0, () => 1.0, 2);
        await Assert.That(layer1.ToTextFormat()).IsEqualTo("""
            2
            2
            """);
    }

    [Test]
    public async Task CanMakeALayer() {
        NeuronLayer layer1 = NeuronLayer.MakeLayer(2, 3, () => 1.0, 2);
        await Assert.That(layer1.ToTextFormat()).IsEqualTo("""
            2 1 1 1
            2 1 1 1
            """);
    }

    [Test]
    public async Task CanMakeAnInputLayerBack() {
        NeuronLayer layer1 = NeuronLayer.FromTextFormat("""
            2
            3
            """);
        await Assert.That(layer1.Neurons.Length).IsEqualTo(2);
        await Assert.That(layer1.Neurons[0].IsInputNeuron).IsEqualTo(true);
        await Assert.That(layer1.Neurons[1].IsInputNeuron).IsEqualTo(true);
        await Assert.That(layer1.Neurons[0].Bias).IsEqualTo(2.0);
        await Assert.That(layer1.Neurons[1].Bias).IsEqualTo(3.0);
    }

    [Test]
    public async Task CanMakeALayerBack() {
        NeuronLayer layer1 = NeuronLayer.FromTextFormat("""
            2 1 1 1
            2 1 1 1
            """);
        await Assert.That(layer1.Neurons.Length).IsEqualTo(2);
        await Assert.That(layer1.Neurons[0].Weights.Length).IsEqualTo(3);
        await Assert.That(layer1.Neurons[1].Weights.Length).IsEqualTo(3);
    }

    [Test]
    public async Task CanMakeAnInputLayerAndBack() {
        NeuronLayer l1 = NeuronLayer.MakeLayer(2, 0, () => 1.0, 2);
        NeuronLayer l2 = NeuronLayer.FromTextFormat(l1.ToTextFormat());
        await Assert.That(l1.Equals(l2)).IsEqualTo(true);

    }

    [Test]
    public async Task CanMakeALayerAndBack() {
        NeuronLayer l1 = NeuronLayer.MakeLayer(2, 3, () => 1.0, 2);
        NeuronLayer l2 = NeuronLayer.FromTextFormat(l1.ToTextFormat());
        await Assert.That(l1.Equals(l2)).IsEqualTo(true);
    }
}