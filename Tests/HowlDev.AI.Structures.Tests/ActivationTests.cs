using HowlDev.AI.Structures.NeuralNetwork;
using HowlDev.AI.Structures.NeuralNetwork.Options;

namespace HowlDev.AI.Structures.Tests;

public class ActivationTests {
    [Test]
    [Arguments(0, 0)]
    [Arguments(10, 10)]
    [Arguments(-10, -10)]
    public async Task IdentityTests(double d, double exp) {
        var func = NeuronActivations.GetFunction(ActivationFunctionKind.Identity);
        await Assert.That(func(d)).IsEqualTo(exp);
    }

    [Test]
    [Arguments(0, 0)]
    [Arguments(1, 1)]
    [Arguments(-1, 0)]
    [Arguments(297.23, 297.23)]
    [Arguments(-125.12, 0)]
    public async Task ReLU(double d, double exp) {
        var func = NeuronActivations.GetFunction(ActivationFunctionKind.ReLU);
        await Assert.That(func(d)).IsEqualTo(exp);
    }

    [Test]
    [Arguments(0)]
    [Arguments(1)]
    [Arguments(-1)]
    [Arguments(297.23)]
    [Arguments(-125.12)]
    public async Task Tanh(double d) {
        var func = NeuronActivations.GetFunction(ActivationFunctionKind.Tanh);
        await Assert.That(func(d)).IsEqualTo(Math.Tanh(d));
    }

    [Test]
    [Arguments(0, 0)]
    [Arguments(1, 1)]
    [Arguments(-1, -0.5)]
    [Arguments(297.23, 297.23)]
    [Arguments(-100, -50)]
    public async Task LeakyReLU(double d, double exp) {
        var func = NeuronActivations.GetFunction(ActivationFunctionKind.LeakyReLU, 0.5);
        await Assert.That(func(d)).IsEqualTo(exp);
    }

    [Test]
    public async Task LeakyReLUThrowsErrorsWithNoParameter() {
        var func = NeuronActivations.GetFunction(ActivationFunctionKind.LeakyReLU);
        await Assert.That(() => func(10))
            .Throws<Exception>()
            .WithMessage("Parameter is required for LeakyReLU function.");
    }

    [Test]
    [Arguments(0, 0)]
    [Arguments(1, 0.5)]
    [Arguments(-1, -0.5)]
    [Arguments(297.23, 0.999)]
    [Arguments(-100, -0.999)]
    public async Task SoftSign(double d, double exp) {
        var func = NeuronActivations.GetFunction(ActivationFunctionKind.SoftSign);
        await Assert.That(func(d) - exp).IsLessThan(0.01); // Is close enough
    }

    [Test]
    [Arguments(0, 0.5)]
    [Arguments(1, 1)]
    [Arguments(-1, 0.26)]
    [Arguments(297.23, 0.999)]
    [Arguments(-100, 0)]
    public async Task Sigmoid(double d, double exp) {
        var func = NeuronActivations.GetFunction(ActivationFunctionKind.Sigmoid);
        await Assert.That(func(d) - exp).IsLessThan(0.01);
    }
}