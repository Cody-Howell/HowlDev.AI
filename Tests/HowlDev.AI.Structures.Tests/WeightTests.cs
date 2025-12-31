using HowlDev.AI.Structures.NeuralNetwork;
using HowlDev.AI.Structures.NeuralNetwork.Options;

namespace HowlDev.AI.Structures.Tests;

public class WeightTests {
    [Test]
    public async Task ConstantReturnsMax() {
        WeightInitializationOptions o = new() {
            Max = 2.0,
            Strategy = WeightInitializationStrategy.Constant
        };
        var func = WeightInitialization.GetWeightGenerationTechnique(o);
        await Assert.That(func()).IsEqualTo(2.0);
    }

    [Test]
    [Retry(3)]
    [Arguments(0.0, 1.0)]
    [Arguments(-10.0, 10.0)]
    public async Task NormalDoesntGoOutOfBounds(double min, double max) {
        WeightInitializationOptions o = new() {
            Min = min,
            Max = max,
            Strategy = WeightInitializationStrategy.Uniform
        };
        var func = WeightInitialization.GetWeightGenerationTechnique(o);
        double localMin = max;
        double localMax = min;
        for (int i = 0; i < 1_000; i++) {
            double result = func();
            if (result < localMin) localMin = result;
            if (result > localMax) localMax = result;

            await Assert.That(result).IsBetween(min, max);
        }
        // Some extra checks to make sure that it gets close
        await Assert.That(localMin).IsBetween(min, min + (max - min) / 100);
        await Assert.That(localMax).IsBetween(max - (max - min) / 100, max);
    }
}