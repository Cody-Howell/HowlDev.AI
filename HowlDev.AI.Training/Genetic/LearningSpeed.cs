namespace HowlDev.AI.Training.Genetic;

public class LearningSpeed {
    public (int generation, WeightAndBiasChange change)[] Changes { get; set; } = [
        (0, new() {WeightValue = 0.5, BiasValue = 0.5}),
        (500, new() {WeightValue = 0.25, BiasValue = 0.25}),
        (1000, new() {WeightValue = 0.1, BiasValue = 0.1}),
        (10000, new() {WeightValue = 0.01, BiasValue = 0.01})
    ];
}

public class WeightAndBiasChange {
    public double WeightValue { get; set; }
    public double BiasValue { get; set; }
}