namespace HowlDev.AI.Training.Genetic;

/// <summary>
/// Update the Weight and Bias values based on the generation.
/// </summary>
public class LearningSpeed {
    /// <summary>
    /// Set an array of changes based on generation. Defaults to the following: 
    /// <code> 
    /// (0, new(0.5, 0.5)),
    /// (500, new(0.25, 0.25)),
    /// (1000, new(0.1, 0.1)),
    /// (10000, new(0.01, 0.01))
    /// </code>
    /// </summary>
    public (int generation, WeightAndBiasChange change)[] Changes { get; set; } = [
        (0, new(0.5, 0.5)),
        (500, new(0.25, 0.25)),
        (1000, new(0.1, 0.1)),
        (10000, new(0.01, 0.01))
    ];
}

/// <summary>
/// Class to hold different weight and bias values per generation.
/// </summary>
public class WeightAndBiasChange(double weightValue, double biasValue) {

    /// <summary>
    /// Weight value range. 
    /// </summary>
    public double WeightValue { get; set; } = weightValue;
    /// <summary>
    /// Bias value range. 
    /// </summary>
    public double BiasValue { get; set; } = biasValue;
}