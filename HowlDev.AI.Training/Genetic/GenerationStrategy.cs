namespace HowlDev.AI.Training.Genetic;

public class GenerationStrategy {
    /// <summary>
    /// How many generations to run.
    /// </summary>
    public int NumOfGenerations { get; set; } = 1000;
    /// <summary>
    /// How many groups will be made. Multiply by <see cref="CountPerGroup"/> to see how many networks will 
    /// be produced each generation.
    /// </summary>
    public int NumberOfGroups { get; set; } = 100;
    /// <summary>
    /// How many networks will be in each group. Multiply by <see cref="NumberOfGroups"/> to see how many networks will 
    /// be produced each generation.
    /// </summary>
    public int CountPerGroup { get; set; } = 5;
    /// <summary>
    /// Number of ticks to run for the <see cref="IGeneticRunner"/>.
    /// </summary>
    public int NumberOfTicks { get; set; } = 1000;
}