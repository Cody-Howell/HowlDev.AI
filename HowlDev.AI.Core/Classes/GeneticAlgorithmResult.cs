namespace HowlDev.AI.Core.Classes;

/// <summary>
/// Result of one generation of a Genetic Algorithm. 
/// </summary>
public class GeneticAlgorithmResult {
    /// <summary/>
    public GeneticAlgorithmResult(int generation) {
        Generation = generation;
    }

    /// <summary>
    /// Current generation.
    /// </summary>
    public int Generation { get; set; }
    /// <summary>
    /// Results from all the networks
    /// </summary>
    public (int id, double result)[] Results { get; set; } = [];
    /// <summary>
    /// Networks saved according to the strategy in the GeneticAlgorithm.
    /// </summary>
    public List<(int id, string network)> Networks { get; set; } = [];
}