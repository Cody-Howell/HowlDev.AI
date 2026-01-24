namespace HowlDev.AI.Core.Classes;

public class GeneticAlgorithmResult {
    public GeneticAlgorithmResult(int generation) {
        Generation = generation;
    }

    public int Generation { get; set; }
    public (int id, double result)[] Results { get; set; } = [];
    public List<(int id, string network)> Networks { get; set; } = [];
}