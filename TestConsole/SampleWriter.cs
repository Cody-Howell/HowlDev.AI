using HowlDev.AI.Core;
using HowlDev.AI.Core.Classes;

namespace TestConsole;

public class SampleWriter : IResultReader {
    public void ReadResult(GeneticAlgorithmResult result) {
        Console.WriteLine($"Path: {result.Generation}");
        Console.WriteLine("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-");
    }
}