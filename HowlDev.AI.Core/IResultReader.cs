using HowlDev.AI.Core.Classes;

namespace HowlDev.AI.Core;

/// <summary>
/// An interface to get information out of a GeneticAlgorithm run. 
/// </summary>
public interface IResultReader {
    /// <summary>
    /// This function is called from the GeneticAlgorithm to give 
    /// you information about what happened in a generation. 
    /// </summary>
    void ReadResult(GeneticAlgorithmResult result);
}