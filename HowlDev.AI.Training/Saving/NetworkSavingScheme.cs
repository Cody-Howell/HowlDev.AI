namespace HowlDev.AI.Training.Saving; 

/// <summary>
/// Determines which networks to send to the FileWriter.
/// </summary>
public enum NetworkSavingScheme {
    /// <summary>
    /// After running a generation, save all networks. 
    /// </summary>
    All,
    /// <summary>
    /// After culling the generation (and before generating the next ones), 
    /// save ones that survived.
    /// </summary>
    Survivors,
    /// <summary>
    /// Save the network that scored the best each generation.
    /// </summary>
    Best
}