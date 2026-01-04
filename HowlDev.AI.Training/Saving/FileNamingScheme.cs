namespace HowlDev.AI.Training.Saving;

public enum FileNamingScheme {
    /// <summary>
    /// Puts the exact DateTime value, the generation number, and the ID of the 
    /// given network. 
    /// </summary>
    DateTimeAndGenerationAndID,
    /// <summary>
    /// Puts the generation number, the exact DateTime value, and the ID of 
    /// the given network.
    /// </summary>
    GenerationAndDateTimeAndID,
    /// <summary>
    /// Records the generation and the ID number of the given network.
    /// </summary>
    GenerationAndID

}