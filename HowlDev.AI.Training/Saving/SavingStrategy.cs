namespace HowlDev.AI.Training.Saving;

/// <summary>
/// Determines what and how things are saved to disk.
/// </summary>
public class SavingStrategy {
    /// <summary>
    /// Determines what folder to output to, relative to the project root. 
    /// </summary>
    public string SavePath { get; set; } = "/output";
    /// <summary>
    /// Determines the filename in the output folder. 
    /// </summary>
    public FileNamingScheme NamingScheme { get; set; } = FileNamingScheme.GenerationAndID;
    /// <summary>
    /// Determines which character splits the different values from the <see cref="NamingScheme"/> parameter. 
    /// </summary>
    public char SchemeSeparator { get; set; } = '_';
}