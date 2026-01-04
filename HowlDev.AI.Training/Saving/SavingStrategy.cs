namespace HowlDev.AI.Training.Saving;

/// <summary>
/// Determines what and how things are saved to disk.
/// </summary>
public class SavingStrategy {
    /// <summary>
    /// Determines the filename in the output folder. Defaults to GenerationAndID.
    /// </summary>
    public FileNamingScheme NamingScheme { get; set; } = FileNamingScheme.GenerationAndID;
    /// <summary>
    /// Determines which character splits the different values from the <see cref="NamingScheme"/> parameter. 
    /// </summary>
    public char SchemeSeparator { get; set; } = '_';
    /// <summary>
    /// Determine which networks to send to the IFileWriter. Defaults to only saving Survivors.
    /// </summary>
    public NetworkSavingScheme SavingScheme {get;set;} = NetworkSavingScheme.Survivors;
}