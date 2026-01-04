namespace HowlDev.AI.Training.Genetic;

public class CullingStrategy {
    /// <summary>
    /// Variance in who to choose to be culled and duplicated. <br/>
    /// 1.0 represents pure randomness in who gets chosen, and 0.0 represents that only 
    /// the lowest get culled every time. 
    /// </summary>
    public double SelectionSoftness { get; set; } = 0.5;
}