namespace HowlDev.AI.Core;

/// <summary>
/// Runs the simulation the algorithm is trying to optimize. 
/// </summary>
/// <typeparam name="T"></typeparam>
public interface IGeneticRunner<T> {
    /// <summary>
    /// Returns an instance of the Trainer with each of the listed IDs in the environment.
    /// </summary>
    static abstract IGeneticRunner<T> Initialize(IEnumerable<T> ids);
    /// <summary>
    /// Get the representation as a <c>double[]</c> for the neuron input to the neural network.
    /// </summary>
    double[] GetRepresentation(T id);
    /// <summary>
    /// Given a list of output neurons, have that ID take that action on the next tick.
    /// </summary>
    void PrepareAction(T id, List<double> outputs);
    /// <summary>
    /// Run one tick of the simulation.
    /// </summary>
    void RunTick();
    /// <summary>
    /// Get the evaluation metric for the given ID after running all the ticks.
    /// </summary>
    double GetEvaluation(T id);
}