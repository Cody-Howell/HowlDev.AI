namespace HowlDev.AI.Structures.NeuralNetwork;

/// <summary>
/// This Neuron holds a weight array and a bias value. Set the weight array to empty 
/// if it is an input neuron, and assign the bias appropriately. 
/// </summary>
public readonly struct Neuron(double[] weights, double bias = 0.0) {
    public double[] Weights => [.. weights];
    public double Bias => bias;
    public bool IsInputNeuron {get => weights.Length == 0;}
    /// <summary>
    /// Takes in an input layer and an activation function and returns the value of this neuron. 
    /// </summary>
    /// <param name="layer">Previous layer to calculate off of</param>
    /// <param name="activation">Function to pass the final sum through</param>
    public double CalculateValue(NeuronLayer? layer, Func<double, double> activation) {
        if (IsInputNeuron) return activation(bias); // Is an input neuron
        if (layer is null) throw new ArgumentException("Input layer is null when neuron is not an input neuron.");
        if (layer.Values.Length != weights.Length) throw new ArgumentException($"Input neuron count is not the same as weight initalization. Found lengths {layer.Values.Length} and {weights.Length}.");

        double sum = bias;
        for (int i = 0; i < layer.Values.Length; i++) {
            sum += layer.Values[i] * weights[i];
        }
        return activation(sum);
    }
}