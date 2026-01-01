namespace HowlDev.AI.Structures.NeuralNetwork;

/// <summary>
/// This Neuron holds a weight array and a bias value. Set the weight array to empty 
/// if it is an input neuron, and assign the bias appropriately. 
/// </summary>
public readonly struct Neuron(double[] weights, double bias = 0.0) : IEquatable<Neuron> {
    public double[] Weights => [.. weights];
    public double Bias => bias;
    public bool IsInputNeuron { get => weights.Length == 0; }
    /// <summary>
    /// Takes in an input layer and an activation function and returns the value of this neuron. 
    /// </summary>
    /// <param name="layer">Previous layer to calculate off of</param>
    /// <param name="activation">Function to pass the final sum through</param>
    /// <exception cref="ArgumentException"/>
    public double CalculateValue(NeuronLayer? layer, Func<double, double> activation) {
        if (IsInputNeuron) return activation(bias);
        if (layer is null) throw new ArgumentException("Input layer is null when neuron is not an input neuron.");
        if (layer.Values.Length != weights.Length) throw new ArgumentException($"Input neuron count is not the same as weight initalization. Found lengths {layer.Values.Length} and {weights.Length}.");

        double sum = bias;
        for (int i = 0; i < layer.Values.Length; i++) {
            sum += layer.Values[i] * weights[i];
        }
        return activation(sum);
    }

    public static Neuron MakeNeuron(int size, Func<double> weightGen, double bias) {
        double[] weights = new double[size];
        for (int i = 0; i < size; i++) {
            weights[i] = weightGen();
        }
        return new(weights, bias);
    }

    /// <summary>
    /// Randomly vary the weights +/- a given variance. Returns a new instance. 
    /// </summary>
    public Neuron VaryWeightsFromNeuron(double variance, double? biasVariance = null) {
        double[] newWeights = new double[weights.Length];
        for (int i = 0; i < weights.Length; i++) {
            newWeights[i] = (Random.Shared.NextDouble() * 2 - 1) * variance + weights[i];
        }
        double newBias = bias;
        if (biasVariance is not null) {
            newBias += (Random.Shared.NextDouble() * 2 - 1) * (double)biasVariance;
        }
        return new(newWeights, newBias);
    }

    public static Neuron FromTextFormat(string input) {
        double[] values = [.. input.Split(' ').Select(d => Convert.ToDouble(d))];
        return new Neuron(weights: values[1..], bias: values[0]);
    }

    public string ToTextFormat() {
        string output = $"{bias}";
        foreach (double weight in weights) {
            output += $" {weight}";
        }
        return output;
    }

    public bool Equals(Neuron other) {
        if (bias != other.Bias || weights.Length != other.Weights.Length) return false;
        for (int i = 0; i < weights.Length; i++) {
            if (weights[i] != other.Weights[i]) return false;
        }
        return true;
    }

    public static implicit operator Neuron((double[], double) tuple) {
        return new Neuron(tuple.Item1, tuple.Item2);
    }
}