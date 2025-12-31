namespace HowlDev.AI.Structures.NeuralNetwork;

public class NeuronLayer : IEquatable<NeuronLayer> {
    private Neuron[] neurons;
    public Neuron[] Neurons { get => neurons; }
    private double[]? values = null;

    public double[] Values {
        get {
            if (values is null) throw new InvalidOperationException("Layer has not yet been calculated/was cleared.");
            return values;
        }
    }

    public NeuronLayer(Neuron[] _neurons) {
        neurons = _neurons;
        var count = neurons.Count(a => a.IsInputNeuron);
        if (count != 0 && count != neurons.Length) {
            throw new ArgumentException("Not all neurons are the same type (input/other).");
        }

        if (neurons.First().IsInputNeuron) {
            CalculateLayer(null, (d) => d);
        }
    }

    public void Clear() {
        values = null;
    }

    public void CalculateLayer(NeuronLayer? layer, Func<double, double> activation) {
        values = new double[neurons.Length];
        int index = 0;
        foreach (Neuron n in neurons) {
            values[index] = n.CalculateValue(layer, activation);
            index++;
        }
    }

    public string ToTextFormat() {
        return $"""
        {string.Join('\n', neurons.Select(n => n.ToTextFormat()))}
        """;
    }

    /// <summary>
    /// Just expects a newline-separated list of neurons.
    /// </summary>
    public static NeuronLayer FromTextFormat(string input) {
        Neuron[] n = [.. input.Split('\n').Select(v => Neuron.FromTextFormat(v))];
        return new NeuronLayer(n);
    }

    /// <summary>
    /// Creates a layer given the size and function required. 
    /// </summary>
    /// <param name="size">Size of this layer</param>
    /// <param name="previousLayerSize">Length of internal weights. Set to 0 if this is an input layer.</param>
    /// <param name="weightGen">Function to generate weights from</param>
    /// <param name="initBias">Bias for all the neurons</param>
    public static NeuronLayer MakeLayer(int size, int previousLayerSize, Func<double> weightGen, double initBias) {
        Neuron[] neurons = new Neuron[size];
        if (previousLayerSize == 0) {
            for (int i = 0; i < size; i++) {
                neurons[i] = ([], initBias);
            }
        } else {
            for (int i = 0; i < size; i++) {
                neurons[i] = Neuron.MakeNeuron(previousLayerSize, weightGen, initBias);
            }
        }
        return new(neurons);
    }

    public bool Equals(NeuronLayer? other) {
        if (other is null || neurons.Length != other.neurons.Length) return false;
        for (int i = 0; i < neurons.Length; i++) {
            if (!neurons[i].Equals(other.neurons[i])) return false;
        }
        return true;
    }
}