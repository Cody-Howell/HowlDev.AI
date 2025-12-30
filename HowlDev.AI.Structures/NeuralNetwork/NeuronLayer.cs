namespace HowlDev.AI.Structures.NeuralNetwork;

public class NeuronLayer {
    private Neuron[] neurons;
    private double[]? values = null;

    public double[] Values {
        get {
            if (values is null) throw new InvalidOperationException("Layer has not yet been calculated/was cleared.");
            return values;
        }
    }

    public void Clear() {
        values = null;
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

    public void CalculateLayer(NeuronLayer? layer, Func<double, double> activation) {
        values = new double[neurons.Length];
        int index = 0;
        foreach (Neuron n in neurons) {
            values[index] = n.CalculateValue(layer, activation);
            index++;
        }
    }
}