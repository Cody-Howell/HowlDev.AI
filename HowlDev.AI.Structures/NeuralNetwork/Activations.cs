using HowlDev.AI.Structures.NeuralNetwork.Options;

namespace HowlDev.AI.Structures.NeuralNetwork; 

public static class NeuronActivations {
    /// <summary>
    /// Get the activation function for a given <c>ActivationFunctionKind</c>. Some operations require an extra parameter, 
    /// such as LeakyReLU, so that parameter must be passed in to activate it. 
    /// </summary>
    /// <param name="kind">Type of activation</param>
    /// <param name="parameter">Parameter to apply</param>
    /// <exception cref="Exception"></exception>
    public static Func<double, double> GetFunction(ActivationFunctionKind kind, double? parameter = null) {
        return kind switch {
            ActivationFunctionKind.Identity => (d) => d,
            ActivationFunctionKind.Tanh => Math.Tanh,
            ActivationFunctionKind.ReLU => (d) => Math.Max(0, d),
            ActivationFunctionKind.LeakyReLU => (d) => {
                if (parameter is null) throw new Exception("Parameter is required for LeakyReLU function.");
                if (d >= 0.0) return d;
                return d * (double)parameter;
            },
            _ => throw new Exception($"NeuronActivations has not yet made a function for the ActivationFunctionKind {kind}."),
        };
    }
}