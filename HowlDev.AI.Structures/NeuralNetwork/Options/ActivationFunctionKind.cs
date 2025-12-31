namespace HowlDev.AI.Structures.NeuralNetwork.Options;

public enum ActivationFunctionKind {
    /// <summary>
    /// Returns the value exactly as provided. <br/>
    /// (-inf, inf)
    /// </summary>
    Identity,
    /// <summary>
    /// Returns the value through the Tanh function. <br/>
    /// (-1, 1)
    /// </summary>
    Tanh,
    /// <summary>
    /// If greater than 0, returns the value. Otherwise returns 0. <br/>
    /// [0, inf)
    /// </summary>
    ReLU,
    /// <summary>
    /// <b>Requires a parameter</b>. <br/>
    /// If greater than 0, returns the value. Otherwise returns the value times the parameter. Normal is maybe 0.01. <br/>
    /// (-inf, inf)
    /// </summary>
    LeakyReLU,
    /// <summary>
    /// Returns the value through the following function: x / (1 + |x|). <br/>
    /// (-1, 1)
    /// </summary>
    SoftSign,
    /// <summary>
    /// Returns the value through the following function: 1 / (1 + e^(-x)). <br/>
    /// (0, 1)
    /// </summary>
    Sigmoid
}