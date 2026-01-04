using HowlDev.AI.Core;

namespace TestConsole;

public class SampleClass : IGeneticRunner<int> {
    public static IGeneticRunner<int> Initialize(IEnumerable<int> ids) {
        return new SampleClass();
    }

    public double GetEvaluation(int id) {
        return Random.Shared.Next(0, 50);
    }

    public double[] GetRepresentation(int id) {
        return [12.0, 3.0, 0.23, 5, 2.4, 5];
    }

    public void PrepareAction(int id, List<double> outputs) {
        return;
    }

    public void RunTick() {
        return;
    }
}