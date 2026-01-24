using HowlDev.AI.Core;
using HowlDev.AI.Core.Classes;
using HowlDev.AI.Structures.NeuralNetwork;
using HowlDev.AI.Structures.NeuralNetwork.Options;
using HowlDev.AI.Training.Saving;
using System.Collections.Concurrent;
using System.Diagnostics;

namespace HowlDev.AI.Training.Genetic;

/// <summary>
/// Runs a genetic algorithm, creating random networks and testing them with the <c>TRunner</c>.
/// </summary>
/// <typeparam name="TRunner">Object that runs the simulation</typeparam>
public class GeneticAlgorithm<TRunner>
    where TRunner : IGeneticRunner<int> {
    private readonly GeneticAlgorithmStrategy strategy;
    private readonly IResultReader reader;
    private readonly ConcurrentDictionary<int, SimpleNeuralNetwork> networks;
    private readonly ConcurrentDictionary<int, double> results;
    private int currentId;
    private int generation;
    private readonly NetworkSavingScheme savingScheme;
    private readonly GenerationStrategy generationStrategy;
    private readonly int countPerGroup;
    private readonly int numberOfTicks;
    private readonly int totalNetworks;
    private readonly ActivationFunctionKind functionKind;
    // provided by guess who
    private readonly Func<double, double, double, double> Lerp = (a, b, t) => a * (1 - t) + b * t;

    /// <summary>
    /// Default constructor. Must provide a Strategy for the system and a Reader to get values out of the simulation.
    /// </summary>
    /// <exception cref="ArgumentNullException"></exception>
    public GeneticAlgorithm(GeneticAlgorithmStrategy strategy, IResultReader reader) {
        // The below is done to ensure the input layer is set to false.
        this.strategy = new(strategy.GenerationStrategy, new(strategy.TopologyOptions.InputCount, strategy.TopologyOptions.HiddenLayerSizes, strategy.TopologyOptions.OutputCount) { CreateInputLayer = false }, strategy.InitializationOptions);
        this.reader = reader ?? throw new ArgumentNullException(nameof(GeneticAlgorithm<TRunner>.reader));
        functionKind = strategy.FunctionKind;
        networks = new ConcurrentDictionary<int, SimpleNeuralNetwork>();
        results = new ConcurrentDictionary<int, double>();
        currentId = 1;
        generation = 1;
        savingScheme = strategy.SavingStrategy;
        generationStrategy = strategy.GenerationStrategy;
        countPerGroup = generationStrategy.CountPerGroup;
        numberOfTicks = generationStrategy.NumberOfTicks;
        totalNetworks = generationStrategy.NumberOfGroups * generationStrategy.CountPerGroup;
    }

    /// <summary>
    /// Run the training system. Will likely take a long time, depending on settings. 
    /// </summary>
    public void StartTraining() {
        InitializeNetworks();
        for (int i = 0; i < generationStrategy.NumOfGenerations; i++) {
            Debug.Assert(networks.Count == totalNetworks);

            int[] randomizedIds = [.. networks.Keys.OrderBy(k => Random.Shared.Next())];

            Parallel.For(0, generationStrategy.NumberOfGroups, (j) => {
                int start = j * countPerGroup;
                int end = Math.Min(start + countPerGroup, randomizedIds.Length);
                int length = end - start;
                var slice = new int[length];
                Array.Copy(randomizedIds, start, slice, 0, length);
                RunGroup(slice);
            });

            Debug.Assert(results.Count == totalNetworks);

            (int id, double result)[] localResults = [.. results.Select((a) => (a.Key, a.Value))];
            localResults = [.. localResults.OrderByDescending(a => a.result)];
            GeneticAlgorithmResult result = new(generation) {
                Results = [.. localResults]
            };

            if (savingScheme == NetworkSavingScheme.All) {
                foreach (var item in networks) {
                    result.Networks.Add((item.Key, item.Value.ToTextFormat()));
                }
            }

            List<(int id, double score)> survivors = [];
            List<(int id, double score)> dead = [.. results.OrderByDescending(a => a.Value)
                .Select((a, index) => (a.Key, Lerp((index / (double)(totalNetworks - 1)) < 0.5 ? 1 : 0, 0.5, strategy.CullingStrategy.SelectionSoftness)))];

            int goal = (int)Math.Round((double)totalNetworks / 2);

            for (int j = 0; j < dead.Count; j++) {
                if (Random.Shared.NextDouble() < dead[j].score) {
                    survivors.Add(dead[j]);
                    dead.Remove(dead[j]);
                    j--;
                }
            }

            if (survivors.Count > goal) {
                dead.AddRange(survivors[goal..]);
                survivors = [.. survivors
                    .OrderByDescending(x => x.score)
                    .Take(goal)];
            }
            if (survivors.Count < goal) {
                int difference = goal - survivors.Count;
                survivors.AddRange(dead.Take(difference));
                dead = [.. dead[difference..]];
            }

            if (savingScheme == NetworkSavingScheme.Best) {
                int bestNetwork = results.OrderByDescending(a => a.Value).First().Key;
                result.Networks.Add((bestNetwork, networks[bestNetwork].ToTextFormat()));
            }

            int removed = 0;
            foreach (var (id, score) in dead) {
                if (networks.TryRemove(id, out _)) removed++;
            }

            if (savingScheme == NetworkSavingScheme.Survivors) {
                foreach (var item in networks) {
                    result.Networks.Add((item.Key, item.Value.ToTextFormat()));
                }
            }

            for (int j = 0; j < survivors.Count; j++) {
                SimpleNeuralNetwork network = GenerateNetwork(networks[survivors[j].id]);
                networks.AddOrUpdate(currentId++, network, (_k, _e) => network);
            }

            reader.ReadResult(result);

            results.Clear();
            generation++;
        }
    }

    private void InitializeNetworks() {
        for (int i = 0; i < totalNetworks; i++) {
            SimpleNeuralNetwork network = GenerateNetwork();
            networks.AddOrUpdate(currentId++, network, (_k, _e) => network);
        }
    }

    private SimpleNeuralNetwork GenerateNetwork(SimpleNeuralNetwork? offOf = null) {
        if (offOf is null) {
            return new SimpleNeuralNetwork(strategy.TopologyOptions, strategy.InitializationOptions);
        } else {
            WeightAndBiasChange last = strategy.LearningSpeed.Changes.Reverse().First(a => a.generation <= generation).change;
            SimpleNeuralNetwork newNetwork = SimpleNeuralNetwork.FromTextFormat(offOf.ToTextFormat());
            newNetwork.VaryNeuronWeights(last.WeightValue, last.BiasValue);
            return newNetwork;
        }
    }

    private void RunGroup(int[] ids) {
        IGeneticRunner<int> runner = TRunner.Initialize(ids);
        SimpleNeuralNetwork[] myNetworks = new SimpleNeuralNetwork[ids.Length];
        for (int i = 0; i < myNetworks.Length; i++) {
            myNetworks[i] = networks.GetValueOrDefault(ids[i])!;
        }

        for (int i = 0; i < numberOfTicks; i++) {
            for (int j = 0; j < myNetworks.Length; j++) {
                double[] input = runner.GetRepresentation(ids[j]);
                myNetworks[j].CalculateLayers(new NeuronLayer(input), functionKind);
                runner.PrepareAction(ids[j], [.. myNetworks[j].OutputLayer.Values]);
            }
            runner.RunTick();
        }

        for (int i = 0; i < myNetworks.Length; i++) {
            double eval = runner.GetEvaluation(ids[i]);
            results.AddOrUpdate(ids[i], eval, (_k, _old) => eval);
        }
    }
}