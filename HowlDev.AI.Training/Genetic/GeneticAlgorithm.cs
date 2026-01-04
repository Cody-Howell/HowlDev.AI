using HowlDev.AI.Core;
using HowlDev.AI.Structures.NeuralNetwork;
using System.Collections.Concurrent;
using System.Data;

namespace HowlDev.AI.Training.Genetic;

public class GeneticAlgorithm<TRunner>(GeneticAlgorithmStrategy strategy, IFileWriter writer)
    where TRunner : IGeneticRunner<int> {
    private ConcurrentDictionary<int, SimpleNeuralNetwork> networks = [];
    private ConcurrentDictionary<int, double> results = [];
    private int currentId = 1;
    private int generation = 1;
    private GenerationStrategy generationStrategy = strategy.GenerationStrategy;
    private int countPerGroup = strategy.GenerationStrategy.CountPerGroup;
    private int numberOfTicks = strategy.GenerationStrategy.NumberOfTicks;
    private int totalNetworks = strategy.GenerationStrategy.NumberOfGroups * strategy.GenerationStrategy.CountPerGroup;
    // provided by guess who
    private Func<double, double, double, double> Lerp = (a, b, t) => a * (1 - t) + b * t;


    public async Task StartTraining() {
        InitializeNetworks();
        for (int i = 0; i < generationStrategy.NumOfGenerations; i++) {
            int[] randomizedIds = [.. networks.Select(a => a.Key)];
            Random.Shared.Shuffle(randomizedIds);

            List<Task> tasks = [];
            for (int j = 0; j < generationStrategy.NumberOfGroups; j++) {
                tasks.Add(Task.Run(() => RunGroup(randomizedIds[(j * countPerGroup)..(j * countPerGroup + countPerGroup)])));
            }
            await Task.WhenAll(tasks);

            (int id, double result)[] localResults = [.. results.Select((a) => (a.Key, a.Value))];
            localResults = [.. localResults.OrderByDescending(a => a.id)];
            writer.WriteFile($"{DateTime.Now:s}-Gen-{i}", string.Join('\n', localResults.Select(a => $"{a.id}: {a.result}")));

            List<(int id, double score)> survivors = [];
            // Reassigns result value as the fitness score to cull by
            List<(int id, double score)> dead = [.. results.Select(
                (a, i) => (a.Key, Lerp((i / totalNetworks - 1) < 0.5 ? 1 : 0, 0.5, strategy.CullingStrategy.SelectionSoftness)))
                .OrderByDescending(a => a.Item2)];

            int goal = (int)Math.Round((double)totalNetworks / 2);

            for (int j = 0; j < localResults.Length; j++) {
                if (Random.Shared.NextDouble() < localResults[j].result) {
                    dead.Remove(localResults[j]);
                    survivors.Add(localResults[j]);
                }
            }
            if (survivors.Count > goal) {
                survivors = [.. survivors
                    .OrderByDescending(x => x.score)
                    .Take(goal)];
            }
            if (survivors.Count < goal) {
                int difference = goal - survivors.Count;
                survivors.AddRange(dead.Take(difference));
            }
            
            for (int j = 0; j < survivors.Count; j++) {
                networks[currentId++] = GenerateNetwork(networks[survivors[j].id]);
            }

            for (int j = 0; j < dead.Count; j++) {
                networks.Remove(dead[j].id, out _);
            }

            results.Clear();
        }
    }

    private void InitializeNetworks() {
        for (int i = 0; i < totalNetworks; i++) {
            networks[currentId++] = GenerateNetwork();
        }
    }

    private SimpleNeuralNetwork GenerateNetwork(SimpleNeuralNetwork? offOf = null) {
        if (offOf is null) {
            return new(strategy.TopologyOptions, strategy.InitializationOptions);
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
                myNetworks[j].CalculateLayers(new NeuronLayer(input));
                runner.PrepareAction(ids[j], [.. myNetworks[j].OutputLayer.Values]);
            }
            runner.RunTick();
        }

        for (int i = 0; i < myNetworks.Length; i++) {
            results.AddOrUpdate(ids[i], runner.GetEvaluation(ids[i]), (d, d2) => ids[i]);
        }
    }
}