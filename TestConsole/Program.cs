

using HowlDev.AI.Training.Genetic;
using TestConsole;

GeneticAlgorithm<SampleClass> alg = new(new(new() {NumOfGenerations = 100, NumberOfGroups = 3, CountPerGroup = 2}, new(2, [5], 3), new()), new SampleWriter());

await alg.StartTraining();