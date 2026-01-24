

using HowlDev.AI.Training.Genetic;
using HowlDev.AI.Training.Saving;
using TestConsole;

GeneticAlgorithm<SampleClass> alg = new(new(new() { NumOfGenerations = 100, NumberOfGroups = 5, CountPerGroup = 4 },
    new(2, [5], 3),
    new()) { SavingStrategy = NetworkSavingScheme.Best },
new SampleWriter());

alg.StartTraining();