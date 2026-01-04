using HowlDev.AI.Core;

namespace TestConsole;

public class SampleWriter : IFileWriter {
    public void WriteFile(string path, string value) {
        Console.WriteLine($"Path: {path}, Value: {value}");
        Console.WriteLine("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-");
    }
}