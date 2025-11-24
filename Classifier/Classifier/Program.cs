// See https://aka.ms/new-console-template for more information

using Classifier;

Controller c = new Controller();

// ClassificationRequest r = new ClassificationRequest("label", "naive-bayes", "simple");
ClassificationRequest r = new ClassificationRequest("label", "knn", "recall", "car", k: 20,
    learningRate: 0.0000001, epochs: 100);

ClassificationResult result = c.Handle(r);

Console.WriteLine(result.ErrorMessage);
Console.WriteLine($"{result.Score:F7}");