namespace Classifier;

public class ClassificationRequest(
    string encoderType,
    string classifierType,
    string metric,
    string dataset,
    int? k = null,
    double? learningRate = null,
    int? epochs = null)
{
    public string EncoderType { get; set; } = encoderType;
    public string ClassifierType { get; set; } = classifierType;
    public string Metric { get; set; } = metric;
    public string Dataset { get; set; } = dataset;
    public int? K { get; set; } = k;
    public double? LearningRate { get; set; } = learningRate;
    public int? Epochs { get; set; } = epochs;
}