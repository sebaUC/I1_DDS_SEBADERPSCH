namespace Classifier.Tests;

public class ControllerKnnTests
{
    [Theory]
    [InlineData("one-hot", "knn_tie", 1, 2)]
    [InlineData("one-hot", "knn_tie", 1, 3)]
    [InlineData("one-hot", "knn_tie2", 0, 2)]
    [InlineData("one-hot", "knn_tie2", 0, 3)]
    [InlineData("one-hot", "knn_tie3", 0, 2)]
    [InlineData("one-hot", "knn_tie3", 0, 3)]
    [InlineData("label", "knn_tie", 1, 2)]
    [InlineData("label", "knn_tie", 1, 3)]
    [InlineData("label", "knn_tie2", 0, 2)]
    [InlineData("label", "knn_tie2", 0, 3)]
    [InlineData("label", "knn_tie3", 0, 2)]
    [InlineData("label", "knn_tie3", 0, 3)]
    [InlineData("one-hot", "knn_noise", 0.5, 1)]
    [InlineData("one-hot", "knn_noise", 0.5, 2)]
    [InlineData("one-hot", "knn_noise", 1, 3)]
    [InlineData("label", "knn_noise", 0.5, 1)]
    [InlineData("label", "knn_noise", 0.5, 2)]
    [InlineData("label", "knn_noise", 1, 3)]
    [InlineData("one-hot", "knn_unseen", 0.5, 1)]
    [InlineData("one-hot", "knn_unseen", 0.5, 2)]
    [InlineData("label", "knn_unseen", 0.5, 1)]
    [InlineData("label", "knn_unseen", 0.5, 2)]
    [InlineData("one-hot", "car", 0.6560694, 1)]
    [InlineData("one-hot", "car", 0.6445087, 3)]
    [InlineData("one-hot", "car", 0.6705202, 5)]
    [InlineData("one-hot", "car", 0.7312139, 20)]
    [InlineData("label", "car", 0.8583815, 1)]
    [InlineData("label", "car", 0.9277457, 3)]
    [InlineData("label", "car", 0.9335260, 5)]
    [InlineData("label", "car", 0.8872832, 20)]
    public void Handle_ReturnCorrectAccuracyForKNN(string encoderType, string dataset, double accuracy, int k)
    {
        ClassificationRequest request = new ClassificationRequest(encoderType, "knn", "accuracy", dataset, k: k);
        AssertSuccessfulResult(request, accuracy);
    }
    
    [Theory]
    [InlineData("one-hot", "knn_tie", 1, 2)]
    [InlineData("one-hot", "knn_tie", 1, 3)]
    [InlineData("one-hot", "knn_tie2", 0, 2)]
    [InlineData("one-hot", "knn_tie2", 0, 3)]
    [InlineData("one-hot", "knn_tie3", 0, 2)]
    [InlineData("one-hot", "knn_tie3", 0, 3)]
    [InlineData("label", "knn_tie", 1, 2)]
    [InlineData("label", "knn_tie", 1, 3)]
    [InlineData("label", "knn_tie2", 0, 2)]
    [InlineData("label", "knn_tie2", 0, 3)]
    [InlineData("label", "knn_tie3", 0, 2)]
    [InlineData("label", "knn_tie3", 0, 3)]
    [InlineData("one-hot", "knn_noise", 0.5, 1)]
    [InlineData("one-hot", "knn_noise", 0.5, 2)]
    [InlineData("one-hot", "knn_noise", 1, 3)]
    [InlineData("label", "knn_noise", 0.5, 1)]
    [InlineData("label", "knn_noise", 0.5, 2)]
    [InlineData("label", "knn_noise", 1, 3)]
    [InlineData("one-hot", "knn_unseen", 0.5, 1)]
    [InlineData("one-hot", "knn_unseen", 0.5, 2)]
    [InlineData("label", "knn_unseen", 0.5, 1)]
    [InlineData("label", "knn_unseen", 0.5, 2)]
    [InlineData("one-hot", "car", 0.3899964 , 1)]
    [InlineData("one-hot", "car", 0.3105531 , 3)]
    [InlineData("one-hot", "car", 0.3109958 , 5)]
    [InlineData("one-hot", "car", 0.3282626 , 20)]
    [InlineData("label", "car", 0.7393856, 1)]
    [InlineData("label", "car", 0.8102693, 3)]
    [InlineData("label", "car", 0.7679707, 5)]
    [InlineData("label", "car", 0.6087889, 20)]
    public void Handle_ReturnCorrectRecallForKNN(string encoderType, string dataset, double recall, int k)
    {
        ClassificationRequest request = new ClassificationRequest(encoderType, "knn", "recall", dataset, k: k);
        AssertSuccessfulResult(request, recall);
    }
    
    private void AssertSuccessfulResult(ClassificationRequest request, double expected)
    {
        Controller controller = new();
        var result = controller.Handle(request);
        Assert.Null(result.ErrorMessage);
        Assert.Equal(expected, (double)result.Score!, 0.0000001);
    }
}