namespace Classifier.Tests;

public class ControllerErrorsTests
{
    [Theory]
    [InlineData("one-hot","accuracy")]
    [InlineData("label","accuracy")]
    [InlineData("one-hot","recall")]
    [InlineData("label","recall")]
    public void Handle_ReturnAnErrorMessageWhenClassifierTypeIsInvalid(string encoderType, string metric)
    {
        ClassificationRequest request =
            new ClassificationRequest(encoderType, "invalid-classifier-type", metric, "car");
        AssertErrorResult(request, "Invalid classifier type");
    }

    [Theory]
    [InlineData("knn", "accuracy")]
    [InlineData("knn", "recall")]
    [InlineData("logistic-regression", "accuracy")]
    [InlineData("logistic-regression", "recall")]
    public void Handle_ReturnAnErrorMessageWhenEncoderTypeIsInvalid(string classifierType, string metric)
    {
        ClassificationRequest request =
            new ClassificationRequest("invalid-encoder-type", classifierType, metric, "car", k: 1, learningRate: 0.1,
                epochs: 1);
        AssertErrorResult(request, "Invalid encoder type");
    }

    [Theory]
    [InlineData("one-hot","knn")]
    [InlineData("label","knn")]
    [InlineData("one-hot","logistic-regression")]
    [InlineData("label","logistic-regression")]
    public void Handle_ReturnAnErrorMessageWhenMetricIsInvalid(string encoderType, string classifierType)
    {
        ClassificationRequest request = new ClassificationRequest(encoderType, classifierType, "invalid-metric", "car",
            k: 1, learningRate: 0.1, epochs: 1);
        AssertErrorResult(request, "Invalid metric");
    }

    [Theory]
    [InlineData("one-hot", "knn", "accuracy")]
    [InlineData("one-hot", "knn", "recall")]
    [InlineData("one-hot", "logistic-regression", "accuracy")]
    [InlineData("one-hot", "logistic-regression", "recall")]
    [InlineData("label", "knn", "accuracy")]
    [InlineData("label", "knn", "recall")]
    [InlineData("label", "logistic-regression", "accuracy")]
    [InlineData("label", "logistic-regression", "recall")]
    public void Handle_ReturnAnErrorMessageWhenTrainDatasetFileIsNotFound(string encoderType, string classifierType,
        string metric)
    {
        ClassificationRequest request =
            new ClassificationRequest(encoderType, classifierType, metric, "non_existent_train_dataset", k: 1,
                learningRate: 0.1, epochs: 1);
        AssertErrorResult(request, "Dataset file not found: non_existent_train_dataset-train");
    }
    
    [Theory]
    [InlineData("one-hot", "knn", "accuracy")]
    [InlineData("one-hot", "knn", "recall")]
    [InlineData("one-hot", "logistic-regression", "accuracy")]
    [InlineData("one-hot", "logistic-regression", "recall")]
    [InlineData("label", "knn", "accuracy")]
    [InlineData("label", "knn", "recall")]
    [InlineData("label", "logistic-regression", "accuracy")]
    [InlineData("label", "logistic-regression", "recall")]
    public void Handle_ReturnAnErrorMessageWhenTestDatasetFileIsNotFound(string encoderType, string classifierType,
        string metric)
    {
        ClassificationRequest request =
            new ClassificationRequest(encoderType, classifierType, metric, "existent_train_dataset", k: 1,
                learningRate: 0.1, epochs: 1);
        AssertErrorResult(request, "Dataset file not found: existent_train_dataset-test");
    }

    [Theory]
    [InlineData("one-hot", "knn", "accuracy")]
    [InlineData("one-hot", "knn", "recall")]
    [InlineData("one-hot", "logistic-regression", "accuracy")]
    [InlineData("one-hot", "logistic-regression", "recall")]
    [InlineData("label", "knn", "accuracy")]
    [InlineData("label", "knn", "recall")]
    [InlineData("label", "logistic-regression", "accuracy")]
    [InlineData("label", "logistic-regression", "recall")]
    public void Handle_ReturnAnErrorMessageWhenTrainDatasetFileIsEmpty(string encoderType, string classifierType,
        string metric)
    {
        ClassificationRequest request =
            new ClassificationRequest(encoderType, classifierType, metric, "empty", k: 1,
                learningRate: 0.1, epochs: 1);
        AssertErrorResult(request, "Dataset empty-train is empty");
    }
    
    [Theory]
    [InlineData("one-hot", "knn", "accuracy")]
    [InlineData("one-hot", "knn", "recall")]
    [InlineData("one-hot", "logistic-regression", "accuracy")]
    [InlineData("one-hot", "logistic-regression", "recall")]
    [InlineData("label", "knn", "accuracy")]
    [InlineData("label", "knn", "recall")]
    [InlineData("label", "logistic-regression", "accuracy")]
    [InlineData("label", "logistic-regression", "recall")]
    public void Handle_ReturnAnErrorMessageWhenTestDatasetFileIsEmpty(string encoderType, string classifierType,
        string metric)
    {
        ClassificationRequest request =
            new ClassificationRequest(encoderType, classifierType, metric, "empty_test", k: 1,
                learningRate: 0.1, epochs: 1);
        AssertErrorResult(request, "Dataset empty_test-test is empty");
    }

    [Theory]
    [InlineData("one-hot", "accuracy")]
    [InlineData("label", "accuracy")]
    [InlineData("one-hot", "recall")]
    [InlineData("label", "recall")]
    public void Handle_ReturnAnErrorMessageIfKIsNotDefinedForKNN(string encoderType, string metric)
    {
        ClassificationRequest request = new ClassificationRequest(encoderType, "knn", metric, "car");
        AssertErrorResult(request, "number of neighbors must be specified");
    }

    [Theory]
    [InlineData("one-hot", "accuracy", 0)]
    [InlineData("one-hot", "accuracy", -1)]
    [InlineData("one-hot", "accuracy", -6)]
    [InlineData("label", "accuracy", 0)]
    [InlineData("label", "accuracy", -1)]
    [InlineData("label", "accuracy", -2)]
    [InlineData("one-hot", "recall", 0)]
    [InlineData("one-hot", "recall", -1)]
    [InlineData("one-hot", "recall", -24)]
    [InlineData("label", "recall", 0)]
    [InlineData("label", "recall", -1)]
    [InlineData("label", "recall", -3)]
    public void Handle_ReturnAnErrorMessageIfKIsNotPositiveForKNN(string encoderType, string metric, int k)
    {
        ClassificationRequest request = new ClassificationRequest(encoderType, "knn", metric, "car", k: k);
        AssertErrorResult(request, "number of neighbors must be positive");
    }

    [Theory]
    [InlineData("one-hot","accuracy","single-sample",2)]
    [InlineData("one-hot","recall","single-sample",2)]
    [InlineData("label","accuracy","single-sample",2)]
    [InlineData("label","recall","single-sample",2)]
    [InlineData("one-hot","accuracy","single-sample",10)]
    [InlineData("one-hot","recall","single-sample",10)]
    [InlineData("label","accuracy","single-sample",10)]
    [InlineData("label","recall","single-sample",10)]
    public void Handle_ReturnAnErrorMessageIfKIsGreaterThanTrainingSamplesForKNN(string encoderType, string metric,
        string dataset, int k)
    {
        ClassificationRequest request = new ClassificationRequest(encoderType, "knn", metric, dataset, k: k);
        AssertErrorResult(request, "number of neighbors must be smaller than the number of training samples");
    }
    
    [Theory]
    [InlineData("one-hot", "accuracy")]
    [InlineData("label", "accuracy")]
    [InlineData("one-hot", "recall")]
    [InlineData("label", "recall")]
    public void Handle_ReturnAnErrorMessageIfLearningRateIsNotDefinedForLogisticRegression(string encoderType,
        string metric)
    {
        ClassificationRequest request =
            new ClassificationRequest(encoderType, "logistic-regression", metric, "car", epochs: 1);
        AssertErrorResult(request, "learning rate must be specified");
    }

    [Theory]
    [InlineData("one-hot", "accuracy")]
    [InlineData("label", "accuracy")]
    [InlineData("one-hot", "recall")]
    [InlineData("label", "recall")]
    public void Handle_ReturnAnErrorMessageIfEpochsAreNotDefinedForLogisticRegression(string encoderType, string metric)
    {
        ClassificationRequest request =
            new ClassificationRequest(encoderType, "logistic-regression", metric, "car", learningRate: 0.01);
        AssertErrorResult(request, "number of epochs must be specified");
    }

    [Theory]
    [InlineData("one-hot", "accuracy", -0.01)]
    [InlineData("one-hot", "accuracy", -1)]
    [InlineData("one-hot", "accuracy", -0.1)]
    [InlineData("one-hot", "accuracy", -100)]
    [InlineData("label", "accuracy", -0.01)]
    [InlineData("label", "accuracy", -1)]
    [InlineData("label", "accuracy", -0.1)]
    [InlineData("label", "accuracy", -100)]
    [InlineData("one-hot", "recall", -0.01)]
    [InlineData("one-hot", "recall", -1)]
    [InlineData("one-hot", "recall", -0.1)]
    [InlineData("one-hot", "recall", -100)]
    [InlineData("label", "recall", -0.01)]
    [InlineData("label", "recall", -1)]
    [InlineData("label", "recall", -0.1)]
    [InlineData("label", "recall", -100)]
    public void Handle_ReturnAnErrorMessageIfLearningRateIsNegativeForLogisticRegression(string encoderType,
        string metric, double learningRate)
    {
        ClassificationRequest request = new ClassificationRequest(encoderType, "logistic-regression", metric, "car",
            learningRate: learningRate, epochs: 10);
        AssertErrorResult(request, "learning rate can not be negative");
    }

    [Theory]
    [InlineData("one-hot", "accuracy", -1)]
    [InlineData("one-hot", "accuracy", -2)]
    [InlineData("one-hot", "accuracy", -10)]
    [InlineData("one-hot", "accuracy", -100)]
    [InlineData("label", "accuracy", -1)]
    [InlineData("label", "accuracy", -2)]
    [InlineData("label", "accuracy", -10)]
    [InlineData("label", "accuracy", -100)]
    [InlineData("one-hot", "recall", -1)]
    [InlineData("one-hot", "recall", -2)]
    [InlineData("one-hot", "recall", -10)]
    [InlineData("one-hot", "recall", -100)]
    [InlineData("label", "recall", -1)]
    [InlineData("label", "recall", -2)]
    [InlineData("label", "recall", -10)]
    [InlineData("label", "recall", -100)]
    public void Handle_ReturnAnErrorMessageIfEpochsAreNegativeForLogisticRegression(string encoderTpe, string metric,
        int epochs)
    {
        ClassificationRequest request = new ClassificationRequest(encoderTpe, "logistic-regression", metric, "car",
            learningRate: 0.01, epochs: epochs);
        AssertErrorResult(request, "epochs can not be negative");
    }

    private void AssertErrorResult(ClassificationRequest request, string errorMessage)
    {
        Controller controller = new();
        var result = controller.Handle(request);
        Assert.Null(result.Score);
        Assert.Equal(errorMessage, result.ErrorMessage);
    }
}