using System.Text.Json;

namespace Classifier;

public class Controller
{
    public readonly Dictionary<string, int> _vocabulary = new();
    public readonly List<Tuple<double[], string>> _trainingData = new();
    public readonly Dictionary<string, double[]> _weightsByLabel = new();

    public ClassificationResult Handle(ClassificationRequest request)
    {
        ClassificationResult result = new ClassificationResult();
        //Load Training Dataset
        string path = Path.Combine("data", "datasets", $"{request.Dataset}-train.json");
        if (!File.Exists(path))
        {
            result.ErrorMessage = $"Dataset file not found: {request.Dataset}-train";
            return result;
        }
        List<Sample> trainDataset = JsonSerializer.Deserialize<List<Sample>>(File.ReadAllText(path))!;
        if (trainDataset.Count == 0)
        {
            result.ErrorMessage = $"Dataset {request.Dataset}-train is empty";
            return result;
        }
        
        //Load Testing Dataset
        string path2 = Path.Combine("data", "datasets", $"{request.Dataset}-test.json");
        if (!File.Exists(path2))
        {
            result.ErrorMessage = $"Dataset file not found: {request.Dataset}-test";
            return result;
        }
        List<Sample> testDataset = JsonSerializer.Deserialize<List<Sample>>(File.ReadAllText(path2))!;
        if (testDataset.Count == 0)
        {
            result.ErrorMessage = $"Dataset {request.Dataset}-test is empty";
            return result;
        }
        
        
        // Fit the encoder with the training dataset
        if (request.EncoderType == "label")
            FitLabelEncoder(trainDataset);
        else if(request.EncoderType == "one-hot")
            FitOneHotEncoder(trainDataset);
        else
        {
            result.ErrorMessage = "Invalid encoder type";
            return result;
        }
        // We expect to add new encoders in the future
        
        // Train the classifier with the training dataset
        (bool, string?) trainResult;
        if (request.ClassifierType == "knn")
            trainResult = TrainKNNClassifier(trainDataset, request.K, request.EncoderType);
        else if (request.ClassifierType == "logistic-regression")
            trainResult = TrainLogisticRegressionClassifier(trainDataset, request.Epochs, request.LearningRate, request.EncoderType);
        else
            trainResult = (false, "Invalid classifier type");
        if (!trainResult.Item1)
        {
            result.ErrorMessage = trainResult.Item2;
            return result;
        }
        // We expect to add new classifiers in the future
        
        
        // Compute metric
        if (request.Metric == "accuracy")
        {
            int correct = 0;
            for (int i = 0 ; i < testDataset.Count; i++)
            {
                Sample sample = testDataset[i];
                (bool, string) prediction;
                // Predict label
                if (request.ClassifierType == "knn")
                    prediction = PredictKNN(sample.Features, request.EncoderType, request.K);
                else if (request.ClassifierType == "logistic-regression")
                    prediction = PredictLogisticRegression(sample.Features, request.EncoderType);
                else
                    prediction = (false, "Invalid classifier type");
                if (!prediction.Item1)
                {
                    result.ErrorMessage = prediction.Item2;
                    return result;
                }
                // We expect to add new classifiers in the future
                
                if (prediction.Item2 == sample.Label)
                    correct++;
            }

            result.Score = (double) correct/testDataset.Count;
        }
        else if (request.Metric == "recall")
        {
            // Get all distinct labels from the test dataset
            HashSet<string> labels = new();
            foreach (Sample sample in testDataset)
                labels.Add(sample.Label);
            string[] classes = labels.ToArray();

            // Compute accuracy for each class
            List<double> recalls = new();

            foreach (string label in classes)
            {
                Sample[] samples = testDataset.Where(p => p.Label == label).ToArray();
                int correct = 0;
                for (int i = 0; i < samples.Length; i++)
                {
                    Sample sample = samples[i];
                    (bool, string) prediction;
                    // Predict label
                    if (request.ClassifierType == "knn")
                        prediction = PredictKNN(sample.Features, request.EncoderType, request.K);
                    else if (request.ClassifierType == "logistic-regression")
                        prediction = PredictLogisticRegression(sample.Features, request.EncoderType);
                    else
                        prediction = (false, "Invalid classifier type");
                    if (!prediction.Item1)
                    {
                        result.ErrorMessage = prediction.Item2;
                        return result;
                    }
                    // We expect to add new classifiers in the future

                    if (sample.Label == prediction.Item2)
                        correct++;
                }

                recalls.Add((double)correct / samples.Length);
            }

            result.Score = recalls.Average();
        }
        else
            result.ErrorMessage = "Invalid metric";
        // We expect to add more metrics in the future

        return result;
    }
    
    private void FitLabelEncoder(List<Sample> trainingDataset)
    {
        _vocabulary.Clear();
        
        // Get all distinct features from the dataset
        HashSet<string> features = new();
        foreach (Sample sample in trainingDataset)
        foreach (string feature in sample.Features)
            features.Add(feature);
        string[] allTokens = features.ToArray();
        
        // Build vocabulary mapping
        int index = 0;
        foreach (string token in allTokens)
            _vocabulary[token] = index++;
    }

    private double[] EncodeLabel(string[] sample)
    {
        double[] vec = new double[sample.Length];
        for (int i = 0; i < sample.Length; i++)
        {
            string token = sample[i];
            if (_vocabulary.ContainsKey(token))
            {
                int idx = _vocabulary[token];
                vec[i] = idx;
            }
        }
        return vec;
    }
    
    private void FitOneHotEncoder(List<Sample> trainingDataset)
    {
        _vocabulary.Clear();
        
        // Get all distinct features from the dataset
        HashSet<string> features = new();
        foreach (Sample sample in trainingDataset)
        foreach (string feature in sample.Features)
            features.Add(feature);
        string[] allTokens = features.ToArray();
        
        // Build vocabulary mapping
        int index = 0;
        foreach (string token in allTokens)
            _vocabulary[token] = index++;
    }

    private double[] EncodeOneHot(string[] sample)
    {
        double[] vec = new double[_vocabulary.Count];
        foreach (string token in sample)
        {
            if (_vocabulary.ContainsKey(token))
            {
                int idx = _vocabulary[token];
                vec[idx] = 1.0;
            }
        }
        return vec;
    }
    
    private (bool, string?) TrainKNNClassifier(List<Sample> trainingDataset, int? k, string encoderType)
    {
        // Validate neighbor count (hyperparameter k)
        if (k == null)
            return (false, "number of neighbors must be specified");
        if (k <= 0)
            return (false, "number of neighbors must be positive");
        if (trainingDataset.Count < k)
            return (false, "number of neighbors must be smaller than the number of training samples");
        
        
        foreach (Sample sample in trainingDataset)
        {
            double[] encodedFeatures;
            if(encoderType == "one-hot")
                encodedFeatures = EncodeOneHot(sample.Features);
            else if (encoderType == "label")
                encodedFeatures = EncodeLabel(sample.Features);
            else
                return (false, "Invalid encoder type");
            // We expect to add new encoders in the future
            
            _trainingData.Add(new Tuple<double[], string>(encodedFeatures, sample.Label));
        }

        return (true, null);
    }

    /*
     * This is legacy code. It turns out that the implementation using LINQ is much faster.
     */
    // private (bool, string) PredictKNN(string[] features, int? k, string encoderType)
    // {
    //     double[] encodedFeatures;
    //     if(encoderType == "one-hot")
    //         encodedFeatures = EncodeOneHot(features);
    //     else if (encoderType == "label")
    //         encodedFeatures = EncodeLabel(features);
    //     else
    //         return (false, "Invalid encoder type");
    //     // We expect to add new encoders in the future
    //
    //     // Calculate distances for each training sample
    //     int n = _trainingData.Count;
    //     double[] distances = new double[n];
    //     string[] labels = new string[n];
    //     for (int i = 0; i < n; i++)
    //     {
    //         // Calculate Euclidean distance
    //         double sum = 0;
    //         for (int j = 0; j < encodedFeatures.Length; j++)
    //             sum += Math.Pow(encodedFeatures[j] - _trainingData[i].Item1[j], 2);
    //         distances[i] = Math.Sqrt(sum);
    //         
    //         // Store corresponding label
    //         labels[i] = _trainingData[i].Item2;
    //     }
    //
    //     // Sort distances and labels
    //     for (int i = 0; i < n - 1; i++)
    //     {
    //         for (int j = 0; j < n - i - 1; j++)
    //         {
    //             if (distances[j] > distances[j + 1])
    //             {
    //                 (distances[j], distances[j + 1]) = (distances[j + 1], distances[j]);
    //                 (labels[j], labels[j + 1]) = (labels[j + 1], labels[j]);
    //             }
    //         }
    //     }
    //
    //     // Take first K neighbors
    //     Dictionary<string, int> labelCounts = new Dictionary<string, int>();
    //     for (int i = 0; i < k && i < n; i++)
    //     {
    //         string label = labels[i];
    //         if (labelCounts.ContainsKey(label))
    //             labelCounts[label]++;
    //         else
    //             labelCounts[label] = 1;
    //     }
    //
    //     // Choose label with the highest count
    //     // If counts tie, pick the one that appears first in the top _k sequence
    //     string bestLabel = null;
    //     int bestCount = -1;
    //     for (int i = 0; i < k && i < n; i++)
    //     {
    //         string label = labels[i];
    //         int count = labelCounts[label];
    //         if (count > bestCount)
    //         {
    //             bestCount = count;
    //             bestLabel = label;
    //         }
    //     }
    //
    //     return (true, bestLabel!);
    // }
    
    public (bool, string) PredictKNN(string[] features, string encoderType, int? k)
    {
        double[] encodedFeatures;
        if(encoderType == "one-hot")
            encodedFeatures = EncodeOneHot(features);
        else if (encoderType == "label")
            encodedFeatures = EncodeLabel(features);
        else
            return (false, "Invalid encoder type");
        // We expect to add new encoders in the future
        
        
        /*
         The calculation of distances and the finding of the K nearest neighbors is implemented using LINQ
         because otherwise the algorithm would take a long time to execute. The helper method Euclides is uses so that
         the code becomes a little easier to read
        */
        var notOrderedDistances =
            _trainingData.Select(t => new { Distance = Euclides(encodedFeatures, t.Item1), Label = t.Item2 });
        var orderedDistances = notOrderedDistances.OrderBy(t => t.Distance);
        var nearedDistances = orderedDistances.Take((int)k!);

        return (true, nearedDistances.GroupBy(t => t.Label).OrderByDescending(g => g.Count()).First().Key);
    }

    public double Euclides(double[] features1, double[] features2)
    {
        double sum = 0;
        for (int i = 0; i < features1.Length; i++)
            sum += Math.Pow(features1[i] - features2[i], 2);
        return Math.Sqrt(sum);
    }

    private (bool, string?) TrainLogisticRegressionClassifier(List<Sample> trainingDataset, int? epochs, double? learningRate, string encoderType)
    {
        //Validate hyperparameters
        if (learningRate == null)
            return (false, "learning rate must be specified");
        if (learningRate < 0)
            return (false, "learning rate can not be negative");
        if (epochs == null)
            return (false, "number of epochs must be specified");
        if (epochs < 0)
            return (false, "epochs can not be negative");
        
        // The actual training logic would go here
        
        // Get distinct labels from dataset
        HashSet<string> hashSet = new();
        foreach (Sample sample in trainingDataset)
            hashSet.Add(sample.Label);
        string[] labels = hashSet.ToArray();
        
        // Encode first sample to determine feature count
        double[] encodedFeatures;
        if(encoderType == "one-hot")
            encodedFeatures = EncodeOneHot(trainingDataset[0].Features);
        else if (encoderType == "label")
            encodedFeatures = EncodeLabel(trainingDataset[0].Features);
        else
            return (false, "Invalid encoder type");
        // We expect to add new encoders in the future
        int featureCount = encodedFeatures.Length;
        
        // Initialize weights
        _weightsByLabel.Clear();
        foreach (string label in labels)
            _weightsByLabel[label] = new double[featureCount + 1];
        
        // Train using One-vs-Rest strategy
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int j = 0; j < trainingDataset.Count; j++)
            {
                Sample sample = trainingDataset[j];
                double[] encoded;
                if(encoderType == "one-hot")
                    encoded = EncodeOneHot(sample.Features);
                else if (encoderType == "label")
                    encoded = EncodeLabel(sample.Features);
                else
                    return (false, "Invalid encoder type");
                
                
                double[] inputVector = BuildInputVector(encoded);
                
                // Update weights for each label in sample
                foreach (string label in labels)
                {
                    int target = sample.Label == label ? 1 : 0;
                    double prediction = PredictWithSigmoid(_weightsByLabel[label], inputVector);
                    
                    // Update Weights and bias
                    for (int i = 0; i < _weightsByLabel[label].Length; i++)
                        _weightsByLabel[label][i] += (double)learningRate * (target - prediction) * inputVector[i];
                }
            }
        }
        return (true, null);
    }

    public double[] BuildInputVector(double[] encoded)
    {
        // Build input vector with bias
        double[] inputVectorWithBias = new double[encoded.Length + 1];
        inputVectorWithBias[0] = 1; // Add bias
        for (int i = 0; i < encoded.Length; i++)
            inputVectorWithBias[i + 1] = encoded[i];
        return inputVectorWithBias;
    }

    public double PredictWithSigmoid(double[] weights, double[] inputVector)
    {
        //Compute Dot Product
        double dotProduct = 0;
        for (int i = 0; i < weights.Length; i++)
            dotProduct += weights[i] * inputVector[i];
                    
        // Compute Prediction with Sigmoid Function
        return 1.0 / (1.0 + Math.Exp(-dotProduct));
    }

    private (bool, string) PredictLogisticRegression(string[] features, string encoderType)
    {
        double[] encoded;
        if(encoderType == "one-hot")
            encoded = EncodeOneHot(features);
        else if (encoderType == "label")
            encoded = EncodeLabel(features);
        else
            return (false, "Invalid encoder type");
        // We expect to add new encoders in the future
        
        var inputVector = BuildInputVector(encoded);
        // Predict the label with the highest score
        string bestLabel = null!;
        double bestScore = double.NegativeInfinity;

        foreach (var kvp in _weightsByLabel)
        {
            double score = PredictWithSigmoid(kvp.Value, inputVector);
            
            // Select Best Label
            if (score > bestScore)
            {
                bestScore = score;
                bestLabel = kvp.Key;
            }
        }

        return (true, bestLabel);
    }
    
}