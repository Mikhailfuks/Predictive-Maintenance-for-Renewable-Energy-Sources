using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace PredictiveMaintenanceRenewableEnergy
{
    // Data class for wind turbine sensor data
    public class TurbineData
    {
        [LoadColumn(0)]
        public float WindSpeed { get; set; } // Wind speed (m/s)

        [LoadColumn(1)]
        public float RotorSpeed { get; set; } // Rotor speed (RPM)

        [LoadColumn(2)]
        public float GeneratorTemperature { get; set; } // Generator temperature (°C)

        [LoadColumn(3)]
        public float GearboxVibration { get; set; } // Gearbox vibration level (g)

        [LoadColumn(4)]
        public float BladePitch { get; set; } // Blade pitch angle (degrees)

        [LoadColumn(5), ColumnName("Label")]
        public bool Failure { get; set; } // True for failure, False for normal operation
    }

    // Class for predictions
    public class TurbinePrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Failure { get; set; } // True for failure, False for normal operation

        [ColumnName("Score")]
        public float Probability { get; set; } // Probability of the predicted outcome
    }

    class Program
    {
        static void Main(string[] args)
        {
            // 1. Load the data
            MLContext mlContext = new MLContext();
            string dataPath = "turbine_data.csv"; // Replace with your data file path
            IDataView dataView = mlContext.Data.LoadFromTextFile<TurbineData>(dataPath, hasHeader: true, separatorChar: ',');

            // 2. Define the training pipeline
            var pipeline = mlContext.Transforms.Concatenate("Features", "WindSpeed", "RotorSpeed", "GeneratorTemperature", "GearboxVibration", "BladePitch")
                .Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Score", "Score"));

            // 3. Train the model
            ITransformer model = pipeline.Fit(dataView);

            // 4. Create a prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<TurbineData, TurbinePrediction>(model);

            // 5. Make a prediction
            TurbineData newTurbineData = new TurbineData()
            {
                // Example sensor data
                WindSpeed = 10,
                RotorSpeed = 1500,
                GeneratorTemperature = 60,
                GearboxVibration = 0.5,
                BladePitch = 10
            };

            TurbinePrediction prediction = predictionEngine.Predict(newTurbineData);

            // 6. Display the prediction
            Console.WriteLine($"Predicted Failure: {(prediction.Failure ? "Yes" : "No")}");
            Console.WriteLine($"Probability: {prediction.Probability}");

            Console.ReadKey();
        }
    }
}
