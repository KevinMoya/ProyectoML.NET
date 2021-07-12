using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Linq;

namespace EjercicioML
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<HousingCase>("./housing.csv", hasHeader: true
                , separatorChar: ',');

            var split = context.Data.TrainTestSplit(data, testFraction: 0.2);

            var features = split.TrainSet.Schema
                .Select(col => col.Name)
                .Where(colName => colName != "Label" && colName != "ocean_proximity")
                .ToArray();

            var pipeline = context.Transforms.Text.FeaturizeText("Text", "ocean_proximity")
                .Append(context.Transforms.Concatenate("Features", features))
                .Append(context.Transforms.Concatenate("Features", "Features", "Text"))
                .Append(context.Regression.Trainers.LbfgsPoissonRegression());

            var model = pipeline.Fit(split.TrainSet);

            var predictions = model.Transform(split.TestSet);

            var metrics = context.Regression.Evaluate(predictions);

            Console.WriteLine($"Valor(R^2): {metrics.RSquared}");
        }

    }

    internal class HousingCase
    {
        [LoadColumn(0)]
        public float longitude { get; set; }

        [LoadColumn(1)]
        public float latitude { get; set; }

        [LoadColumn(2)]
        public float housing_median_age { get; set; }

        [LoadColumn(3)]
        public float total_rooms { get; set; }

        [LoadColumn(4)]
        public float total_bedrooms { get; set; }

        [LoadColumn(5)]
        public float population { get; set; }

        [LoadColumn(6)]
        public float households { get; set; }

        [LoadColumn(7)]
        public float median_income { get; set; }

        [LoadColumn(8), ColumnName("Label")]
        public float median_house_value { get; set; }

        [LoadColumn(9)]
        public string ocean_proximity { get; set; }
    }
}
