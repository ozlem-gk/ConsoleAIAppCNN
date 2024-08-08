using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using OpenCvSharp;
using Tensorflow;
using Tensorflow.NumPy;
using Tensorflow.Keras.Models;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Utils;
using Tensorflow.Keras.Callbacks;
using static Tensorflow.Binding;
using np = Tensorflow.NumPy.np;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.ArgsDefinition;

namespace ConsoleAIApp
{
    public static class Program
    {
        static string[] labels = { "PNEUMONIA", "NORMAL" };
        static int img_size = 150;

        static void Main(string[] args)
        {
            // Veriyi yükleyip işleyelim
            var train_data = GetTrainingData("C:/Users/ozlem/Downloads/ChestXRay2017/chest_xray/train");
            var test_data = GetTrainingData("C:/Users/ozlem/Downloads/ChestXRay2017/chest_xray/test");
            var val_data = GetTrainingData("C:/Users/ozlem/Downloads/ChestXRay2017/chest_xray/val");

            var (x_train, y_train) = PrepareData(train_data);
            var (x_test, y_test) = PrepareData(test_data);
            var (x_val, y_val) = PrepareData(val_data);

            // Modeli oluşturma
            var model = BuildModel();

            // Modeli derleme
            model.compile(optimizer: "rmsprop",
                          loss: "binary_crossentropy",
                          metrics: new[] { "accuracy" });

            // Modeli eğitme
            var history = model.fit(x_train, y_train, batch_size: 32, epochs: 12, validation_data: (x_val, y_val));

            // Modeli değerlendirme
            var score = model.evaluate(x_test, y_test).ToString();
            Console.WriteLine($"Test loss: {score[0]}");
            Console.WriteLine($"Test accuracy: {(score[1] * 100)}%");

            // Sınıflandırma raporu ve karışıklık matrisi
            GenerateClassificationReport(x_test, y_test);
        }

        static List<(Mat, int)> GetTrainingData(string data_dir)
        {
            var data = new List<(Mat, int)>();

            foreach (var label in labels)
            {
                var path = Path.Combine(data_dir, label);
                var class_num = Array.IndexOf(labels, label);
                foreach (var imgPath in Directory.GetFiles(path))
                {
                    try
                    {
                        var img = Cv2.ImRead(imgPath, ImreadModes.Grayscale);
                        var resizedArr = new Mat();
                        Cv2.Resize(img, resizedArr, new Size(img_size, img_size));
                        data.Add((resizedArr, class_num));
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine(e.Message);
                    }
                }
            }
            return data;
        }

        static (NDArray, NDArray) PrepareData(List<(Mat, int)> data)
        {
            var images = new List<NDArray>();
            var labels = new List<int>();

            foreach (var (mat, label) in data)
            {
                var ndArray = mat.ToNDArray();
                ndArray = ndArray / 255.0f; // Normalize
                images.Add(ndArray);
                labels.Add(label);
            }

            // Convert List<NDArray> to NDArray
            var x_data = np.array(images.ToArray());

            // Reshape to (batch_size, img_size, img_size, 1)
            x_data = x_data.reshape(new Shape(-1, img_size, img_size, 1)); // Correct usage

            var y_data = np.array(labels.ToArray());

            return (x_data, y_data);
        }


     
        static Sequential BuildModel()
        {
            var model = new Sequential(new SequentialArgs());
            model.add(new Conv2D(new Conv2DArgs()
            {
                Filters = 32,
                KernelSize = (3, 3),
                Activation = tf.keras.activations.Relu,
                Padding = "same",
                InputShape = new Shape(img_size, img_size, 1)
            }));
            model.add(new BatchNormalization(new BatchNormalizationArgs()));
            model.add(new MaxPooling2D(new MaxPooling2DArgs()
            {
                PoolSize = (2, 2),
                Padding = "same"
            }));
            model.add(new Conv2D(new Conv2DArgs()
            {
                Filters = 64,
                KernelSize = (3, 3),
                Activation = tf.keras.activations.Relu,
                Padding = "same"
            }));
            model.add(new Dropout(new DropoutArgs { Rate = 0.1f }));
            model.add(new BatchNormalization(new BatchNormalizationArgs()));
            model.add(new MaxPooling2D(new MaxPooling2DArgs()
            {
                PoolSize = (2, 2),
                Padding = "same"
            }));
            model.add(new Conv2D(new Conv2DArgs()
            {
                Filters = 64,
                KernelSize = (3, 3),
                Activation = tf.keras.activations.Relu,
                Padding = "same"
            }));
            model.add(new BatchNormalization(new BatchNormalizationArgs()));
            model.add(new MaxPooling2D(new MaxPooling2DArgs()
            {
                PoolSize = (2, 2),
                Padding = "same"
            }));
            model.add(new Conv2D(new Conv2DArgs()
            {
                Filters = 128,
                KernelSize = (3, 3),
                Activation = tf.keras.activations.Relu,
                Padding = "same"
            }));
            model.add(new Dropout(new DropoutArgs { Rate = 0.2f }));
            model.add(new BatchNormalization(new BatchNormalizationArgs()));
            model.add(new MaxPooling2D(new MaxPooling2DArgs()
            {
                PoolSize = (2, 2),
                Padding = "same"
            }));
            model.add(new Conv2D(new Conv2DArgs()
            {
                Filters = 256,
                KernelSize = (3, 3),
                Activation = tf.keras.activations.Relu,
                Padding = "same"
            }));
            model.add(new Dropout(new DropoutArgs { Rate = 0.2f }));
            model.add(new BatchNormalization(new BatchNormalizationArgs()));
            model.add(new MaxPooling2D(new MaxPooling2DArgs()
            {
                PoolSize = (2, 2),
                Padding = "same"
            }));
            model.add(new Flatten( new FlattenArgs()));
            model.add(new Dense(new DenseArgs()
            {
                Units = 128,
                Activation = tf.keras.activations.Relu
            }));
            model.add(new Dropout(new DropoutArgs { Rate = 0.2f }));
            model.add(new Dense(new DenseArgs()
            {
                Units = 1,
                Activation = tf.keras.activations.Sigmoid
            }));
            return model;
        }
        static void GenerateClassificationReport(NDArray predictions, NDArray y_test)
        {
            // Convert NDArray to int arrays for easier processing
            var yTrue = y_test.ToArray<int>();
            var yPred = predictions.ToArray<int>();

            // Calculate confusion matrix
            var confusionMatrix = GetConfusionMatrix(yTrue, yPred);

            // Print confusion matrix
            Console.WriteLine("Confusion Matrix:");
            Console.WriteLine($"TN: {confusionMatrix.Item1}, FP: {confusionMatrix.Item2}");
            Console.WriteLine($"FN: {confusionMatrix.Item3}, TP: {confusionMatrix.Item4}");

            // Calculate and print precision, recall, and F1-score
            var (precision, recall, f1Score) = CalculateMetrics(confusionMatrix);
            Console.WriteLine($"Precision: {precision:0.##}");
            Console.WriteLine($"Recall: {recall:0.##}");
            Console.WriteLine($"F1-Score: {f1Score:0.##}");
        }

        static (int, int, int, int) GetConfusionMatrix(int[] yTrue, int[] yPred)
        {
            int tp = 0, tn = 0, fp = 0, fn = 0;
            for (int i = 0; i < yTrue.Length; i++)
            {
                if (yTrue[i] == 1 && yPred[i] == 1) tp++;
                if (yTrue[i] == 0 && yPred[i] == 0) tn++;
                if (yTrue[i] == 0 && yPred[i] == 1) fp++;
                if (yTrue[i] == 1 && yPred[i] == 0) fn++;
            }
            return (tn, fp, fn, tp);
        }

        static (double precision, double recall, double f1Score) CalculateMetrics((int tn, int fp, int fn, int tp) confusionMatrix)
        {
            double precision = (double)confusionMatrix.tp / (confusionMatrix.tp + confusionMatrix.fp);
            double recall = (double)confusionMatrix.tp / (confusionMatrix.tp + confusionMatrix.fn);
            double f1Score = 2 * (precision * recall) / (precision + recall);

            return (precision, recall, f1Score);
        }
        public static NDArray ToNDArray(this Mat mat)
        {
            var height = mat.Rows;
            var width = mat.Cols;
            var channels = 1; // For grayscale images

            // Convert Mat to byte array
            var byteArray = mat.ToBytes();

            // Create NDArray from byte array
            var ndArray = np.array(byteArray, dtype: np.float32);

            // Reshape to (height, width, channels)
            ndArray = ndArray.reshape( new Shape(height, width, channels));

            return ndArray;
        }
    }
}