
open System
open System.IO
open Microsoft.ML
open Microsoft.ML.Data

[<CLIMutable>]
type SampleData = {
    [<LoadColumn(0)>] X : float32
    [<LoadColumn(1)>] Label : float32
}

/// A single model prediction.
[<CLIMutable>]
type SamplePrediction = {
    Score : float32
    Label : float32
}
let dataPath = sprintf "%s\\data\\data2.csv" Environment.CurrentDirectory
let leaRate = 0.0001f
let traStep = 100

[<EntryPoint>]
let main arv =
    let mlContext = new MLContext()
    let data = mlContext.Data.LoadFromTextFile<SampleData>(dataPath, hasHeader = false, separatorChar = ',')
    
    //printfn "%d" (data.Schema["X"])

    // 15% for testing
    let partitions = mlContext.Data.TrainTestSplit(data, testFraction = 0.01)

    // set up a training pipeline
    let pipeline = 
        EstimatorChain()
            .Append(mlContext.Transforms.Concatenate("Features", "X"))
            .Append(mlContext.Regression.Trainers.OnlineGradientDescent(learningRate = leaRate, numberOfIterations = traStep)) //learningRate = leaRate, numberOfIterations = traStep

    // train the model
    let model = partitions.TrainSet |> pipeline.Fit
    
    //metrics
    let metrics = partitions.TestSet |> model.Transform |> mlContext.Regression.Evaluate
    
    //printfn "%f" metrics.LossFunction

    // set up a prediction engine
    let engine = mlContext.Model.CreatePredictionEngine<SampleData, SamplePrediction>(model)

    // sample
    let testSample = 
        { X = 12.32f
          Label = 0f
        }
    
    let prediction = testSample |> engine.Predict

    // show the prediction
    printfn "\r"
    printfn "Тест: "
    printfn "Х: 12.32F, Y:%f" prediction.Score

    0 // return value