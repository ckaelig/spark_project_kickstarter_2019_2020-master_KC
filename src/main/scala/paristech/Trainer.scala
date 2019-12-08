package paristech

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{udf, _}
import org.apache.spark.sql.types.IntegerType


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP 3 Spark : Trainer")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("hello world ! from Trainer") // << add this

    val path_to_data : String = "/mnt/d/09_SPARK/git_Flooorent/cours-spark-telecom/data/prepared_trainingset/"

    val myDataFrame = spark.read.parquet(path_to_data+"*.parquet")
    myDataFrame.printSchema()

    // Stage 1 : récupérer les mots des textes
    myDataFrame
      .select("text")
      .show
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")
    val countTokens = udf { (words: Seq[String]) => words.length }

    val regexTokenized = tokenizer.transform(myDataFrame)
    regexTokenized.select("text", "tokens")
      .withColumn("tokens", countTokens(col("tokens"))).show(false)
    regexTokenized
      .select("text","tokens")
      .show

    regexTokenized.printSchema()

    // Stage 2 : retirer les stop words
    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol) //  .setInputCol("tokens")
      .setOutputCol("filtered")

    val dfFiltered     =   remover.transform(regexTokenized) //.toDF().show(true)
    dfFiltered.select("tokens").show(10)

    // Stage 3 : computer la partie TF
    dfFiltered.printSchema()
    val countVectorizer: CountVectorizer = new CountVectorizer()
      .setInputCol(remover.getOutputCol)
      .setOutputCol("rawFeatures")
      .setMinDF(1)

    val tfModel = countVectorizer.fit(dfFiltered)
    val featurizedData = tfModel.transform(dfFiltered)
    featurizedData.select("filtered", "rawFeatures").show(20)

    // Stage 4 : computer la partie IDF
    val idf = new IDF()
      .setInputCol(countVectorizer.getOutputCol) // .setInputCol("rawFeatures")
      .setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)

    // Conversion des variables catégorielles en variables numériques
    // Stage 5 : convertir country2 en quantités numériques dans une colonne country_indexed.
    val indexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    val DFindexed = indexer.fit(rescaledData).transform(rescaledData)
    DFindexed.select("country2","country_indexed").distinct().show

    // Stage 6 : convertir currency2 en quantités numériques dans une colonne currency_indexed.
    val indexerCur = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    val DFindexedCur = indexerCur.fit(DFindexed).transform(DFindexed)
    DFindexedCur.select("currency2","currency_indexed").distinct().show

    // Stages 7 et 8: One-Hot encoder ces deux catégories avec un "one-hot encoder" en créant les colonnes country_onehot et currency_onehot.
    val oneHotEncoder = new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed", "currency_indexed"))
      .setOutputCols(Array("country_onehot", "currency_onehot"))

    val DFOne = oneHotEncoder.fit(DFindexedCur).transform(DFindexedCur)
    DFOne.select("country_onehot","currency_onehot").show

    DFOne.printSchema()

    // Mettre les données sous une forme utilisable par Spark.ML
    // Stage 9 : assembler tous les features en un unique vecteur
    val assembler = new VectorAssembler()
      .setInputCols(Array("features", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("join_features")

    val DFGroupFeat = assembler.transform(DFOne)
    DFGroupFeat.select("join_features").show

    // Stage 10 : créer/instancier le modèle de classification
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(50) //comparer les deux valeurs et commenter

    // Création du Pipeline
    val myPipeline = new Pipeline().setStages(Array(tokenizer, remover,
      countVectorizer, idf,
      indexer, indexerCur,
      oneHotEncoder, assembler, lr))

    // Entraînement, test, et sauvegarde du modèle
    // Split des données en training et test sets
    val Array(training, test) = myDataFrame.randomSplit(Array(0.9, 0.1), 98765L)

    // Entraînement du modèle
    val model1 = myPipeline.fit(training)

    //Test du modèle
    val dfWithSimplePredictions = model1.transform(test)
    dfWithSimplePredictions.groupBy("final_status", "predictions").count.show()
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    println("Modèle simple")
    val f1score1 = evaluator.evaluate(dfWithSimplePredictions)
    println("f1score du modele simple (avant grid search) sur les donnees = %.3f".format(f1score1))

    // Réglage des hyper-paramètres du modèle
    // par Grid search
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(countVectorizer.minDF, Array[Double](55, 75, 95))
      .build()

    val cross_valid = new TrainValidationSplit()
      .setEstimator(myPipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7) // 70% training, 30% validation

    val model2 = cross_valid.fit(training)
    model2.write.overwrite().save("src/main/resources/optimodel")
    val dfWithPredictions: DataFrame = model2.transform(test)

    dfWithPredictions.groupBy("final_status", "predictions").count.show()
    println("Modèle paramétrique (grid search)")
    val f1score2 = evaluator.evaluate(dfWithPredictions)
    println("f1score du modele paramétrique (avec grid search) sur les données = %.3f".format(f1score2))

  }
}
