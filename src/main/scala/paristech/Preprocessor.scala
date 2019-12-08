package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions.{lower, udf, when, _}
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{DataFrame, SparkSession}


object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP 2 Spark : Preprocessor")
      .getOrCreate()

    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("\n")
    println("Hello World ! from Preprocessor")
    println("\n")
    import spark.implicits._ // << add this

    val df: DataFrame = spark
      .read
      .option("header", true)        // utilise la première ligne du (des) fichier(s) comme
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int,
      .csv("/mnt/d/09_SPARK/cours-spark-telecom-master/data/train_clean.csv")
    println(s"Nombre de lignes : ${df.count}")
    println(s"Nombre de colonnes : ${df.columns.length}")
    df.show()
    df.printSchema()

    val dfCasted: DataFrame = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline" , $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))

    dfCasted.printSchema()

    dfCasted
      .select("goal", "backers_count", "final_status")
      .describe()
      .show

    dfCasted.groupBy("disable_communication").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("country").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("currency").count.orderBy($"count".desc).show(100)
    dfCasted.select("deadline").dropDuplicates.show()
    dfCasted.groupBy("state_changed_at").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("backers_count").count.orderBy($"count".desc).show(100)
    dfCasted.select("goal", "final_status").show(30)
    dfCasted.groupBy("country", "currency").count.orderBy($"count".desc).show(50)

    dfCasted.filter($"country" === "False")
      .groupBy("currency")
      .count
      .orderBy($"count".desc)
      .show(50)

    val df2: DataFrame = dfCasted.drop("disable_communication") // on enleve une colonne majoritairement false
    val dfNoFutur: DataFrame = df2.drop("backers_count", "state_changed_at") // on retire les colonnes backers_count et state_changed_at

    // creation de deux UDF :
    def cleanCountry(country: String, currency: String): String = {
      if (country == "False")
        currency
      else if (country.length != 2)
        null
      else
        country
    }
    def cleanCurrency(currency: String): String = {
      if (currency != null && currency.length != 3)
        null
      else
        currency
    }

    val cleanCountryUdf = udf(cleanCountry _)
    val cleanCurrencyUdf = udf(cleanCurrency _)

/**
    // application des UDF functions
    val dfCountry: DataFrame = dfNoFutur
      .withColumn("country2", cleanCountryUdf($"country", $"currency"))
      .withColumn("currency2", cleanCurrencyUdf($"currency"))
      .drop("country", "currency")

    // ou encore, en utilisant sql.functions.when:
  */
    val dfCountry: DataFrame = dfNoFutur
      .withColumn("country2", when($"country" === "False", $"currency").otherwise($"country"))
      .withColumn("currency2", when($"country".isNotNull && length($"currency") =!= 3, null).otherwise($"currency"))
      .drop("country", "currency")


    val dfCountry2 = dfCountry   // On ajoute une colonne days_campaign qui représente la durée de la campagne en jours (le nombre de jours entre launched_at et deadline).
      .withColumn("days_campaign", round( ( dfCountry.col("deadline")-dfCountry.col("launched_at") ) /86400 *1000)/1000 )

    dfCountry2
      .select("deadline", "launched_at", "days_campaign")
      .describe()
      .show

    // On ajoute une colonne hours_prepa qui représente
    // le nombre d’heures de préparation de la campagne
    // entre created_at et launched_at. On arrondit le résultat à 3 chiffres après la virgule.
    val dfCountry3 = dfCountry2
      .withColumn("hours_prepa", round((dfCountry2.col("launched_at")-dfCountry2.col("created_at"))/3600*1000)/1000)

    dfCountry3
      .select("created_at","deadline","launched_at","days_campaign","hours_prepa")
      .show

    // On supprime les colonnes launched_at, created_at, et deadline, elles ne sont pas exploitables pour un modèle.
    val dfCountry4: DataFrame = dfCountry3
      .drop("launched_at", "created_at", "deadline")

    print(dfCountry4.columns.toList)
    dfCountry4.show
    dfCountry4.printSchema()

    // On met les colonnes name, desc, et keywords en minuscules,
    // et on ajoute une colonne text, qui contient la concaténation
    // des Strings des colonnes name, desc, et keywords,
    // avec un espace entre les chaînes de caractères concaténées
    val dfCountry5: DataFrame = dfCountry4
      .withColumn("name", lower($"name"))
      .withColumn("desc", lower($"desc"))
      .withColumn("keywords", lower($"keywords"))

    dfCountry5.select("name","desc","keywords").show()
    print(dfCountry5.columns.toList)

    val colsToConcat = Seq(col("name"), col("desc"), col("keywords"))

    val dfCountry6: DataFrame = dfCountry5
      .withColumn("text",concat_ws(" ", colsToConcat :_*))
    dfCountry6.select("text").show()
    print(dfCountry6.columns.toList)

    // On remplace les valeurs nulles des colonnes
    // days_campaign, hours_prepa, et goal par la valeur -1
    // et par "unknown" pour les colonnes country2 et currency2
    val dfCountry7 = dfCountry6
                              .na.fill("-1" , Seq("days_campaign"))
                              .na.fill("-1" , Seq("hours_prepa"))
                              .na.fill("-1" , Seq("goal"))
                              .na.fill("Unknown", Seq("country2"))
                              .na.fill("Unknown", Seq("currency2"))

    // dfCountry7.select("days_campaign", "hours_prepa", "goal", "country2", "currency2").show()

    // print(dfCountry7.columns.toList)

    /*


    val dfCountry8: DataFrame = dfCountry7
      .withColumn("country2", when($"country2".isNull, "unknown").otherwise($"country2"))
      .withColumn("currency2", when($"currency2".isNull, "unknown").otherwise($"currency2"))

    dfCountry8
      .select("days_campaign", "hours_prepa", "goal", "country2", "currency2")
      .describe()
      .show(100)

    dfCountry8.printSchema()

      */

    /**
    object ColumnExt {

      implicit class ColumnMethods(col: Column) {
        def changeUnknown: Column = {
          when(col.isNull, "unknown").otherwise(col)
        }

        dfCountry7
          .withColumn("country2", col("country2").changeUnknown)
          .withColumn("currency2", col("currency2").changeUnknown)
          .show()
      }

    }
    dfCountry7
      .select("days_campaign", "hours_prepa", "goal", "country2", "currency2")
      .describe()
      .show(100)

    dfCountry7.select("days_campaign", "hours_prepa", "goal", "country2", "currency2").show()
      */


    val monDataFrameFinal = dfCountry7
    monDataFrameFinal.show()

    monDataFrameFinal.write.mode("overwrite").parquet("/mnt/d/09_SPARK/monDataFrameFinal")
  }
}
