/**
  * Created by michaelseeber on 6/19/17.
  */

import org.apache.spark.sql.SQLContext
import scala.util.parsing.json._

import org.apache.spark.ml.feature.{RegexTokenizer, HashingTF, IDF}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator



object twitter {

  def findVal(str: String, ToFind: String): String = {
    try {
      JSON.parseFull(str) match {
        case Some(m: Map[String, String]) => m(ToFind)
      }
    } catch {
      case e: Exception => null
    }
  }

  def getTweetsAndLang(input: String): (String, Int) = {
    try {
      var result = (findVal(input, "text"), -1)

      if (findVal(input, "lang") == "en") result.copy(_2 = 0)
      else if (findVal(input, "lang") == "es") result.copy(_2 = 1)
      else result
    } catch {
      case e: Exception => ("unknown", -1)
    }
  }

  def main(args: Array[String]) {
    val sc = SparkMLExtension.SparkMLExtension.main(args) // reuses code to create a SparkContext
    val sqlcontext = new SQLContext(sc) // creates a SQLContext needed for DataFrames--be sure to import this
    import sqlcontext.implicits._// gives me the .toDF() method to turn an RDD into a DataFrame

    val tweets = sqlcontext.read.textFile("/Users/michaelseeber/Documents/Projects/Data/Tweets/mike")
    val df = tweets.map(getTweetsAndLang).filter(x=> x._2 != -1).toDF("tweet", "lang")

    val tokenizer = new RegexTokenizer()
      .setInputCol("tweet")
      .setOutputCol("words")
      .setPattern("\\s+|[,.\"]")
    val hashing = new HashingTF()
      .setInputCol("words")
      .setOutputCol("rawFeatures")
      .setNumFeatures(200)
    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")
    val forest = new RandomForestClassifier()
      .setLabelCol("lang")
      .setFeaturesCol("features")
      .setNumTrees(10)

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashing, idf, forest))

    val Array(dfTrain, dfTest) = df.randomSplit(Array(0.7, 0.3), 123)

    val model = pipeline.fit(dfTrain)
    val test = model.transform(dfTest)

    val evaluator = new BinaryClassificationEvaluator()
      .setRawPredictionCol("probability")
      .setLabelCol("lang")
      .setMetricName("areaUnderROC")

    println("AUC for Random Forest: ", evaluator.evaluate(test))
  }

}