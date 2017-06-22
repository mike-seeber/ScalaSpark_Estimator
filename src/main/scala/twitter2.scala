/**
  * Created by michaelseeber on 6/19/17.
  */

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer}
import org.apache.spark.sql.SQLContext

import scala.util.parsing.json._



object twitter2 {

  def main(args: Array[String]) {
    val sc = SparkMLExtension.SparkMLExtension.main(args) // reuses code to create a SparkContext
    val sqlcontext = new SQLContext(sc) // creates a SQLContext needed for DataFrames--be sure to import this
    import sqlcontext.implicits._// gives me the .toDF() method to turn an RDD into a DataFrame

    val tweets = sqlcontext.read.textFile("/Users/michaelseeber/Documents/Projects/Data/Tweets/mike").toDF("value")

    val mTransformer = new mikeTransformer.mTransformer()
      .setInputCol("value")
      .setOutputCol1("tweet")
      .setOutputCol2("lang")
      .setOutputCol2FilterList(List("en", "es"))
    val simpleIndexer = new mikeEstimator.SimpleIndexer()
      .setInputCol("lang")
      .setOutputCol("lang_ind")
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
      .setLabelCol("lang_ind")
      .setFeaturesCol("features")
      .setNumTrees(10)

    val pipeline = new Pipeline()
      .setStages(Array(mTransformer, simpleIndexer, tokenizer, hashing, idf, forest))

    val Array(dfTrain, dfTest) = tweets.randomSplit(Array(0.7, 0.3), 123)

    val model = pipeline.fit(dfTrain)
    val test = model.transform(dfTest)
    test.show()

    val evaluator = new BinaryClassificationEvaluator()
      .setRawPredictionCol("probability")
      .setLabelCol("lang_ind")
      .setMetricName("areaUnderROC")

    println("AUC for Random Forest: ", evaluator.evaluate(test))

  }

}
