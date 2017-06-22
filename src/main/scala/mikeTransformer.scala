/**
  * Created by michaelseeber on 6/19/17.
  * Transformer based on https://github.com/high-performance-spark/high-performance-spark-examples/blob/master/src/main/scala/com/high-performance-spark-examples/ml/CustomPipeline.scala
  */

import org.apache.spark.ml.{Transformer}
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{Identifiable}
import org.apache.spark.sql.types.{StringType, StructType, StructField, IntegerType}
import org.apache.spark.sql.{Dataset, DataFrame}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.SQLContext

import scala.util.parsing.json.JSON


object mikeTransformer {


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

  def getTweet(input: String): String = getTweetsAndLang(input)._1
  def getLang(input: String): String = findVal(input, "lang")
  val getTweetUDF = udf{x: String => getTweet(x)}
  val getLangUDF = udf{x: String => getLang(x)}


  class mTransformer(override val uid: String) extends Transformer {
    final val inputCol = new Param[String](this, "inputCol", "The input column")
    final val outputCol1 = new Param[String](this, "outputCol1", "The first output column")
    final val outputCol2 = new Param[String](this, "outputCol2", "The second output column")
    final val outputCol2FilterList = new Param[List[String]](this, "outputCol2FilterList", "List to filter on outputCol2")

    def setInputCol(value: String): this.type = set(inputCol, value)
    def setOutputCol1(value: String): this.type = set(outputCol1, value)
    def setOutputCol2(value: String): this.type = set(outputCol2, value)
    def setOutputCol2FilterList(value: List[String]): this.type = set(outputCol2FilterList, value)

    def this() = this(Identifiable.randomUID("mTransformer"))

    def copy(extra: ParamMap): mTransformer = {
      defaultCopy(extra)
    }

    override def transformSchema(schema: StructType): StructType = {
      // Check that the input type is a string
      val idx = schema.fieldIndex($(inputCol))
      val field = schema.fields(idx)
      if (field.dataType != StringType) {
        throw new Exception(s"Input type ${field.dataType} did not match input type StringType")
      }
      // Add the return field
      schema.add(StructField($(outputCol1), StringType, false))
          .add(StructField($(outputCol2), StringType, false))
    }

    def transform(df: Dataset[_]): DataFrame = {
      val df2 = df.withColumn($(outputCol1), getTweetUDF(df.col($(inputCol))))
        .withColumn($(outputCol2), getLangUDF(df.col($(inputCol))))
      df2.filter(df2.col($(outputCol2)).isNotNull)
        .filter(df2.col($(outputCol2)) isin ($(outputCol2FilterList): _*))
    }
  }


    def main(args: Array[String]): Unit = {
      val sc = SparkMLExtension.SparkMLExtension.main(args) // reuses code to create a SparkContext
      val spark = new SQLContext(sc) // creates a SQLContext needed for DataFrames--be sure to import this
      import spark.implicits._
      val tweets = spark.read.textFile("/Users/michaelseeber/Documents/Projects/Data/Tweets/mike").toDF("value")


      val mTransformer = new mTransformer()
        .setInputCol("value")
        .setOutputCol1("tweet")
        .setOutputCol2("lang")
        .setOutputCol2FilterList(List("en", "es"))
      mTransformer.transform(tweets).show()
      val test = mTransformer.transform(tweets)
    }



}

