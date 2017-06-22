/**
  * Created by michaelseeber on 6/21/17.
  * Estimator obtained from https://github.com/high-performance-spark/high-performance-spark-examples/blob/master/src/main/scala/com/high-performance-spark-examples/ml/CustomPipeline.scala
  */

import org.apache.spark.ml.{Estimator, Model, Transformer}
import org.apache.spark.ml.param.{Param, Params, ParamMap}
import org.apache.spark.ml.util.{Identifiable}
import org.apache.spark.sql.types.{StringType, StructType, StructField, IntegerType}
import org.apache.spark.sql.{Dataset, DataFrame}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.SQLContext

import scala.util.parsing.json.JSON


object mikeEstimator {

  trait SimpleIndexerParams extends Params {
    final val inputCol = new Param[String](this, "inputCol", "The input column")
    final val outputCol = new Param[String](this, "outputCol", "The output column")
  }

  class SimpleIndexer(override val uid: String)
    extends Estimator[SimpleIndexerModel] with SimpleIndexerParams {

    def setInputCol(value: String) = set(inputCol, value)

    def setOutputCol(value: String) = set(outputCol, value)

    def this() = this(Identifiable.randomUID("simpleindexer"))

    override def copy(extra: ParamMap): SimpleIndexer = {
      defaultCopy(extra)
    }

    override def transformSchema(schema: StructType): StructType = {
      // Check that the input type is a string
      val idx = schema.fieldIndex($(inputCol))
      val field = schema.fields(idx)
      if (field.dataType != StringType) {
        throw new Exception(
          s"Input type ${field.dataType} did not match input type StringType")
      }
      // Add the return field
      schema.add(StructField($(outputCol), IntegerType, false))
    }

    override def fit(dataset: Dataset[_]): SimpleIndexerModel = {
      import dataset.sparkSession.implicits._
      val words = dataset.select(dataset($(inputCol)).as[String]).distinct
        .collect()
      val model = new SimpleIndexerModel(uid, words)
      model.set(inputCol, $(inputCol))
      model.set(outputCol, $(outputCol))
      model
    }
  }

  class SimpleIndexerModel(override val uid: String, words: Array[String])
    extends Model[SimpleIndexerModel] with SimpleIndexerParams {

    override def copy(extra: ParamMap): SimpleIndexerModel = {
      defaultCopy(extra)
    }

    private val labelToIndex: Map[String, Double] = words.zipWithIndex.
      map { case (x, y) => (x, y.toDouble) }.toMap

    override def transformSchema(schema: StructType): StructType = {
      // Check that the input type is a string
      val idx = schema.fieldIndex($(inputCol))
      val field = schema.fields(idx)
      if (field.dataType != StringType) {
        throw new Exception(
          s"Input type ${field.dataType} did not match input type StringType")
      }
      // Add the return field
      schema.add(StructField($(outputCol), IntegerType, false))
    }

    override def transform(dataset: Dataset[_]): DataFrame = {
      val indexer = udf { label: String => labelToIndex(label) }
      dataset.select(col("*"),
        indexer(dataset($(inputCol)).cast(StringType)).as($(outputCol)))
    }

  }

  def main(args: Array[String]): Unit = {
    val sc = SparkMLExtension.SparkMLExtension.main(args) // reuses code to create a SparkContext
    val spark = new SQLContext(sc) // creates a SQLContext needed for DataFrames--be sure to import this
    import spark.implicits._
    val tweets = spark.read.textFile("/Users/michaelseeber/Documents/Projects/Data/Tweets/mike").toDF("value")


    val mTransformer = new mikeTransformer.mTransformer()
      .setInputCol("value")
      .setOutputCol1("tweet")
      .setOutputCol2("lang")

    val tweets2 = mTransformer.transform(tweets)
    tweets2.show()

    val simpleIndexer = new SimpleIndexer()
      .setInputCol("lang")
      .setOutputCol("lang2")

    simpleIndexer.fit(tweets2).transform(tweets2).show()

  }

}
