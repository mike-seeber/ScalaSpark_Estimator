package SparkMLExtension

/**
  * Created by michaelseeber on 6/19/17.
  */

import org.apache.spark.{SparkConf, SparkContext}

object SparkMLExtension {

  def main(args: Array[String]) = {
    val conf = new SparkConf()
      .setAppName("my ml app")
      .setMaster("local[*]")
      .set("spark.driver.host", "localhost")
    val sc = new SparkContext(conf)

//    val adding = sc.parallelize(1 to 1000).sum
//    println("Success!  Returned: " + adding)
    sc
  }

}
