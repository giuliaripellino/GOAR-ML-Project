import org.apache.spark.sql.SparkSession
object SparksInTheDarkMain {
  def main(args: Array[String]): Unit = {
    println("Starting SparkSession...")
    val spark = SparkSession.builder()
      .appName("SparksInTheDark")
      .master("local[*]")
      .config("spark.driver.binAdress","127.0.0.1")
      .getOrCreate()

    // Read in data from parquet

    // Create histogram as outlined in SparkDensityTree-Introduction.scala

    // Write out csv file on the form 
    // deltaR_min_bb	<	2.7
    // delta_R_min_bb > 0.5
    // .
    // .
    // .  

    spark.stop()
    println("Stopping SparkSession...")

  }
}
