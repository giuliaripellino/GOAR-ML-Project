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
    val df_background = spark.read
      .parquet("gs://sitd-parquet-bucket/ntuple_em_v2.parquet")

    df_background.show()
    df_background.printSchema()

    val df_signal = spark.read
      .parquet("gs://sitd-parquet-bucket/ntuple_SU2L_25_500_v2.parquet")

    df_signal.show()
    df_signal.printSchema()

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
