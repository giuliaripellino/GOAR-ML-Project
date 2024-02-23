import org.apache.spark.sql.{DataFrame, SparkSession}
object SparksInTheDarkMain {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("SparksInTheDark")
      // .master & .config commands are for local running
      .master("local[*]")
      .config("spark.driver.binAdress","127.0.0.1")
      .getOrCreate()

    // Read in data from parquet
    val df_background = spark.read
      .parquet("data/ntuple_em_olga.parquet")
    df_background.show()

    // WRONG SIGNAL SAMPLE
    val df_signal = spark.read
      .parquet("data/ntuple_SU2L_25_500_v2.parquet")

    // Function which filters based on pre-defined pre-selection & selects the interesting variables
    def filterAndSelect(df: DataFrame): DataFrame = {
      val filtered_df = df.filter("jet_n == 6 AND bjet_n == 4")
      val selectedColumns = filtered_df.select("deltaRLep2ndClosestBJet","LJet_m_plus_RCJet_m_12","bb_m_for_minDeltaR")
      selectedColumns
    }

    val filtered_background = filterAndSelect(df_background)
    filtered_background.show()

    // We can also see how many events we had before and after filtering:
    val events_pre_filter = df_background.count()
    val events_post_filter = filtered_background.count()
    println(s"Events before filter: ${events_pre_filter}")
    println(s"Events after filter: ${events_post_filter}")

    // Create histogram as outlined in SparkDensityTree-Introduction.scala

    // Write out csv file on the form 
    // deltaR_min_bb	<	2.7
    // delta_R_min_bb > 0.5
    // .
    // .
    // .  

    spark.stop()
  }
}
