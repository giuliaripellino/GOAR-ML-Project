import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.random.RandomRDDs.normalVectorRDD
import org.apache.spark.mllib.linalg.{Vectors, Vector => MLVector, _}
import org.apache.spark.sql.{DataFrame, Row, SQLContext, SQLImplicits, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
object SparksInTheDarkMain {
  def main(args: Array[String]): Unit = {
    println("Starting SparkSession...")
    val spark = SparkSession.builder()
      .appName("SparksInTheDark")
      // .master & .config commands are for local running
      .master("local[*]")
      .config("spark.driver.binAdress","127.0.0.1")
      .getOrCreate()

    // Read in data from parquet
    val df_background = spark.read
      //.parquet("gs://sitd-parquet-bucket/ntuple_em_v2.parquet")
      .parquet("data/ntuple_em_v2.parquet")
    df_background.show()

    // WRONG SIGNAL SAMPLE
    val df_signal = spark.read
      //.parquet("gs://sitd-parquet-bucket/ntuple_SU2L_25_500_v2.parquet")
      .parquet("data/ntuple_SU2L_25_500_v2.parquet")
    df_signal.show()

    // Function which filters based on pre-defined pre-selection & selects the interesting variables
    def filterAndSelect(df: DataFrame): DataFrame = {
      val filtered_df = df.filter("jet_n == 6 AND bjet_n == 4")
      val selectedColumns = filtered_df.select("deltaRLep2ndClosestBJet","LJet_m_plus_RCJet_m_12","bb_m_for_minDeltaR")
      selectedColumns
    }

    val filtered_background = filterAndSelect(df_background)
    filtered_background.show()

    val filtered_signal = filterAndSelect(df_signal)
    filtered_signal.show()

    // We can also see how many events we had before and after filtering:
    val original_bkg_count = df_background.count()
    val filtered_bkg_count = filtered_background.count()
    val original_signal_count = df_signal.count()
    val filtered_signal_count = filtered_signal.count()

    println(s"# Background events before filter: ${original_bkg_count}")
    println(s"# Background events after filter: ${filtered_bkg_count}")
    println(s"# Signal events before filter: ${original_signal_count}")
    println(s"# Signal events after filter: ${filtered_signal_count}")
    val trainSize : Long = math.pow(10, 3).toLong
    val trainingRDD : RDD[MLVector] = normalVectorRDD(spark.sparkContext, trainSize, 4, 3, 1230568)
    trainingRDD.take(10).foreach(println)

    // Turn spark dataframes into RDD
    def df_to_RDD(df: DataFrame): org.apache.spark.rdd.RDD[org.apache.spark.mllib.linalg.Vector]  = {
      df.rdd.map {row =>
        val deltaRLep2ndClosestBJet = row.getAs[Float]("deltaRLep2ndClosestBJet").toDouble
        val lJet_m_plus_RCJet_m_12 = row.getAs[Float]("LJet_m_plus_RCJet_m_12").toDouble
        val bb_m_for_minDeltaR = row.getAs[Float]("bb_m_for_minDeltaR").toDouble
        Vectors.dense(deltaRLep2ndClosestBJet, lJet_m_plus_RCJet_m_12, bb_m_for_minDeltaR)
        }
      }
    val backgroundRDD = df_to_RDD(filtered_background)

    println("Showing the first 10 RDD vectors")
    backgroundRDD.take(10).foreach(println)

    // Turn RDD into Minimum Density Estimate Histograms (mdeHists)


    // Plot mdeHists

    // Get 10% highest density regions

    // Plot 10% highest density regions


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
