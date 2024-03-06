import co.wiklund.disthist.SpatialTreeFunctions.widestSideTreeRootedAt
import co.wiklund.disthist.Types.Count
import co.wiklund.disthist.NodeLabel
import co.wiklund.disthist._
import co.wiklund.disthist.MergeEstimatorFunctions._
import co.wiklund.disthist.HistogramFunctions._
import co.wiklund.disthist.MDEFunctions._

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.random.RandomRDDs.normalVectorRDD
import org.apache.spark.mllib.linalg.{Vector => MLVector, _}
import org.apache.spark.sql.{DataFrame, SparkSession,SQLContext,SQLImplicits, Row}
import org.apache.spark.{SparkContext,SparkConf}
object SparksInTheDarkMain {
  def main(args: Array[String]): Unit = {
    println("Starting SparkSession...")
    val spark = SparkSession.builder()
      .appName("SparksInTheDark")
      // .master & .config commands are for local running
      .master("local[*]")
      .config("spark.driver.binAdress","127.0.0.1")
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

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

    // Create histogram as outlined in SparkDensityTree-Introduction.scala

    val dimensions : Int = df_signal.columns.length // The amount of variables we have
    val len_data : Int = 6
    val numPartitions : Int = 32
    val trainSize : Long = math.pow(10, len_data).toLong
    println(trainSize)

    // Using the randomized values
    val old_trainingRDD : RDD[MLVector] = normalVectorRDD(spark.sparkContext, trainSize, dimensions, numPartitions, 1230568)
    old_trainingRDD.take(5).foreach(println)
    val validationSize = trainSize/2
    val old_validationRDD : RDD[MLVector] = normalVectorRDD(spark.sparkContext, validationSize, dimensions, numPartitions, 5465694)
    // Using our background as input
    // Define case class
    //val trainingRDD : RDD[MLVector] = filtered_background.select("deltaRLep2ndClosestBJet").as[Double].collect()
    //val validationRDD : RDD[MLVector] = filtered_background.select("LJet_m_plus_RCJet_m_12").as[Double].collect()

    //trainingRDD.take(5).foreach(println)
    //sys.exit(0)

    val trainingRDD: RDD[MLVector] = filtered_background.map {
      case Row(col1: Double, col2: Double, col3: Double, _*) =>
        // Adjust the types according to your actual columns' types
        (col1, col2, col3)
    }

    val validationRDD: RDD[MLVector] = filtered_background.map {
      case Row(col1: Double, col2: Double, col3: Double, _*) =>
        // Adjust the types according to your actual columns' types
        (col1, col2, col3)
    }

    var rectTrain : Rectangle = RectangleFunctions.boundingBox(trainingRDD)
    var rectValidation : Rectangle = RectangleFunctions.boundingBox(validationRDD)
    val rootBox : Rectangle = RectangleFunctions.hull(rectTrain, rectValidation)

    val tree : WidestSplitTree = widestSideTreeRootedAt(rootBox)

    val finestResSideLength = 1e-2
    val finestResDepth : Int = tree.descendBoxPrime(Vectors.dense(rootBox.low.toArray)).dropWhile(_._2.widths.max > finestResSideLength).head._1.depth

    var countedTrain : RDD[(NodeLabel, Count)] = quickToLabeled(tree, finestResDepth, trainingRDD).cache
    var countedValidation : RDD[(NodeLabel, Count)] = quickToLabeledNoReduce(tree, finestResDepth, validationRDD)

    val sampleSizeHint : Int = 1000

    val partitioner : SubtreePartitioner = new SubtreePartitioner(numPartitions, countedTrain, sampleSizeHint)

    implicit val ordering : Ordering[NodeLabel] = leftRightOrd
    val subtreeRDD = countedTrain.repartitionAndSortWithinPartitions(partitioner)

    val depthLimit = partitioner.maxSubtreeDepth

    val minimumCountLimit : Int = 400
    val countLimit = getCountLimit(countedTrain, minimumCountLimit)

    val finestHistogram : Histogram = mergeLeavesHistogram(tree, subtreeRDD, countLimit, depthLimit)
    countedTrain.unpersist()

    val kInMDE = 10

    val numCores = 4

    val minimumDistanceHistogram : Histogram = getMDE(finestHistogram, countedValidation, validationSize, kInMDE, numCores, true)

    val minimumDistanceDensity : DensityHistogram = toDensityHistogram(minimumDistanceHistogram)

    val density : DensityHistogram = minimumDistanceDensity.normalize

    /* TODO: write a folder to save data to */
    val pathToDataFolder = "data/"

    val rootPath = s"${pathToDataFolder}/introduction"
    val treePath = rootPath + "spatialTree"
    val finestResDepthPath = rootPath + "finestRes"
    val mdeHistPath = rootPath + "mdeHist"

    Vector(tree.rootCell.low, tree.rootCell.high).toIterable.toSeq.toDS.write.mode("overwrite").parquet(treePath)
    Array(finestResDepth).toIterable.toSeq.toDS.write.mode("overwrite").parquet(finestResDepthPath)
    minimumDistanceHistogram.counts.toIterable.toSeq.toDS.write.mode("overwrite").parquet(mdeHistPath)
    countedValidation.unpersist()
    // Write out csv file on the form
    // deltaR_min_bb	<	2.7
    // delta_R_min_bb > 0.5

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
