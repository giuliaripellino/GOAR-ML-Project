import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.random.RandomRDDs.normalVectorRDD
import org.apache.spark.mllib.linalg.{Vectors, Vector => MLVector, _}
import org.apache.spark.sql.{DataFrame, Row, SQLContext, SparkSession}
import scala.math.{min, max, abs, sqrt, sin, cos, BigInt}
import java.math.BigInteger
import org.apache.spark.{SparkContext,_}
import co.wiklund.disthist._
import co.wiklund.disthist.Types._
import co.wiklund.disthist.RectangleFunctions._
import co.wiklund.disthist.MDEFunctions._
import co.wiklund.disthist.LeafMapFunctions._
import co.wiklund.disthist.SpatialTreeFunctions._
import co.wiklund.disthist.HistogramFunctions._
import co.wiklund.disthist.TruncationFunctions._
import co.wiklund.disthist.MergeEstimatorFunctions._
import co.wiklund.disthist.SubtreePartitionerFunctions._
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
    val sqlContext = new SQLContext(sc) // Only way to get "toDS" working
    import sqlContext.implicits._

    // ================ IF TRUE, PATHS CORRESPOND TO GCLOUD PATHS ======================

    val gcloudRunning: Boolean = false // local running (i.e. false) is mostly used for testing

    // =================================================================================

    // Defining paths
    var rootPath: String = ""
    rootPath = if (gcloudRunning) {
      "gs://sitd-parquet-bucket/"
    } else {
      "output/"
    }

    val treePath: String = rootPath + "output/spatialTree"
    val finestResDepthPath: String = rootPath + "output/finestRes"
    val finestHistPath: String = rootPath + "output/finestHist"
    val mdeHistPath: String = rootPath + "output/mdeHist"
    val trainingPath: String = rootPath + "output/countedTrain"


    // Read in data from parquet
    val background: String = rootPath + "ntuple_em_v2.parquet"
    val signal: String = rootPath + "ntuple_SU2L_25_500_v2.parquet"

    val df_background: DataFrame = spark.read.parquet(background)
    df_background.show()

    val df_signal: DataFrame = spark.read.parquet(signal)
    df_signal.show()

    // Function which filters based on pre-defined pre-selection & selects the interesting variables
    def filterAndSelect(df: DataFrame): DataFrame = {
      val filtered_df = df.filter("jet_n == 6 AND bjet_n == 4")
      val selectedColumns = filtered_df.select("deltaRLep2ndClosestBJet","LJet_m_plus_RCJet_m_12","bb_m_for_minDeltaR")
      selectedColumns
    }

    val filtered_background = filterAndSelect(df_background)
    //filtered_background.show()

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

    // Turn spark dataframes into RDD
    def df_to_RDD(df: DataFrame): org.apache.spark.rdd.RDD[org.apache.spark.mllib.linalg.Vector]  = {
      df.rdd.map {row =>
        val deltaRLep2ndClosestBJet = row.getAs[Float]("deltaRLep2ndClosestBJet").toDouble
        val lJet_m_plus_RCJet_m_12 = row.getAs[Float]("LJet_m_plus_RCJet_m_12").toDouble
        val bb_m_for_minDeltaR = row.getAs[Float]("bb_m_for_minDeltaR").toDouble
        Vectors.dense(deltaRLep2ndClosestBJet, lJet_m_plus_RCJet_m_12, bb_m_for_minDeltaR)
        }
      }

    val Array(trainingDF, validationDF) = filtered_background.randomSplit(Array(0.8,0.2))

    val trainSize : Long = 1e6.toLong
    val backgroundRDD : RDD[MLVector] = normalVectorRDD(spark.sparkContext, trainSize, 3, 1000, 1230568)
    val validationRDD : RDD[MLVector] = normalVectorRDD(spark.sparkContext, trainSize/2, 3, 1000, 12305)

    // COMMENTED OUT OUR DATA

    //val backgroundRDD = df_to_RDD(trainingDF)
    //val validationRDD = df_to_RDD(validationDF)

    // Turn RDD into Minimum Density Estimate Histograms (mdeHists)
      //  Deriving the box hull of validation & training data. This will be our root regular paving
    var rectTrain = RectangleFunctions.boundingBox(backgroundRDD)
    var rectValidation = RectangleFunctions.boundingBox(validationRDD)
    val rootBox = RectangleFunctions.hull(rectTrain, rectValidation)

      // finestResSideLength is the depth where every leafs cell has no side w. length larger than 1e-5.
    val finestResSideLength = 1e-3 // Was 1e-5
    val tree = widestSideTreeRootedAt(rootBox)
    val finestResDepth = tree.descendBoxPrime(Vectors.dense(rootBox.low.toArray)).dropWhile(_._2.widths.max > finestResSideLength).head._1.depth

    Vector(tree.rootCell.low, tree.rootCell.high).toIterable.toSeq.toDS.write.mode("overwrite").parquet(treePath)
    Array(finestResDepth).toIterable.toSeq.toDS.write.mode("overwrite").parquet(finestResDepthPath)
      // Finding the leaf box address, label, for every leaf with a data point inside of it. HEAVY COMPUTATIONAL
    val countedTrain = quickToLabeled(tree, finestResDepth, validationRDD)
    /* Only works for depth < 128 */
    // dbutils.fs.rm(trainingPath,true) // Command only works in databricks notebook.
    countedTrain.toDS.write.mode("overwrite").parquet(trainingPath)


      // Setting a minimum count limit of 1e5. If a leaf if found with maximum leaf count larger than minimum; we pick that one.
    val minimumCountLimit = 100 //was 100000
    val countedTrain2 = spark.read
      .parquet(trainingPath)
      .as[(NodeLabel, Count)]
      .rdd
    val maxLeafCount = countedTrain2.map(_._2).reduce(max(_,_))
    println("Max is count is " + maxLeafCount + " at depth " + finestResDepth)
    val countLimit = max(minimumCountLimit, maxLeafCount)

      // Taking the leaf data at the finest resolution and merging the leaves up to the count limit.
      // This produces the most refined histogram we're willing to use as a density estimate

    val numTrainingPartitions = 10 // was 205

    implicit val ordering : Ordering[NodeLabel] = leftRightOrd
    val sampleSizeHint = 100 // was 1000
    val partitioner = new SubtreePartitioner(numTrainingPartitions, countedTrain, sampleSizeHint)
    val depthLimit = partitioner.maxSubtreeDepth
    val subtreeRDD = countedTrain.repartitionAndSortWithinPartitions(partitioner)
    val finestHistogram_presave : Histogram = mergeLeavesHistogram(tree, subtreeRDD, countLimit, depthLimit)

      // Saving the (NodeLabel,Count)'s to disk. Has to be used if depth of the leaves are >126.
    finestHistogram_presave.counts.toIterable.toSeq.map(t => (t._1.lab.bigInteger.toByteArray, t._2)).toDS.write.mode("overwrite").parquet(finestHistPath)

      // Reload the histogram
    val counts = spark.read.parquet(finestHistPath).as[(Array[Byte], Count)].map(t => (NodeLabel(new BigInt(new BigInteger(t._1))), t._2)).collect
    val finestHistogram_postsave = Histogram(tree, counts.map(_._2).sum, fromNodeLabelMap(counts.toMap)) // .reduce(_+_) has been replaced with .sum

      // Label the validation data
    val countedValidation = quickToLabeledNoReduce(tree, finestResDepth, validationRDD)

    val kInMDE = 10
    val numCores = 8 // Number of cores in cluster

    val mdeHist = getMDE(
      finestHistogram_postsave,
      countedValidation,
      validationRDD.count(),
      kInMDE,
      numCores,
      true
    )
    mdeHist.counts.toIterable.toSeq.map(t => (t._1.lab.bigInteger.toByteArray, t._2)).toDS.write.mode("overwrite").parquet(mdeHistPath)
    val density = toDensityHistogram(mdeHist).normalize

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
