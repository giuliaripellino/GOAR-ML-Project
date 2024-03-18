import org.apache.spark.mllib.linalg.{Vectors, Vector => MLVector, _}
import org.apache.spark.sql.{DataFrame, Row, SQLContext, SparkSession}
import org.apache.spark.{SparkContext, _}
import org.apache.log4j.{Level, Logger}
import scala.util.Random
import org.apache.spark.mllib.random.RandomRDDs.normalVectorRDD
import org.apache.spark.rdd.RDD
import org.apache.commons.rng.sampling.distribution.SharedStateDiscreteSampler
import org.apache.commons.rng.sampling.distribution.AliasMethodDiscreteSampler
import org.apache.commons.math3.distribution.BetaDistribution
import org.apache.commons.rng.simple.RandomSource
import org.apache.commons.rng.UniformRandomProvider

import scala.math.{BigInt, abs, cos, max, min, sin, sqrt}
import java.math.BigInteger
import scala.sys.process._
import co.wiklund.disthist._
import co.wiklund.disthist.Types._
import co.wiklund.disthist.MDEFunctions._
import co.wiklund.disthist.LeafMapFunctions._
import co.wiklund.disthist.SpatialTreeFunctions._
import co.wiklund.disthist.HistogramFunctions._
import co.wiklund.disthist.MergeEstimatorFunctions._

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

    // Reduces INFO print statements in terminal
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
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
    val prefix: String = "HPS_count10000_sampleSize10000/" // Supposed to define the output folder in "SparksInTheDark/output/"
    val treePath: String = rootPath + prefix + "spatialTree"
    val finestResDepthPath: String = rootPath + prefix + "finestRes"
    val finestHistPath: String = rootPath + prefix + "finestHist"
    val mdeHistPath: String = rootPath + prefix + "mdeHist"
    val trainingPath: String = rootPath + prefix + "countedTrain"
    val limitsPath = rootPath + prefix + "limits"
    val plotValuesPath = rootPath + prefix + "plotValues"
    val samplePath = rootPath + prefix + "sample"

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
    // TODO: HAS BEEN CHANGED TO MAKE BACKGROUND RUN. NEED MORE DATA
    val filtered_background = df_background //filterAndSelect(df_background)
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

    // Turn spark dataframes into RDD
    def df_to_RDD(df: DataFrame): org.apache.spark.rdd.RDD[org.apache.spark.mllib.linalg.Vector]  = {
      df.rdd.map {row =>
        val deltaRLep2ndClosestBJet = row.getAs[Float]("deltaRLep2ndClosestBJet").toDouble
        val lJet_m_plus_RCJet_m_12 = row.getAs[Float]("LJet_m_plus_RCJet_m_12").toDouble
        val bb_m_for_minDeltaR = row.getAs[Float]("bb_m_for_minDeltaR").toDouble
        Vectors.dense(deltaRLep2ndClosestBJet, lJet_m_plus_RCJet_m_12, bb_m_for_minDeltaR)
        //Vectors.dense(bb_m_for_minDeltaR,lJet_m_plus_RCJet_m_12)
        }
      }
    // Set Randomization seed
    val seed = 1234
    Random.setSeed(seed)

    val Array(trainingDF, validationDF) = filtered_background.randomSplit(Array(0.75,0.25),seed)

    val numTrainingPartitions = 100 // When using filtering, see line 67, the partition number which works locally for me is 23.

    /* data definition used in example notebooks
    val trainSize : Long = filtered_bkg_count
    val trainingRDD : RDD[MLVector] = normalVectorRDD(spark.sparkContext, trainSize, 3, numTrainingPartitions, 1230568)
    val validationRDD : RDD[MLVector] = normalVectorRDD(spark.sparkContext, trainSize/2, 3, numTrainingPartitions, 12305)
    */

    val trainingRDD = df_to_RDD(trainingDF).repartition(numTrainingPartitions)
    val validationRDD = df_to_RDD(validationDF).repartition(numTrainingPartitions)

    val dimensions = trainingRDD.first().size

    // Getting the RDDs into mdeHists
      //  Deriving the box hull of validation & training data. This will be our root regular paving
    var rectTrain = RectangleFunctions.boundingBox(trainingRDD)
    var rectValidation = RectangleFunctions.boundingBox(validationRDD)
    val rootBox = RectangleFunctions.hull(rectTrain, rectValidation)

      // finestResSideLength is the depth where every leafs cell has no side w. length larger than 1e-5.
    val finestResSideLength = 1e-5 // Was 1e-5
    val tree = widestSideTreeRootedAt(rootBox)
    val finestResDepth = tree.descendBoxPrime(Vectors.dense(rootBox.low.toArray)).dropWhile(_._2.widths.max > finestResSideLength).head._1.depth

    Vector(tree.rootCell.low, tree.rootCell.high).toIterable.toSeq.toDS.write.mode("overwrite").parquet(treePath)
    Array(finestResDepth).toIterable.toSeq.toDS.write.mode("overwrite").parquet(finestResDepthPath)
      // Finding the leaf box address, label, for every leaf with a data point inside of it. HEAVY COMPUTATIONAL
    val countedTrain_pre = quickToLabeled(tree, finestResDepth, trainingRDD)
    /* Only works for depth < 128 */
    // dbutils.fs.rm(trainingPath,true) // Command only works in databricks notebook.
    countedTrain_pre.toDS.write.mode("overwrite").parquet(trainingPath)


      // Setting a minimum count limit of 1e5. If a leaf if found with maximum leaf count larger than minimum; we pick that one.
    val minimumCountLimit = 10000 //was 100000 // try with 10000 over night
    val countedTrain = spark.read
      .parquet(trainingPath)
      .as[(NodeLabel, Count)]
      .rdd
    val maxLeafCount = countedTrain.map(_._2).reduce(max)
    println("Max is count is " + maxLeafCount + " at depth " + finestResDepth)
    val countLimit = max(minimumCountLimit, maxLeafCount)

      // Taking the leaf data at the finest resolution and merging the leaves up to the count limit.
      // This produces the most refined histogram we're willing to use as a density estimate

    implicit val ordering : Ordering[NodeLabel] = leftRightOrd
    val sampleSizeHint = 10000 // was 1000
    val partitioner = new SubtreePartitioner(numTrainingPartitions, countedTrain, sampleSizeHint)
    val depthLimit = partitioner.maxSubtreeDepth
    val subtreeRDD = countedTrain.repartitionAndSortWithinPartitions(partitioner)
    val finestHistogram_presave : Histogram = mergeLeavesHistogram(tree, subtreeRDD, countLimit, depthLimit)

      // Saving the (NodeLabel,Count)'s to disk. Has to be used if depth of the leaves are >126.
    finestHistogram_presave.counts.toIterable.toSeq.map(t => (t._1.lab.bigInteger.toByteArray, t._2)).toDS.write.mode("overwrite").parquet(finestHistPath)

      // Reload the histogram
    val counts = spark.read.parquet(finestHistPath).as[(Array[Byte], Count)].map(t => (NodeLabel(new BigInt(new BigInteger(t._1))), t._2)).collect
    val finestHistogram = Histogram(tree, counts.map(_._2).sum, fromNodeLabelMap(counts.toMap)) // .reduce(_+_) has been replaced with .sum

      // Label the validation data
    val countedValidation = quickToLabeledNoReduce(tree, finestResDepth, validationRDD)

    val kInMDE = 10
    val numCores = 16 // Number of cores in cluster

    val mdeHist = getMDE(
      finestHistogram,
      countedValidation,
      validationRDD.count(),
      kInMDE,
      numCores,
      true
    )
    mdeHist.counts.toIterable.toSeq.map(t => (t._1.lab.bigInteger.toByteArray, t._2)).toDS.write.mode("overwrite").parquet(mdeHistPath)
    //val density = toDensityHistogram(mdeHist).normalize

    // Read mdeHist into plottable objects
    val treeVec = spark.read.parquet(treePath).as[Vector[Double]].collect
    val lowArr : Array[Double] = new Array(dimensions)
    val highArr : Array[Double] = new Array(dimensions)
    for (j <- 0 until dimensions) {
      if(treeVec(0)(j) < treeVec(1)(j)) {
        lowArr(j) = treeVec(0)(j)
        highArr(j) = treeVec(1)(j)
      } else {
        lowArr(j) = treeVec(1)(j)
        highArr(j) = treeVec(0)(j)
      }
    }
    val tree_histread = widestSideTreeRootedAt(Rectangle(lowArr.toVector, highArr.toVector))
    //val finestResDepth_read = spark.read.parquet(finestResDepthPath).as[Depth].collect()(0)
    val mdeHist_read = {
      val mdeCounts = spark.read.parquet(mdeHistPath).as[(Array[Byte], Count)].map(t => (NodeLabel(BigInt(new BigInteger(t._1))), t._2))
      Histogram(tree_histread, mdeCounts.map(_._2).reduce(_+_), fromNodeLabelMap(mdeCounts.collect.toMap))
    }
    val density = toDensityHistogram(mdeHist_read).normalize
    def savePlotValues(density : DensityHistogram, rootCell : Rectangle, pointsPerAxis : Int, limitsPath : String, plotValuesPath : String): Unit = {

      val limits : Array[Double] = Array(
        rootCell.low(0),
        rootCell.high(0),
        rootCell.low(1),
        rootCell.high(1),
        rootCell.low(2),
        rootCell.high(2)
      )
      Array(limits).toIterable.toSeq.toDS.write.mode("overwrite").parquet(limitsPath)

      val x4Width = rootCell.high(0) - rootCell.low(0)
      val x6Width = rootCell.high(1) - rootCell.low(1)
      val x8Width = rootCell.high(2) - rootCell.low(2)

      val values: Array[Double] = new Array(pointsPerAxis * pointsPerAxis * pointsPerAxis)
      println("Filling array with a lot of values. Time-complexity here is O(n^3)... ")
      for (i <- 0 until pointsPerAxis) {
        val x4_p = rootCell.low(0) + (i + 0.5) * (x4Width / pointsPerAxis)
        for (j <- 0 until pointsPerAxis) {
          val x6_p = rootCell.low(1) + (j + 0.5) * (x6Width / pointsPerAxis)
          for (k <- 0 until pointsPerAxis) {
            val x8_p = rootCell.low(2) + (k + 0.5) * (x8Width / pointsPerAxis)
            values(i * pointsPerAxis * pointsPerAxis + j * pointsPerAxis + k) = density.density(Vectors.dense(x4_p, x6_p, x8_p))
          }
        }
      }
      Array(values).toIterable.toSeq.toDS.write.mode("overwrite").parquet(plotValuesPath)
    }

    def saveSample(density : DensityHistogram, sampleSize : Int, dimensions : Int, limitsPath : String, samplePath : String, seed : Long): Unit = {

      val limits : Array[Double] = Array(
        density.tree.rootCell.low(0),
        density.tree.rootCell.high(0),
        density.tree.rootCell.low(1),
        density.tree.rootCell.high(1),
      )
      //Array(limits).toIterable.toSeq.toDS.write.mode("overwrite").parquet(limitsPath)

      val rng : UniformRandomProvider = RandomSource.XO_RO_SHI_RO_128_PP.create(seed)
      val sample = density.sample(rng, sampleSize).map(_.toArray)

      var arr : Array[Double] = new Array(dimensions * sample.length)
      println("ARRAY LENGTH:",arr.length,"ARRAY COUNT"," DIMENSIONS:",dimensions," SAMPLE LENGTH:",sample.length)
      for (i <- 0 until sample.length) {
        for (j <- 0 until dimensions) {
          arr(j + dimensions*i) = sample(i)(j)
        }
      }

      Array(arr).toIterable.toSeq.toDS.write.mode("overwrite").parquet(samplePath)
      println("sampleValues saved!")
    }
    val pointsPerAxis = 256
    saveSample(density,200,dimensions,limitsPath,samplePath,seed)
    savePlotValues(density, density.tree.rootCell, pointsPerAxis, limitsPath, plotValuesPath)

    // Plot mdeHists
    // important for this section is to have "limitsPath","valuesPath", "samplePath" defined and populated with parquet files.
    // Also populate this section with scala.sys.process._ , capable of calling ../postprocessing/plotting.py, where the plotting scripts will live
    val plottingScriptPath = "../Postprocessing/Plotting.py"
    val process = Process(Seq("python3",plottingScriptPath,rootPath+prefix)).run()
    process.exitValue()
    /*


   // Get 10% highest density regions

   // Plot 10% highest density regions


   // Write out csv file on the form
   // deltaR_min_bb	<	2.7
   // delta_R_min_bb > 0.5
   // .
   // .
   // .
   */
    spark.stop()
    println("Stopping SparkSession...")

  }
}
