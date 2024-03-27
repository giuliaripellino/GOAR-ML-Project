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

    // ============================= PARAMETER LIST ==================================
    val gcloudRunning: Boolean = false // local running (i.e. false) is mostly used for testing

    val RunBackground: Boolean = true // if false, we run on signal. if true, we run on background

    val Filtering: Boolean = false

    val numTrainingPartitions = 1000
    val finestResSideLength = 1e-5
    val sampleSizeHint = 1000
    val minimumCountLimit = 1

    val kInMDE = 10
    val numCores = 8 // Number of cores in cluster

    val pointsPerAxis = 256 // Important parameter for d-dimensional nested for-loop. The larger value, the slower it will be. Time-complexity is O(pointsPerAxis^dimensions)
    val DensityPercentage = 1

    val StringSigBkg: String = if (RunBackground) {"bkg"} else {"signal"}
    val prefix: String = s"4D_marginalization_tailProb${(DensityPercentage*100).toInt}_${StringSigBkg}_count${minimumCountLimit}_res${finestResSideLength}/" // Supposed to define the output folder in "SparksInTheDark/output/"
    // ===============================================================================

    // Defining paths
    var rootPath: String = ""
    rootPath = if (gcloudRunning) {"gs://sitd-parquet-bucket/"} else {"output/"}

    val treePath: String = rootPath + prefix + "spatialTree"
    val finestResDepthPath: String = rootPath + prefix + "finestRes"
    val finestHistPath: String = rootPath + prefix + "finestHist"
    val mdeHistPath: String = rootPath + prefix + "mdeHist"
    val trainingPath: String = rootPath + prefix + "countedTrain"
    val limitsPath = rootPath + prefix + "limits"
    val plotValuesPath = rootPath + prefix + "plotValues"
    val samplePath = rootPath + prefix + "sample"

    // Read in data from parquet
    val background: String = rootPath + "ntuple_em_v2_scaled.parquet"
    val signal: String = rootPath + "ntuple_SU2L_25_500_v2_scaled.parquet"

    val data: DataFrame = if (RunBackground) {spark.read.parquet(background)}  else {spark.read.parquet(signal)}
    data.show()

    // Function which filters based on pre-defined pre-selection & selects the interesting variables
    def filterAndSelect(df: DataFrame): DataFrame = {
      val filtered_df = df.filter("jet_n == 6 AND bjet_n == 4")
      val selectedColumns = filtered_df.select("deltaRLep2ndClosestBJet","LJet_m_plus_RCJet_m_12","bb_m_for_minDeltaR","HT")
      selectedColumns
    }
    // TODO: HAS BEEN CHANGED TO MAKE BACKGROUND RUN. NEED MORE DATA
    val filtered_data = if (Filtering) {filterAndSelect(data)}  else {data}
    filtered_data.show()

    // We can also see how many events we had before and after filtering:
    val original_data_count = data.count()
    val filtered_data_count = filtered_data.count()

    println(s"# Events before filter: ${original_data_count}")
    println(s"# Events after filter: ${filtered_data_count}")

    // Turn spark dataframes into RDD
    def df_to_RDD(df: DataFrame): org.apache.spark.rdd.RDD[org.apache.spark.mllib.linalg.Vector]  = {
      df.rdd.map {row =>
        val deltaRLep2ndClosestBJet = row.getAs[Float]("deltaRLep2ndClosestBJet").toDouble
        val lJet_m_plus_RCJet_m_12 = row.getAs[Float]("LJet_m_plus_RCJet_m_12").toDouble
        val bb_m_for_minDeltaR = row.getAs[Float]("bb_m_for_minDeltaR").toDouble
        val ht = row.getAs[Float]("HT").toDouble
        Vectors.dense(deltaRLep2ndClosestBJet, lJet_m_plus_RCJet_m_12, bb_m_for_minDeltaR,ht)
        }
      }

    // Set Randomization seed
    val seed = 1234
    Random.setSeed(seed)

    val Array(trainingDF, validationDF) = filtered_data.randomSplit(Array(0.75,0.25),seed)

    /* data definition used in example notebooks
    val trainSize : Long = filtered_data_count
    val trainingRDD : RDD[MLVector] = normalVectorRDD(spark.sparkContext, trainSize, 3, numTrainingPartitions, 1230568)
    val validationRDD : RDD[MLVector] = normalVectorRDD(spark.sparkContext, trainSize/4, 3, numTrainingPartitions, 12305)
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
    val tree = widestSideTreeRootedAt(rootBox)
    val finestResDepth = tree.descendBoxPrime(Vectors.dense(rootBox.low.toArray)).dropWhile(_._2.widths.max > finestResSideLength).head._1.depth

    Vector(tree.rootCell.low, tree.rootCell.high).toIterable.toSeq.toDS.write.mode("overwrite").parquet(treePath)
    Array(finestResDepth).toIterable.toSeq.toDS.write.mode("overwrite").parquet(finestResDepthPath)

    // Finding the leaf box address, label, for every leaf with a data point inside of it. HEAVY COMPUTATIONALLY
    val countedTrain_pre = quickToLabeled(tree, finestResDepth, trainingRDD)
    countedTrain_pre.toDS.write.mode("overwrite").parquet(trainingPath)


    val countedTrain = spark.read.parquet(trainingPath).as[(NodeLabel, Count)].rdd
    val maxLeafCount = countedTrain.map(_._2).reduce(max)
    println("Max is count is " + maxLeafCount + " at depth " + finestResDepth)
    val countLimit = max(minimumCountLimit, maxLeafCount)


    implicit val ordering : Ordering[NodeLabel] = leftRightOrd
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


    val mdeHist = getMDE(
      finestHistogram,
      countedValidation,
      validationRDD.count(),
      kInMDE,
      numCores,
      true
    )
    mdeHist.counts.toIterable.toSeq.map(t => (t._1.lab.bigInteger.toByteArray, t._2)).toDS.write.mode("overwrite").parquet(mdeHistPath)

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

    val margin1 = marginalize(density,Vector(0,1)).normalize
    val margin2 = marginalize(density,Vector(0,2)).normalize
    val margin3 = marginalize(density,Vector(0,3)).normalize
    val margin4 = marginalize(density,Vector(1,2)).normalize
    val margin5 = marginalize(density,Vector(1,3)).normalize
    val margin6 = marginalize(density,Vector(2,3)).normalize

    val margin_dimension = margin1.tree.rootCell.dimension()

    def savePlotValues(density : DensityHistogram, rootCell : Rectangle, coverage : Double, pointsPerAxis : Int, limitsPath : String, plotValuesPath : String): Unit = {
      val coverageRegions : TailProbabilities = density.tailProbabilities
      val limits : Array[Double] = Array(
        rootCell.low(0),
        rootCell.high(0),
        rootCell.low(1),
        rootCell.high(1),
      )
      Array(limits).toIterable.toSeq.toDS.write.mode("overwrite").parquet(limitsPath)

      val x4Width = rootCell.high(0) - rootCell.low(0)
      val x6Width = rootCell.high(1) - rootCell.low(1)

      val values : Array[Double] = new Array(pointsPerAxis * pointsPerAxis)
      for (i <- 0 until pointsPerAxis) {
        val x4_p = rootCell.low(0) + (i + 0.5) * (x4Width / pointsPerAxis)
        for (j <- 0 until pointsPerAxis) {
          val x6_p = rootCell.low(1) + (j + 0.5) * (x6Width / pointsPerAxis)
          if (coverageRegions.query(Vectors.dense(x4_p, x6_p)) <= coverage)
            values(i * pointsPerAxis + j) = density.density(Vectors.dense(x4_p, x6_p))
          else
            values(i * pointsPerAxis + j) = 0.0
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
      Array(limits).toIterable.toSeq.toDS.write.mode("overwrite").parquet(limitsPath)

      val rng : UniformRandomProvider = RandomSource.XO_RO_SHI_RO_128_PP.create(seed)
      val sample = density.sample(rng, sampleSize).map(_.toArray)
      var arr : Array[Double] = new Array(dimensions * sample.length)
      for (i <- sample.indices) {
        for (j <- 0 until dimensions) {
          arr(j + dimensions*i) = sample(i)(j)
        }
      }

      Array(arr).toIterable.toSeq.toDS.write.mode("overwrite").parquet(samplePath)
    }

    /* Looping through all of the margin combinations*/
    val combList = List(margin1, margin2, margin3, margin4, margin5, margin6)
    var col1: String = ""
    var col2: String = ""

    for (margin <- combList) {
      if (margin == margin1) {
        col1 = "deltaRLep2ndClosestBJet"
        col2 = "LJet_m_plus_RCJet_m_12"
      }
      else if (margin == margin2) {
        col1 = "deltaRLep2ndClosestBJet"
        col2 = "bb_m_for_minDeltaR"
      }
      else if (margin == margin3){
        col1 = "deltaRLep2ndClosestBJet"
        col2 = "HT"
      }
      else if (margin == margin4){
        col1 = "LJet_m_plus_RCJet_m_12"
        col2 = "bb_m_for_minDeltaR"
      }
      else if (margin == margin5){
        col1 = "LJet_m_plus_RCJet_m_12"
        col2 = "HT"
      }
      else if (margin == margin6){
        col1 = "bb_m_for_minDeltaR"
        col2 = "HT"
      }

      saveSample(margin,filtered_data_count.toInt,margin_dimension,limitsPath,samplePath,seed)
      savePlotValues(margin, margin.tree.rootCell,DensityPercentage ,pointsPerAxis, limitsPath, plotValuesPath)

      val plottingScriptPath = "../Postprocessing/Plotting.py"
      val originalDataPath: String = if (RunBackground) {background}  else {signal}
      val passString = rootPath + prefix
      val ColStrings = List(col1,col2)
      val process = Process(Seq("python3",plottingScriptPath,passString,pointsPerAxis.toString, margin_dimension.toString, originalDataPath,ColStrings.toString)).run()
      process.exitValue()



    }

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
