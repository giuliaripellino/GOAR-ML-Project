import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.random.RandomRDDs.normalVectorRDD
import org.apache.commons.rng.simple.RandomSource
import org.apache.commons.rng.UniformRandomProvider
import org.apache.spark.mllib.linalg.{Vectors, Vector => MLVector, _}
import org.apache.spark.sql.{DataFrame, Row, SQLContext, SparkSession}
import scala.math.{min, max, abs, sqrt, sin, cos, BigInt}
import java.math.BigInteger
import org.apache.spark.{SparkContext,_}
import scala.sys.process._
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
    val limitsPath = rootPath + "output/limits"
    val plotValuesPath = rootPath + "output/plotValues"
    val samplePath = rootPath + "output/sample"

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

    val filtered_background = df_background//filterAndSelect(df_background)
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
        }
      }

    val Array(trainingDF, validationDF) = filtered_background.randomSplit(Array(0.8,0.2))

    val numTrainingPartitions = 100 // When using filtering, see line 69, the partition number which works locally for me is 23.

    /*
    val trainSize : Long = filtered_bkg_count
    val backgroundRDD : RDD[MLVector] = normalVectorRDD(spark.sparkContext, trainSize, 3, numTrainingPartitions, 1230568)
    val validationRDD : RDD[MLVector] = normalVectorRDD(spark.sparkContext, trainSize/2, 3, numTrainingPartitions, 12305)
    */

    val backgroundRDD = df_to_RDD(trainingDF).repartition(numTrainingPartitions)
    val validationRDD = df_to_RDD(validationDF).repartition(numTrainingPartitions)
    println("PARTITIONS FOR RDD",backgroundRDD.getNumPartitions,validationRDD.getNumPartitions)

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
    //val density = toDensityHistogram(mdeHist).normalize

    // Read mdeHist into plottable objects
    val treeVec = spark.read.parquet(treePath).as[Vector[Double]].collect
    val lowArr : Array[Double] = new Array(2)
    val highArr : Array[Double] = new Array(2)
    for (j <- 0 to 1) {
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

    def savePlotValues(density : DensityHistogram, rootCell : Rectangle, pointsPerAxis : Int, limitsPath : String, plotValuesPath : String) = {

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
          values(i * pointsPerAxis + j) = density.density(Vectors.dense(x4_p, x6_p))
        }
      }
      Array(values).toIterable.toSeq.toDS.write.mode("overwrite").parquet(plotValuesPath)
    }
    def saveSupportPlot(density : DensityHistogram, rootCell : Rectangle, coverage : Double, limitsPath : String, supportPath : String) = {
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

      var n = 0
      /* bottom_left, width1, width2 */
      var values : Array[Double] = new Array(4 * density.densityMap.vals.length)
      for (i <- density.densityMap.vals.indices) {
        val rect = tree.cellAt(density.densityMap.truncation.leaves(i))
        val centre = Vectors.dense(rect.centre(0), rect.centre(1))
        if (coverageRegions.query(centre) <= coverage) {
          values(n + 0) = rect.low(0)
          values(n + 1) = rect.low(1)
          values(n + 2) = rect.high(0) - rect.low(0)
          values(n + 3) = rect.high(1) - rect.low(1)
          n += 4
        }
      }

      Array(values.take(n)).toIterable.toSeq.toDS.write.mode("overwrite").parquet(supportPath)
    }
    /*
    def savePlotValuesCoverage(density : DensityHistogram, rootCell : Rectangle, coverage : Double, pointsPerAxis : Int, limitsPath : String, plotValuesPath : String) = {
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
    def saveSample(density : DensityHistogram, sampleSize : Int, limitsPath : String, samplePath : String, seed : Long) = {

      val limits : Array[Double] = Array(
        density.tree.rootCell.low(0),
        density.tree.rootCell.high(0),
        density.tree.rootCell.low(1),
        density.tree.rootCell.high(1),
      )
      Array(limits).toIterable.toSeq.toDS.write.mode("overwrite").parquet(limitsPath)

      val rng : UniformRandomProvider = RandomSource.XO_RO_SHI_RO_128_PP.create(seed)
      val sample = density.sample(rng, sampleSize).map(_.toArray)

      var arr : Array[Double] = new Array(2 * sample.length)
      for (i <- sample.indices) {
        arr(2*i + 0) = sample(i)(0)
        arr(2*i + 1) = sample(i)(1)
      }

      Array(arr).toIterable.toSeq.toDS.write.mode("overwrite").parquet(samplePath)
    }

    // Saving the important files
      // plotValues
      */
    val pointsPerAxis = 256
    savePlotValues(density, density.tree.rootCell, pointsPerAxis, limitsPath, plotValuesPath)

    //val seed : Long = 123463
    //saveSample(density, 10000, limitsPath, samplePath, seed)

    // Plot mdeHists
    // important for this section is to have "limitsPath","valuesPath", "samplePath" defined and populated with parquet files.
    // Also populate this section with scala.sys.process._ , capable of calling ../postprocessing/plotting.py, where the plotting scripts will live
    val plottingScriptPath = "../Postprocessing/Plotting.py"
    val process = Process(Seq("python3",plottingScriptPath)).run()
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
