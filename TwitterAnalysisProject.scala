import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}

object TwitterAnalysisProject extends App{
  val startTime = System.currentTimeMillis()

  case class User(id: String, name: String)

  val numPartitions = 4
  type Tag = String
  type Likes = Long
  type Feature = Float
  type FeatureTuple = (Feature, Feature, Feature, Feature, Likes)
  type Theta = Array[Float]

  case class Tweet(text: String, user: User, hashTags: List[Tag], likes: Likes)

  def extractFeatures(tweets: RDD[Tweet]): RDD[FeatureTuple] = {
    tweets.map(tweet => {
      val length = tweet.text.length.toFloat
      val numHashtags = tweet.hashTags.size.toFloat
      val userId = tweet.user.id.toFloat
      val likes = tweet.likes
      (length, numHashtags, userId, 1.0f, likes)
    })
  }

  def scaleFeatures(featureRDD: RDD[FeatureTuple]): RDD[FeatureTuple] = {
    val vectors = featureRDD.map(t => Vectors.dense(t.productIterator.toArray.map(_.toString.toDouble)))
    val summary: MultivariateStatisticalSummary = Statistics.colStats(vectors)
    val means = summary.mean.toArray
    val stdDevs = summary.variance.toArray.map(math.sqrt)

    featureRDD.map { case (f1, f2, f3, f4, likes) =>
      val scaledF1 = if (stdDevs(0) != 0) ((f1 - means(0)) / stdDevs(0)).toFloat else f1
      val scaledF2 = if (stdDevs(1) != 0) ((f2 - means(1)) / stdDevs(1)).toFloat else f2
      val scaledF3 = if (stdDevs(2) != 0) ((f3 - means(2)) / stdDevs(2)).toFloat else f3
      val scaledF4 = if (stdDevs(3) != 0) ((f4 - means(3)) / stdDevs(3)).toFloat else f4
      (scaledF1, scaledF2, scaledF3, scaledF4, likes)
    }
  }

  def cost(scaledFeatureRDD: RDD[FeatureTuple], theta: Theta): Float = {
    val m = scaledFeatureRDD.count()
    scaledFeatureRDD.map { case (f1, f2, f3, f4, likes) =>
      val prediction = theta(0) + theta(1) * f1 + theta(2) * f2 + theta(3) * f3 + theta(4) * f4
      math.pow(prediction - likes, 2)
    }.reduce(_ + _).toFloat / (2 * m)
  }


  def gradientDescent(scaledFeatureRDD: RDD[FeatureTuple], theta: Theta, alpha: Float, sigma: Float): Theta = {
    var error = cost(scaledFeatureRDD, theta)
    var delta = Float.MaxValue
    var newTheta = theta.clone()
    val m = scaledFeatureRDD.count()
    do {
      val oldError = error
      val gradient = scaledFeatureRDD.map { case (f1, f2, f3, f4, likes) =>
        val prediction = newTheta(0) + newTheta(1) * f1 + newTheta(2) * f2 + newTheta(3) * f3 + newTheta(4) * f4
        val error = prediction - likes
        (error * f1, error * f2, error * f3, error * f4, error)
      }.reduce((a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3, a._4 + b._4, a._5 + b._5))

      newTheta(0) -= alpha * gradient._5 / m
      newTheta(1) -= alpha * gradient._1 / m
      newTheta(2) -= alpha * gradient._2 / m
      newTheta(3) -= alpha * gradient._3 / m
      newTheta(4) -= alpha * gradient._4 / m

      error = cost(scaledFeatureRDD, newTheta)
      delta = oldError - error
    } while (delta > sigma)
    newTheta
  }


  val spark = SparkSession.builder
    .appName("Twitter Analysis")
    .master("local[*]")
    .config("spark.executor.memory", "4g")
    .getOrCreate()


  val df = spark.read.json("data/tweets")

  val tweets: RDD[Tweet] = df.rdd.flatMap(row => {
    def parseTweet(sourceRow: Row): Tweet = {
      val text = sourceRow.getAs[String]("text")

      val userRow = sourceRow.getAs[Row]("user")
      val user = User(userRow.getAs[String]("id_str"), userRow.getAs[String]("name"))

      val entitiesRow = sourceRow.getAs[Row]("entities")
      val hashTags = if (entitiesRow != null && !entitiesRow.isNullAt(entitiesRow.fieldIndex("hashtags"))) {
        entitiesRow.getAs[Seq[Row]]("hashtags").map(_.getAs[String]("text")).toList
      } else {
        List[String]()
      }

      val likes = sourceRow.getAs[Long]("favorite_count")

      Tweet(text, user, hashTags, likes)
    }

    val retweetedStatus = if (!row.isNullAt(row.fieldIndex("retweeted_status"))) {
      val retweetedRow = row.getAs[Row]("retweeted_status")
      Some(parseTweet(retweetedRow))
    } else None

    val quotedStatus = if (!row.isNullAt(row.fieldIndex("quoted_status"))) {
      val quotedRow = row.getAs[Row]("quoted_status")
      Some(parseTweet(quotedRow))
    } else None

    retweetedStatus.orElse(quotedStatus)
  }).repartition(numPartitions)

  val featureRDD = extractFeatures(tweets).repartition(numPartitions).persist()

  val scaledFeatureRDD = scaleFeatures(featureRDD).repartition(numPartitions).persist()

  // Initialize theta and hyperparameters for gradient descent
  val theta = Array(0f, 0f, 0f, 0f, 0f)
  val alpha = 0.01f
  val sigma = 0.0001f

  // Run gradient descent and print the resulting theta
  val resultTheta = gradientDescent(scaledFeatureRDD, theta, alpha, sigma)
  println("Resulting theta: " + resultTheta.mkString(", "))

  //print time to check the total time for the process
  println("Total time: " + (System.currentTimeMillis() - startTime) / 1000 + " seconds")

  scaledFeatureRDD.unpersist()
  featureRDD.unpersist()

  spark.stop()

}
