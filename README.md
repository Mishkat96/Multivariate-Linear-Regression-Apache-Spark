Design Decisions:

Case classes:
We have used two case classes here one is User which will store id and name
and another case class is Tweet which will store text, user, hashtags and likes.

Functions:
Feature Extraction:
The extractFeature() function will take the raw data from Tweet and return a tuple of five
element which are length, numHashtags, userId, likes and a constant value of 1.0f. The
constant value is generally used as an intercept term in linear regression models.
Feature Scaling:
To ensure all feature are in the same scale we have used the function scaleFeatures(). It
returns an RDD of FeatureTuple with five elements which are scaledF1, scaledF2, scaledF3,
scaledF4 and likes. The likes part has not been scaled as it is a dependent variable.
Cost:
The cost() function will return the mean squared error J((θ). It is a single float value.
Gradient Descent:
The gradientDescent() function will return our resulting theta. Firstly, it initializes the error
by using the cost() function. Afterwards, it keeps on iterating till the error change is less than
sigma. While iterating with respect to the each parameters (f1, f2, f3, f4, likes) it will
calculate the gradient of the cost function. It keeps on updating the newTheta by
subtracting the alpha with the corresponding gradient. After all the iterations are over and if
the error is less than sigma, we will get our newTheta.

Workflow:
1. spark = We have defined a val spark where we start the program by initiating a
SparkSession and naming it Twitter Analysis.
2. df = We loaded the tweets Json file on a val df using Spark’s read.json method.
3. tweets = The raw data that we have gained is not extracted and transoformed into a
more readable and usable format mapping each tweet data to the Tweet case class
4. featureRDD = We have defined a val featureRDD which calls the function
extractFeature. This function transoforms each Tweet into a FeatureTuple and return
a tuple of five element which are length, numHashtags, userId, likes and a constant
value of 1.0f.
5. scaledFeatureRDD = This val calls the function scaleFeature() by providing the
parameter featureRDD. This method is called to standardize the feature by
subtracting the mean and dividing the standard deviation for every feature.
6. resultTheta = This val calls the function gradientDescent to find the optimal model
parameter (theta) which will minimize the cost function. It will keep on iterating till
the change in error will will less than the sigma value.
7. Print:
Finally we will print the resulting theta that we have obtained and, we will also print
the total time that was needed to complete the full flow.

<img width="654" alt="Screenshot 2024-02-24 at 2 10 01 AM" src="https://github.com/Mishkat96/Multivariate-Linear-Regression-Apache-Spark/assets/47037691/bd24239b-2625-4b93-ada1-7ca34c708f3c">
