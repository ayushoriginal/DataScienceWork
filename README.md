# Data Science Work
This repository contains some of my projects in Exploratory Data Analysis, Statistical Inference, Regression Models and other aspects of Practical Machine Learning

## [Taxi Trips](https://github.com/ayushoriginal/DataScienceWork/tree/master/TaxiTrip)
### AIM: Predict the total ride duration of taxis

The goal is to construct a model that can predict the cumulative ride duration of taxi trips in a city. I've taken the dataset released by the NYC Taxi and Limousine Commission, which includes 
-pickup time 
-geo-coordinates 
-number of passengers
 and several other variables.

### First part - Data exploration
The first part is to analyze the dataframe and observe correlation between variables.
![Distributions](https://github.com/ayushoriginal/DataScienceWork/blob/master/TaxiTrip/pic/download.png)
![Rush Hour](https://github.com/ayushoriginal/DataScienceWork/blob/master/TaxiTrip/pic/rush_hour.png)

### Second part - Clustering
The goal of this playground is to predict the trip duration of test set. We know that some neighborhoods are more congested. So, I used K-Means to compute geo-clusters for pickup and drop off.
![Cluster](https://github.com/ayushoriginal/DataScienceWork/blob/master/TaxiTrip/pic/nyc_clusters.png)

### Third part - Cleaning and feature selection 
I have found some odd long trips : one day trip with a mean spead < 1km/h.   
![Outliners](https://github.com/ayushoriginal/DataScienceWork/blob/master/TaxiTrip/pic/outliners.png)
I have removed these outliners.  

I also added features from the data available : Haversine distance, Manhattan distance, means for clusters, PCA for rotation.

### Forth part - Prediction
I compared Random Forest and XGBoost.  
Current Root Mean Squared Logarithmic error : 0.391

Feature importance for RF & XGBoost
![Feature importance](https://github.com/ayushoriginal/DataScienceWork/blob/master/TaxiTrip/pic/feat_importance.png)

## [Face Recognition](https://github.com/ayushoriginal/DataScienceWork/tree/master/FaceRecognition)
### AIM: Face Recognition using deep learning and Histogram of Oriented Gradients

#### Methodology

1. Find faces in image (HOG Algorithm)   
2. Affine Transformations (Face alignment using an ensemble of regression
trees)   
3. Encoding Faces (FaceNet)  
4. Make a prediction (Linear SVM)  

I'm using the [Histogram of Oriented Gradients](http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) (HOG) method. I'm computing the weighted vote orientation  gradients of 16x16 pixels squares (Instead of computing gradients for every pixel of the image which would have a lot of detail) . We get a simple representation (HOG image) that captures the basic structure of a face.  
All we have to do is find the part of our image that looks the most similar to a known trained HOG pattern.  
For this technique, we use the dlib Python library to generate and view HOG representations of images.  
```
face_detector = dlib.get_frontal_face_detector()
detected_faces = face_detector(image, 1)
```

After isolating the faces in our image, we need to warp (posing and projecting)the picture so the face is always in he same place. To do this, we are going to use the [face landmark estimation algorithm](http://www.csc.kth.se/~vahidk/papers/KazemiCVPR14.pdf). Following this method, there are 68 specific points (landmarks) on every face and we train a machine learning algorithm to find these 68 specific points on any face. 
```
face_pose_predictor = dlib.shape_predictor(predictor_model)
pose_landmarks = face_pose_predictor(img, f)
```
After find those landmarks, we need to use affine transformations (such as rotating, scaling and shearing --like translations) on the image so that the eyes and mouth are centered as best as possible.
```
face_aligner = openface.AlignDlib(predictor_model)
alignedFace = face_aligner.align(534, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
```

The next step is encoding the detected face. For this, we use Deep Learning. We train a neural net to generate [128 measurements (face embedding)](http://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf) for each face.  
The training process works by looking at 3 face images at a time:  

- Load two training face images of the same known person and generate for the two pictures the 128 measurements
- Load a picture of a  different person and generate for the two pictures the 128 measurements  
Then we tweak the neural network slightly so that it makes sure the measurements for the same person are slightly closer while making sure the measurements for the two different persons are slightly further apart.
Once the network has been trained, it can generate measurements for any face, even ones it has never seen before!

```
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)
face_encoding = np.array(face_encoder.compute_face_descriptor(image, pose_landmarks, 1))
```

Finally, we need a classifier (Linear SVM or other classifier) to find the person in our database of known people who has the closest measurements to our test image. We train the classifier with the measurements as input.

Thanks to Adam Geitgey who wrote a great [post](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78) about this, I followed his pipeline.

![Result](https://github.com/ayushoriginal/DataScienceWork/blob/master/FaceRecognition/result.png)
 
Tried with my own pics added to the dataset :P

![ayush](https://github.com/ayushoriginal/DataScienceWork/tree/master/pics/webcam.jpg)

## [Neural Machine Translation](https://github.com/ayushoriginal/DataScienceWork/tree/master/FaceRecognition)

Neural Machine Translation using Tensorflow using embedding with attention sequence-to-sequence model.

![20](https://github.com/ayushoriginal/DataScienceWork/tree/master/pics/rnn.jpg)
![21](https://github.com/ayushoriginal/DataScienceWork/tree/master/pics/seq2seq.JPG)

#### English

``Why did the naive Bayesian suddenly feel patriotic when he heard fireworks? (Question)
He assumed independence. (Assertion)

What did the gym coach say about the machine learning model? (Question)
"You do not need training, you're overfit!" (Exclaimation)

Ayush has skin. Potatoes have skin. Ayush must be a potato. (Set of assertions)
``
#### German Translation

```

Warum fühlte sich der naive Bayesianer plötzlich patriotisch, als er ein Feuerwerk hörte?
Er hat Unabhängigkeit angenommen.

Was hat der Fitnesstrainer über das maschinelle Lernmodell gesagt?
"Du brauchst kein Training, du bist überfit!"

Ayush hat Haut. Kartoffeln haben Haut. Ayush muss eine Kartoffel sein.

````

#### Reverse translating via Google

![Translate](https://i.imgur.com/EJQPf7l.jpg)

## [Bike Sharing Demand](https://github.com/ayushoriginal/DataScienceWork/tree/master/BikeSharingDemand)
### AIM: Forecast use of a city bikeshare system

Bike sharing systems are a means of renting bicycles where the process of obtaining membership, rental, and bike return is automated via a network of kiosk locations throughout a city. Using these systems, people are able rent a bike from a one location and return it to a different place on an as-needed basis. Currently, there are over 500 bike-sharing programs around the world.

The data generated by these systems makes them attractive for researchers because the duration of travel, departure location, arrival location, and time elapsed is explicitly recorded. Bike sharing systems therefore function as a sensor network, which can be used for studying mobility in a city. In this competition, participants are asked to combine historical usage patterns with weather data in order to forecast bike rental demand in the Capital Bikeshare program in Washington, D.C.

The goal of this challenge is to build a model that predicts the count of bike shared, exclusively based on contextual features. The first part of this challenge was aimed to understand, to analyse and to process those dataset. I wanted to produce meaningful information with plots. The second part was to build a model and use a Machine Learning library in order to predict the count.

The more importants parameters were the time, the month, the temperature and the weather.  
Multiple models were tested during this challenge (Linear Regression, Gradient Boosting, SVR and Random Forest). Finally, the chosen model was Random Forest. The accuracy was measured with [Out-of-Bag Error](https://www.stat.berkeley.edu/~breiman/OOBestimation.pdf) and the OOB score was 0.85.

The results on implementing several models is shown-

### Linear Model

Predicting using the attributes from testing dataset and plot them against the true values shows that the simple linear model is limited and cannot explain most of the variation in the response variable. 

![16](https://github.com/ayushoriginal/DataScienceWork/tree/master/plots/13_lm_predict.png)

### Generalized Linear Model

Predicting using the attributes from testing dataset and plot them against the true values shows that the generalized linear model is significantly more accurate in predicting the variations in the response variable.

![17](https://github.com/ayushoriginal/DataScienceWork/tree/master/plots/14_glm_predict.png)

### Generalized Addictive Model

Here, I only used the third generalized addictive model in predicting. The plot shows that the spread of the response variables is similar to generalized linear model. This is understandable since the goodness of fit only improved by about 1%.

![18](https://github.com/ayushoriginal/DataScienceWork/tree/master/plots/15_gam3_predict.png)

It is important to note that none of the statistical models has predicted the trend of the bike sharing rental count. This is due to the fact that the dataset does not contain relative predictor variables that can explain the seasonality, plus, I cannot simply transform the dataset and remove the trend without proper information allowing me to.


## [Analysis of Soccer data](https://github.com/ayushoriginal/DataScienceWork/tree/master/Soccer)
### AIM: Analyze diverse soccer datasets and give useful conclusions (TODO: add a real aim)
   
## First Part - Parsing data

The first part aims to parse data from multiple websites : games, teams, players, etc.  
To collect data about a team and dump a json file (must have created a ``./teams`` folder) :
``python3 dumper.py team_name`` 

## Second Part - Data Analysis

The second part is to analyze the dataset to understand what I can do with it.

![Correlation Matrix](https://github.com/ayushoriginal/DataScienceWork/blob/master/pics/psg_stats.png)

![PSG vs Saint-Etienne](https://github.com/ayushoriginal/DataScienceWork/blob/master/pics/psg_ste.png)


## [Understanding the Amazon from Space](https://github.com/ayushoriginal/DataScienceWork/tree/master/Amazon) 
### AIM: Use satellite data to track the human footprint in the Amazon rainforest.
 (Deep Learning model (using Keras) to label satellite images)

The goal of this challenge is to label satellite image chips with atmospheric conditions and various classes of land cover/land use. Resulting algorithms will help the global community better understand where, how, and why deforestation happens all over the world - and ultimately how to respond.

This problem was tackled with Deep Learning models (using TensorFlow and Keras).  
Submissions are evaluated based on the F-beta score (F2 score), it measures acccuracy using precision and recall.

![19](https://github.com/ayushoriginal/DataScienceWork/tree/master/plots/Dataflow.png)


## [Predicting IMDB movie rating](https://github.com/ayushoriginal/DataScienceWork/tree/master/MovieRating)
### AIM: Predict ratings of a movie before its released
Project inspired by Chuan Sun [work](https://www.kaggle.com/deepmatrix/imdb-5000-movie-dataset)  

Main question : How can we tell the 'greatness' of a movie before it is released in cinema?

## First Part - Parsing data

The first part aims to parse data from the imdb and the numbers websites : casting information, directors, production companies, awards, genres, budget, gross, description, imdb_rating, etc.  
To create the movie_contents.json file :  
``python3 parser.py nb_elements``  

## Second Part - Data Analysis

The second part is to analyze the dataframe and observe correlation between variables. For example, are the movie awards correlated to the worlwide gross ? Does the more a movie is a liked, the more the casting is liked ? 
See the jupyter notebook file.  

![Correlation Matrix](https://github.com/ayushoriginal/DataScienceWork/blob/master/pics/corr_matrix.png)

As we can see in the pictures above, the imdb score is correlated to the number of awards and the gross but not really to the production budget and the number of facebook likes of the casting.  
Obviously, domestic and worlwide gross are highly correlated. However, the more important the production budget, the more important the gross.  
As it is shown in the notebook, the budget is not really correlated to the number of awards.  
What's funny is that the popularity of the third most famous actor is more important for the IMDB score than the popularity of the most famous score (Correlation 0.2 vs 0.08).  
(Many other charts in the Jupyter notebook)

## Third Part - Predict the IMDB score

Machine Learning to predict the IMDB score with the meaningful variables.  
Using a Random Forest algorithm (500 estimators). 
![Most important features](https://github.com/ayushoriginal/DataScienceWork/blob/master/pics/features.png)

