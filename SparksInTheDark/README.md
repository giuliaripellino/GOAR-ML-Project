## Beginning of SparksInTheDark README

To begin with, all development and testing have been done with the following package versions: 

```
java -version -> 11.0.22
sbt -version -> 1.9.7
scala -version -> 2.12.15
```

## Building

To build and run the code, clone the github repository to the desired location and build the project by
```
cd SparksInTheDark/
sbt clean compile
```

## Running locally
Running the code locally _**[in its current state]**_ is done the easiest with ```sbt run``` from ```SparksInTheDark/```.

## Running remote
In order to run the code on a remote cluster, one firstly needs to assemble the code into a ```.jar``` file. This is done by 
```
sbt clean compile assembly
```
The name of this ```.jar``` file is defined in ```build.sbt```. Note that everytime one edits the ```build.sbt``` file, one should
```
sbt reload
```
for the effects to take place. Another point here is that one _can_ also use ```sbt package ```, but since we're "packaging a package" (the [disthist](https://github.com/lamastex/SparkDensityTree) package) and have a lot of dependencies, ```sbt assembly``` seems to be recommended (see [documentation](https://github.com/sbt/sbt-assembly)).  

The command to actually submit the code to the cluster is (see documentation [here](https://cloud.google.com/sdk/gcloud/reference/dataproc/jobs/submit/spark):
```
gcloud dataproc jobs submit spark --cluster='cluster-sparks-in-the-dark' --region='europe-west4' --jar='target/scala-2.12/JARNAME.jar'
```
A bottleneck which appears with this way of submission is that you have to ```sbt assembly``` (i.e. create a new ```.jar```), everytime you want to run new code on the cluster. From my testing, the usual time it takes to create a ```.jar``` is about 3-5 minutes. 

## TL;DR running
From ```SparksInTheDark/```
```
sbt clean compile assembly
gcloud dataproc jobs submit spark --cluster='cluster-sparks-in-the-dark' \ 
--region='europe-west4' \
--jar='target/scala-2.12/JARNAME.jar'
```