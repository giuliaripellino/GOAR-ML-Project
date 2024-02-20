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
for the effects to take place.

The command to actually submit the code to the cluster is (see documentation [here](https://cloud.google.com/sdk/gcloud/reference/dataproc/jobs/submit/spark):
```
 gcloud dataproc jobs submit spark --cluster='cluster-sparks-in-the-dark' --region='europe-west4' --jar='target/scala-2.12/JARNAME.jar'
```

to be continued... 


