# NYT Data Science Interview Assignment

Welcome to our Data Science take-home exercise!

The problem you will work on as part of this exercise is based on a real Data Science project at The New York Times. The dataset is representative, though simulated. Note that the data might contain inconsistencies or artifacts, just like real-world data often does. Part of the exercise is thinking through how the data should look, to make a best guess, and to document the choices you made.
In completing this work, you should feel free to work in whatever way is typical for you and use whatever resources you would have regular access to as a member of our team. If you have any follow-up questions, feel free to open a GitHub issue in this repo.

## Data Description
In `homework.json.gz` you will find a series of JSON objects (one object per decompressed line) containing partial content of several New York Times assets (articles, feature pieces, etc.). You will also find a set of opaque tags for each asset. These tags correspond to the emotional reaction(s) to an asset, as labeled by a human. Specifically:

|Column|Description|
|------|-----------|
|`headline`|The complete headline of the asset at time of first publication|
|`summary`|A sentence from the body of the article deemed representative of its overall content|
|`worker_id`|A numeric ID indicating the specific human who generated the emotional tags|
|`emotion_{0-9}`|Binary flags indicating emotions evoked in the reader by the article headline and summary|

 
## Deliverables
Please construct a python program that, when run, will do the following:

* Construct a predictive model of which emotional reactions are present in a given asset
* Output, to stderr or a file, an appropriate quantitative estimate of the model's ability to correctly predict emotional reactions

You may use any python modules, libraries, or frameworks you require.
Along with the program, please submit a write-up reflecting:
* any steps you took before modeling the data (e.g., data cleaning, EDA),
* the subjective design choices you made as part of the analysis (e.g., feature selection, model selection),
* what trade-offs these choices reflect, and
* what you might consider doing were you to spend more time on the challenge.

Please submit solutions as pushed+committed changes.


Best regards, The Data Science Group, The New York Times
