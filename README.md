1. Files:

sport_articles.csv: dataset to build classification tree
small_train.csv: small training dataset to implement unit tests
small_test.csv: small test dataset to implement unit tests

Node.py: a class to store all elements of a node
Tree.py: a class to do operations about a tree, like building, pruning
test_CART.py: units test to test tree model
split.py: functions of splitting region
credentials.py: store username and password
sql.py: store dataset into database
test_sql_and_data_frame.py: implement unit tests to compare the results of classifier on SQL version and the data frame version
sk_learn.py: build decision tree classifier using sk-learn

benchmark.txt: general summary comments
complexity.txt: theoretical complexity of the main parts of my implementation

error.png: plot of error rate
running_time.png: plot of running time
performance.png: plot of performance



2. Datasets:

The dataset is loaded from UCI Machine Learning Repository(https://archive.ics.uci.edu/ml/datasets/Sports+articles+for+objectivity+analysis). It contains 1000 samples with 53 attributes.1000 sports articles were labeled using Amazon Mechanical Turk as objective or subjective. However, this dataset is too large to build SQL Tree. Therefore, after consulting professor, I decide to build the classification tree using a smaller subset of this dataset with 700 samples and 31 attributes.



3. Command Line Arguments:

python test_CART.py [args....]

Where above [args...] is a placeholder for five command-line arguments: <file_name><number_alpha><low_alpha><high_alpha><kFold><criterion><max_depth>. These arguments are described in detail below:
	1.<file_name>: the file name of dataset
	2.<number_alpha>: the number of alpha which will be generated. 
	3.<low_alpha>: the lowest alpha. 
	4.<high_alpha>: the highest alpha. 
	5.<kFold>: the number of fold used for cross-validation. 
	6.<criterion>: different methods to calculate impurity, where 1 is bayes error, 2 is cross-entropy, 3 is gini index
	7.<max_depth>: the maximum depth of tree

An example of arguments: sports_articles.csv 3 0 1 5 3 4


