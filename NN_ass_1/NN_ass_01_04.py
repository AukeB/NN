import numpy as np
import matplotlib.pyplot as plt
import random
import sys

DATA_PATH = "/home/auke/MEGA/Study/Courses/2018-2019/Neural_Networks/Assignments/Assignment_01/data/"
train_in_path = "train_in.csv"
train_out_path = "train_out.csv"
test_in_path = "test_in.csv"
test_out_path = "test_out.csv"

train_in = np.genfromtxt ( DATA_PATH + train_in_path, dtype = 'float', delimiter = ',' )
train_out = np.genfromtxt ( DATA_PATH + train_out_path, dtype = 'float', delimiter = ',' )
test_in = np.genfromtxt ( DATA_PATH + test_in_path, dtype = 'float', delimiter = ',' )
test_out = np.genfromtxt ( DATA_PATH + test_out_path, dtype = 'float', delimiter = ',' )

dim_1_train = len ( train_in ) # 1707. Number of digits in training data.
dim_1_test = len ( test_in ) # 1000. Number of digits in testing data.
dim_2 = len ( train_in[0] ) # 256. 256 dimensional space.
dim_3 = 10 # Number of possible digits.

def train_perceptron ( data_in, data_out ):	
	bias = np.zeros ( ( len ( data_in ), 1 ) ) # Initialise bias array.
	data_in = np.append ( data_in, bias, axis = 1 ) # Merge data_in and weights.
	weights = np.random.rand ( 257, 10 ) # Initialise weights array.randomly.

	weighted_sum = np.dot ( data_in, weights ) # Compute weighted sum.
	output_indices = np.argmax ( weighted_sum, axis = 1 ) # Select maximum value.

	epoch = 0

	while np.array_equal ( output_indices, data_out ) == False:
		wrongly_classified_indices = []

		for i in range ( len ( data_in ) ):
			if output_indices[i] != data_out[i]:
				wrongly_classified_indices.append ( i )

		if epoch % 10 == 0: plt.plot ( epoch, ( len ( data_in ) - len ( wrongly_classified_indices ) ) * 100 / float ( len ( data_in ) ), '.' )

		sys.stdout.write ( "{:03.1f}% ".format ( ( len ( data_in ) - len ( wrongly_classified_indices ) ) * 100 / float ( len ( data_in ) ) ) )
		sys.stdout.write ( " {0}\r".format ( epoch ) )
		sys.stdout.flush ()

		k = wrongly_classified_indices[random.randint ( 0, len ( wrongly_classified_indices ) - 1 ) ]

		for j in range ( dim_3 ):
			if weighted_sum[k][j] > weighted_sum[k][int(data_out[k])]:
				weights[:,j] -= data_in[k]
			if j == int ( data_out[k] ):
				weights[:,j] += data_in[k]

		weighted_sum = np.dot ( data_in, weights )
		output_indices = np.argmax ( weighted_sum, axis = 1 )

		epoch += 1

	sys.stdout.write ( "Iterations: {0}\n".format ( epoch ) )

	plt.show () # Toggle

	return weights

def test_perceptron ( data_in, data_out, weights ):
	bias = np.zeros ( ( len ( data_in ), 1 ) ) # Initialise bias array.
	data_in = np.append ( data_in, bias, axis = 1 ) # Merge data_in and weights.

	weighted_sum = np.dot ( data_in, weights )
	output_indices = np.argmax ( weighted_sum, axis = 1 )

	correct_counter = 0

	for i in range ( len ( data_in ) ):
		if output_indices[i] == data_out[i]:
			correct_counter += 1

	sys.stdout.write ( "Accuracy:   {:03.1f}%\n".format ( correct_counter * 100 / len ( data_out ) ) )

for i in range ( 1 ):
	sys.stdout.write ( "Run {0}\n".format ( i + 1 ) )
	weights = train_perceptron ( train_in, train_out )
	test_perceptron ( train_in, train_out, weights )
	test_perceptron ( test_in, test_out, weights )
	sys.stdout.write ( "\n" )

