#!/usr/bin/python

import numpy as np
import pyximport; pyximport.install(setup_args={
                              "include_dirs":np.get_include()},
                            reload_support=True)

import TsetlinMachine
import vteam_params

#build no noise dataset
X_training_no_noise = np.random.randint(2,size=[10000,2],dtype=np.int32)
y_training_no_noise = np.ones([10000],dtype=np.int32)
for i in range(len(X_training_no_noise)):
    if X_training_no_noise[i, 0] == X_training_no_noise[i, 1]:
        y_training_no_noise[i] = 0

X_test_no_noise = np.random.randint(2,size=[10000,2],dtype=np.int32)
y_test_no_noise = np.ones([10000],dtype=np.int32)
for i in range(len(X_training_no_noise)):
    if X_test_no_noise[i, 0] == X_test_no_noise[i, 1]:
        y_test_no_noise[i] = 0


# Parameters for Memristor

voltage = 1.2

Th = 1
init_memristor_state = 0.5

selected_params = vteam_params.get_vteam_params("Linear12")
alpha_off = selected_params["alpha_off"]
alpha_on = selected_params["alpha_on"]
v_off = selected_params["v_off"]
v_on = selected_params["v_on"]
r_on = selected_params["r_on"]
r_off = selected_params["r_off"]
k_off = selected_params["k_off"]
k_on = selected_params["k_on"]
d = selected_params["d"]
dt_off = d / (k_off * (((voltage / v_off) - 1) ** alpha_off))
dt_on = d / (k_on * (((-voltage / v_on) - 1) ** alpha_on))
dt = max(dt_off, -dt_on)/100


# Parameters for the Tsetlin Machine
T = 15 
s = 3.9
number_of_clauses = 20
states = 200

# Parameters of the pattern recognition problem
number_of_features = 12
number_of_features_no_noise = 2
number_of_classes = 2

# Training configuration
epochs = 200

# Loading of training and test data
training_data = np.loadtxt("NoisyXORTrainingData.txt").astype(dtype=np.int32)
test_data = np.loadtxt("NoisyXORTestData.txt").astype(dtype=np.int32)

X_training = training_data[:,0:12] # Input features
y_training = training_data[:,12] # Target value

X_test = test_data[:,0:12] # Input features
y_test = test_data[:,12] # Target value

# This is a multiclass variant of the Tsetlin Machine, capable of distinguishing between multiple classes
tsetlin_machine = TsetlinMachine.TsetlinMachine(number_of_clauses, number_of_features, states, s, T, Th, init_memristor_state, alpha_off, alpha_on, v_off, v_on, r_off, r_on, k_off, k_on, d,voltage, dt_off, dt_on)
tsetlin_machine_no_nose = TsetlinMachine.TsetlinMachine(number_of_clauses, number_of_features_no_noise, states, s, T, Th, init_memristor_state, alpha_off, alpha_on, v_off, v_on, r_off, r_on, k_off, k_on, d,voltage, dt_off, dt_on)

# Training of the Tsetlin Machine in batch mode. The Tsetlin Machine can also be trained online
tsetlin_machine.fit(X_training, y_training, y_training.shape[0], epochs=epochs)

# Some performance statistics

print ("Accuracy on test data (no noise):", tsetlin_machine.evaluate(X_test, y_test, y_test.shape[0]))
print ("Accuracy on training data (40% noise):", tsetlin_machine.evaluate(X_training, y_training, y_training.shape[0]))
print ("Prediction: x1 = 1, x2 = 0, ... -> y = ", tsetlin_machine.predict(np.array([1,0,1,1,1,0,1,1,1,0,0,0],dtype=np.int32)))
print ("Prediction: x1 = 0, x2 = 1, ... -> y = ", tsetlin_machine.predict(np.array([0,1,1,1,1,0,1,1,1,0,0,0],dtype=np.int32)))
print ("Prediction: x1 = 0, x2 = 0, ... -> y = ", tsetlin_machine.predict(np.array([0,0,1,1,1,0,1,1,1,0,0,0],dtype=np.int32)))
print ("Prediction: x1 = 1, x2 = 1, ... -> y = ", tsetlin_machine.predict(np.array([1,1,1,1,1,0,1,1,1,0,0,0],dtype=np.int32)))



# no noise test

tsetlin_machine_no_nose.fit(X_training_no_noise, y_training_no_noise, y_training_no_noise.shape[0], epochs=epochs)

print ("\nAccuracy on test data (no noise):", tsetlin_machine_no_nose.evaluate(X_test_no_noise, y_test_no_noise, y_test_no_noise.shape[0]))
print ("Accuracy on training data (no noise):", tsetlin_machine_no_nose.evaluate(X_training_no_noise, y_training_no_noise, y_training_no_noise.shape[0]))
print ("Prediction: x1 = 1, x2 = 0, ... -> y = ", tsetlin_machine_no_nose.predict(np.array([1,0],dtype=np.int32)))
print ("Prediction: x1 = 0, x2 = 1, ... -> y = ", tsetlin_machine_no_nose.predict(np.array([0,1],dtype=np.int32)))
print ("Prediction: x1 = 0, x2 = 0, ... -> y = ", tsetlin_machine_no_nose.predict(np.array([0,0],dtype=np.int32)))
print ("Prediction: x1 = 1, x2 = 1, ... -> y = ", tsetlin_machine_no_nose.predict(np.array([1,1],dtype=np.int32)))