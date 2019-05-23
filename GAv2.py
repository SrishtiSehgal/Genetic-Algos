#genetic algorithm wrapper
import random
import numpy as np
import operator
from sklearn.ensemble import RandomForestRegressor
import math
from easygui import multenterbox
import pandas as pd

def INITIALIZATION(popsize, length):
	def generateInd(length):
		return  np.round_(np.random.rand(length+1,), decimals=3) #last item is fitness score which will be updated with fitness()
	pop = [] #list of np.arrays
	for i in range(popsize):
		pop.append(generateInd(length))
	return pop
class SELECTION_CLASS:
	def __init__(self, best_sample):
		self.best = best_sample
		self.random_number = random.random()/2
	def __call__(self, pop):
		nextGeneration = []
		for i in range(self.best):
			nextGeneration.append(pop.pop(i))
		#ROULETTE SELECTION FOR REMAINING
		prob_sum, fitness_sum = 0, 0
		for ind in pop:
			fitness_sum += ind[-1]
		for i in range(len(pop)):
			prob = pop[i][-1]/fitness_sum
			if prob > self.random_number:
				nextGeneration.append(pop[i])
		return nextGeneration
def FITNESS(X, Y, inputsize, individual):
	masked_input = np.multiply(X.T, individual[:-1]).T #all but last item of ind
	tf_model = RandomForestRegressor(n_estimators=100, max_features=inputsize, random_state=0, max_leaf_nodes=1000)
	tf_model.fit(masked_input, Y)
	Y_predicted = tf_model.predict(masked_input)
	rmse = math.sqrt(np.mean((Y-Y_predicted)**2))
	individual[-1] = rmse
	
	#TERMINATION CRITERIA
	if rmse < 0.00005*max(Y):
		print("Solution found")
		print("Individual weights saved")
		np.savetxt("Solution_weights.csv", individual[:-1], fmt = '%f', comments = '', delimiter = ',')
		print("Root Mean Squared Error = ", rmse)
		quit()
	return individual
def CROSSOVER(breeders, popsize):
	def createChildren(individual1, individual2):
		#choose random crossover point and marge lists
		crossover = random.randint(0, len(individual1)-1)
		child1, child2 = np.zeros((individual1.shape[0],)), np.zeros((individual1.shape[0],))
		child1[:crossover],child2[:crossover] = individual1[:crossover], individual2[:crossover]
		child1[crossover:],child2[crossover:] = individual2[crossover:], individual1[crossover:]
		return child1, child2
	nextPopulation = []
	while len(nextPopulation) < (popsize):
		for i in range(len(breeders)):
			child1, child2 =createChildren(breeders[i], breeders[len(breeders) -1 -i])
			nextPopulation.append(child1)
			if len(nextPopulation)>= popsize:
				break
			else:
				nextPopulation.append(child2)
	return nextPopulation
def MUTATION(population, chance_of_mutation):
	def mutateInd(ind):
		for i in range(len(ind)):
			if random.random() < 0.2:
				ind[i] = np.random.uniform(1,10,1)
		return ind
	for i in range(len(population)):
		if random.random() < chance_of_mutation:
			population[i] = mutateInd(population[i])
	return population
def sort_population(pop):
	def fitness_score(ind):
		return ind[-1]
	pop.sort(key=fitness_score)
	return pop
def input_data():
	def same_file(filename):
		'''
		PURPOSE: 
			A method of upload which involves only one file. The user 
			must specify boundaries, outside of which training data 
			will be extracted. Testing data is all values of the file.
		OUTPUT:
			training and testing dataframes
		'''
		all_data = pd.read_csv(filename) #entire set of data
		training = all_data
		# bool_Fv,bool_Fh = (all_data['Fv'] < lowerbound) |(all_data['Fv'] > upperbound), (all_data['Fh']<lowerbound) | (all_data['Fh'] > upperbound)
		# training = all_data[bool_Fh | bool_Fv] #subset of data
		return training.drop(['Fv','Fh'], axis=1).values, training['Fv'].values, training['Fh'].values
	
	msg = "Enter Parameters for Genetic Algorithm"
	title = "Parameter Collection"
	fieldNames = ["Generations", "Population Size", "Chance of Mutation (between 0 and 1)", "Filepath to Input Data"]
	fieldValues = []#['10','50','0.25','0','0','C:\\Users\\sehgals\\Desktop\\Genetic Algorithm for Input Data Selection\\Regression Code\\Regression\\input\\Dummy wing lugs calibration-strains-modifiedmanually - Copy.csv']  # we start with blanks for the values
	fieldValues = multenterbox(msg,title, fieldNames)

	# make sure that none of the fields was left blank
	while 1:
		if fieldValues == None: break
		errmsg = ""
		for i in range(len(fieldNames)):
			if fieldValues[i].strip() == "":
				errmsg = errmsg + ('"%s" is a required field.\n\n' % fieldNames[i])
		if errmsg == "": break # no problems found
		fieldValues = multenterbox(errmsg, title, fieldNames, fieldValues)

	X_data, target1, target2 = same_file(fieldValues[3])

	return int(fieldValues[0]), int(fieldValues[1]), float(fieldValues[2]), X_data, target1, target2

if __name__ == '__main__':
	generations, popsize, chance_of_mutation, X, Y1, Y2 = input_data()
	best_sample = int(math.ceil(0.1*popsize))

	#goal is to give weights to the rows of data to see which rows of data are most useful to the model
	#the fitness score is the rmse (loss fn) given by the model after optimization
	#50 individuals are created
	#each individual is a list with inputdata.shape[0] elements
	#each element is a weight to 3 decimal places

	#INITIALIZATION
	SELECTION = SELECTION_CLASS(best_sample)
	population = INITIALIZATION(popsize, X.shape[0])
	for j in range(generations):

		#EVALUATION
		for i in range(popsize):
			population[i] = FITNESS(X, Y1, X.shape[1], population[i]) #add score as last element
		population = sort_population(population)
		print('Generation ', j)
		print('best individual ', population[0][:-1])
		print('rmse of individual', population[0][-1])
		#SELECTION
		breeders = SELECTION(population)
		population = [] #clear list of old generation elements

		#CROSSOVER
		population = CROSSOVER(breeders, popsize)

		#MUTATION
		population = MUTATION(population, chance_of_mutation)

	#EVALUATION of last generation    
	for k in range(popsize):
		population[k] = FITNESS(X, Y1, X.shape[1], population[k]) #add score as last element
	population = sort_population(population)
	print(population[0])
	#TERMINATION CRITERIA
	print("Max Generations Reached")	
	print("Best Individual saved")
	np.savetxt("Solution_weights.csv", population[0], fmt = '%f', comments = '', delimiter = ',')
	print("Root Mean Squared Error = ", population[0][-1])


# input_ori_data = np.random.uniform(10,100,5)
	# input_ori_target = np.random.uniform(1,10,5)
	# print(input_ori_data.shape, input_ori_target.shape)
	# rows = input_ori_data.shape[0]
	# masked_input = masked_input.reshape((-1,1))

'''
	 Elitism reserves two slots in the next generation for the highest scoring chromosome of the current generation, 
	 without allowing that chromosome to be crossed over in the next generation. In one of those slots, 
	 the elite chromosome will also not be subject to mutation in the next generation.
	'''
# def __init__(self, max_generations, alpha, T_o, threshold):
	# 	self.G = max_generations
	# 	self.alpha = alpha
	# 	self.T_naught = T_o
	# 	self.threshold = threshold
	# def __call__(self, current_generation, pop):

	# 	#ELITISM
	# 	nextGeneration = []
	# 	for i in range(self.best):
	# 		nextGeneration.append(pop.pop(pop[i]))

	# 	#BOLTZMANN SELECTION ALGORITHM
	# 	# rate of selection is controlled by a continuously
	# 	# varying temperature. Initially the temperature is high
	# 	# and selection pressure is inversely proportional to
	# 	# temperature. So selection pressure is low initially.
	# 	# The temperature is decreased gradually which
	# 	# increases the selection pressure. This results in
	# 	# narrowing of search space along with maintaining
	# 	# the diversity in population
	# 	k = (1+100*g/self.G)
	# 	Temp = self.T_naught*(1-self.alpha)**k
	# 	max_fitness = pop[0][-1] #min rmse
	# 	for ind in pop:
	# 		probability = exp((ind[-1]-max_fitness)/Temp)
	# 		if probability > self.threshold:
	# 			nextGeneration.append(ind) 
	# 	return nextGeneration