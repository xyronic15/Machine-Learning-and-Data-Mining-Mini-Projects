from os import popen
import random

P_MUTATION = 50
POP_SIZE = 4
CROSSOVER = 4

def generate_start_pop():

    population = []
    for i in range(POP_SIZE):
        child = []
        for j in range(8):
            child.append(random.randint(0,7))
        population.append(child)
    return population

def get_parents(population):

    fitness_levels = []

    # get fitness levels of each chromosome
    for i in range(len(population)):
        fit = calc_fitness(population[i])
        fitness_levels.append(fit)
    
    # get indices of top two fitness levels
    top_2_inds = sorted(range(len(fitness_levels)), key=lambda x: fitness_levels[x])[-2:]
    
    father_info = (population[top_2_inds[0]], fitness_levels[top_2_inds[0]])
    mother_info = (population[top_2_inds[1]], fitness_levels[top_2_inds[1]])

    # return the chromosome and fitness levels of the parents
    return father_info, mother_info

def calc_fitness(child):

    clashes = 0

    # check if in the same row
    for i in range(8):
        for j in range(i+1, 8):
            if child[i] == child[j]:
                clashes += 1
    
    # check if clashes in the diagonal using gradient
    for i in range(8):
        for j in range(i+1, 8):
            if abs(child[j] - child[i]) == abs(j - i):
                clashes += 1

    # 28 is the maximum number of unique clashes
    # highest fitness is 28
    return 28 - clashes

def generate_new_population(father, mother):

    population = []

    # iterate until populations size met
    while len(population) < POP_SIZE:

        # generate randint to decide if crossover or population
        selector = random.randint(1,100)

        if selector > P_MUTATION:
            # crossover
            child1, child2 = crossover(father, mother)
            population.extend((child1, child2))
        else:
            child = mutation(father, mother)
            population.append(child)
        
    # check if population is larger than POP_SIZE
    if len(population) > POP_SIZE:
        population = population[:POP_SIZE]

    return population

def crossover(father, mother):

    child1 = father[:CROSSOVER]
    child1.extend(mother[CROSSOVER:])
    child2 = mother[:CROSSOVER]
    child2.extend(father[CROSSOVER:])

    return child1, child2

def mutation(father, mother):

    # randomly choose parent, column and row to change for child
    parent = random.randint(0,1)
    index = random.randint(0,7)
    value = random.randint(0,7)

    if parent == 0:
        child = father[:]
        child[index] = value
    else:
        child = mother[:]
        child[index] = value
    
    return child

def view_solutions(solutions):

    i = 1

    for solution in solutions:

        print("Solution " + str(i) +":")
        print(str(solution) + "\n")

        for y in range(8, -1, -1):
            for x in range(0,8):
                if solution[x] == y:
                    print('[Q]', end='')
                else:
                    print('[ ]', end='')
            print()
    
        print("\n")
        i+=1

def write_solutions(solutions):

    with open('solutions.txt', 'w+') as f:
        for solution in solutions:
            f.write(str(solution) + '\n')
        f.close

def main():

    # Ask user how many solutions they would like
    num_solutions = int(input("How many solutions would you like?:\n"))

    solutions = []

    # 1. Generate initial population
    population = generate_start_pop()
    # print("Initial population: " + str(population))

    generation_count = 1

    # Iterate to desired num of solutions
    while len(solutions) < num_solutions:
    # 2. Calculate the fitness of each chromosome + 3. select parents for mating

        if generation_count % 10000 == 0:
            print("Generation number " + str(generation_count))

        father_info, mother_info = get_parents(population)

        # check if either father/mother is a solution and add to solutions if it is
        if father_info[1] == 28:
            if father_info[0] not in solutions:
                print(str(len(solutions) + 1) + ":   " + str(father_info[0]) + "    Fitness level: " + str(father_info[1]))
                solutions.append(father_info[0])
        if mother_info[1] == 28:
            if mother_info[0] not in solutions:
                print(str(len(solutions) + 1) + ":   " + str(mother_info[0]) + "    Fitness level: " + str(mother_info[1]))
                solutions.append(mother_info[0])
        
        # 4/5/6/7. Create offspring using crossover and mutation
        population = generate_new_population(father_info[0], mother_info[0])

        generation_count += 1

    print("Total number of generations: " + str(generation_count))

    view_option = input("Would you like to view the solutions on a board? (y/n): ")
    if view_option.lower() == "y":
        view_solutions(solutions)
    write_solutions(solutions)



if __name__ == '__main__':
    main()
    