import random
import numpy as np


def genetic_q_learning_knapsack(items, capacity):
    # Inicialización
    population_size = 11
    q_tables = [np.random.rand(len(items), 2) for _ in range(population_size)]

    # Funciones auxiliares
    def evaluate(q_table):
        selected_items = []
        total_weight = 0
        total_value = 0
        for i, item in enumerate(items):
            action = np.argmax(q_table[i])  # 0: no incluir, 1: incluir
            if action == 1 and total_weight + item[0] <= capacity:
                selected_items.append(item)
                total_weight += item[0]
                total_value += item[1]
        return total_value

    def crossover(parent1, parent2):
        crossover_point = np.random.randint(0, len(items))
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def mutate(q_table):
        mutation_point = np.random.randint(0, len(items))
        q_table[mutation_point] = np.random.rand(2)

    def q_learning(q_table, item_index, action, reward):
        alpha = 0.1  # tasa de aprendizaje
        gamma = 0.9  # factor de descuento
        next_max_q_value = np.max(q_table[item_index])
        q_table[item_index, action] = q_table[item_index, action] + alpha * (reward + gamma * next_max_q_value - q_table[item_index, action])

    # Iteración a través de generaciones
    generations = 100
    for _ in range(generations):
        # Evaluación
        fitness_scores = [evaluate(q_table) for q_table in q_tables]
        
        # Selección
        selected_indices = np.argsort(fitness_scores)[-population_size//2:]  # selecciona la mitad superior
        selected_q_tables = [q_tables[i] for i in selected_indices]
        
        # Cruce y Mutación
        new_q_tables = []
        for i in range(0, len(selected_q_tables), 2):
            parent1, parent2 = selected_q_tables[i], selected_q_tables[i+1]
            child1, child2 = crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            new_q_tables.extend([child1, child2])
        
        # Reemplazar la población antigua
        q_tables = new_q_tables
        
    for q_table in q_tables:
        for item_index, item in enumerate(items):
            for action in range(2):  # 0: no incluir, 1: incluir
                # Calcula el peso total de los ítems seleccionados
                total_weight = np.sum([items[i][0] for i in range(len(items)) if np.argmax(q_table[i]) == 1])
                # Calcula la recompensa
                reward = item[1] if action == 1 and total_weight + item[0] <= capacity else 0
                q_learning(q_table, item_index, action, reward)

    # Imprimir la mejor solución encontrada
    best_q_table = q_tables[np.argmax([evaluate(q_table) for q_table in q_tables])]
    best_solution = [np.argmax(best_q_table[i]) for i in range(len(items))]
    return best_solution, evaluate(best_q_table)


def genetic_knapsack(items, capacity):
    # Parámetros
    population_size = 200
    crossover_rate = 0.8
    mutation_rate = 0.02
    generations = 100
    
    def initialize_population():
        # Inicializar una población de soluciones aleatorias
        return [np.random.choice([0, 1], len(items)) for _ in range(population_size)]
    
    def fitness(individual):
        # Calcular el fitness de un individuo
        total_weight = sum(item[0] * individual[i] for i, item in enumerate(items))
        total_value = sum(item[1] * individual[i] for i, item in enumerate(items))
        return total_value if total_weight <= capacity else 0
    
    def selection(population):
        # Seleccionar individuos para la reproducción
        scores = [fitness(ind) for ind in population]
        selected_indices = np.argsort(scores)[-population_size//2:]  # selecciona la mitad superior
        return [population[i] for i in selected_indices]
    
    def crossover(parent1, parent2):
        # Realizar el cruce entre dos padres
        if random.random() < crossover_rate:
            point = random.randint(0, len(items) - 1)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2
        else:
            return parent1, parent2  # No crossover, return parents unmodified
    
    def mutate(individual):
        # Realizar la mutación en un individuo
        for i in range(len(items)):
            if random.random() < mutation_rate:
                individual[i] = 1 - individual[i]  # Flip bit
    
    # Inicializar población
    population = initialize_population()
    
    # Iterar a través de las generaciones
    for _ in range(generations):
        # Selección
        population = selection(population)
        
        # Cruce y Mutación
        new_population = []
        for i in range(0, len(population), 2):
            parent1, parent2 = population[i], population[i+1]
            child1, child2 = crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])
        population = new_population
    
    # Encontrar la mejor solución en la población final
    best_individual = max(population, key=fitness)
    best_value = fitness(best_individual)
    return best_individual.tolist(), best_value


def pos_knapsack(items, capacity):
    # Parámetros
    swarm_size = 200
    generations = 300
    inertia = 0.3
    cognitive_component = 1.4
    social_component = 1.4
    
    def initialize_swarm():
        # Inicializar una población de soluciones aleatorias
        positions = [np.random.choice([0, 1], len(items)) for _ in range(swarm_size)]
        velocities = [np.random.uniform(-1, 1, len(items)) for _ in range(swarm_size)]
        return positions, velocities
    
    def fitness(individual):
        # Calcular el fitness de un individuo
        total_weight = sum(item[0] * individual[i] for i, item in enumerate(items))
        total_value = sum(item[1] * individual[i] for i, item in enumerate(items))
        return total_value if total_weight <= capacity else 0
    
    def update_velocity(velocity, position, personal_best, global_best):
        # Actualizar la velocidad de una partícula
        inertia_term = inertia * velocity
        cognitive_term = cognitive_component * np.random.random() * (personal_best - position)
        social_term = social_component * np.random.random() * (global_best - position)
        new_velocity = inertia_term + cognitive_term + social_term
        return np.clip(new_velocity, -1, 1)  # Mantener la velocidad en el rango [-1, 1]
    
    def update_position(position, velocity):
        # Actualizar la posición de una partícula
        new_position = position + np.round(velocity).astype(int)
        return np.clip(new_position, 0, 1)  # Mantener la posición en el rango {0, 1}
    
    # Inicializar enjambre
    positions, velocities = initialize_swarm()
    personal_bests = positions.copy()
    global_best = max(personal_bests, key=fitness)
    
    # Iterar a través de las generaciones
    for _ in range(generations):
        for i in range(swarm_size):
            # Actualizar velocidad y posición
            velocities[i] = update_velocity(velocities[i], positions[i], personal_bests[i], global_best)
            positions[i] = update_position(positions[i], velocities[i])
            
            # Actualizar los mejores personales y globales
            if fitness(positions[i]) > fitness(personal_bests[i]):
                personal_bests[i] = positions[i]
            if fitness(positions[i]) > fitness(global_best):
                global_best = positions[i]
    
    # Devolver la mejor solución encontrada
    best_value = fitness(global_best)
    return global_best.tolist(), best_value


def knapsack(items, capacity):
    n = len(items)
    # Crear una tabla para almacenar los valores máximos de los subproblemas
    K = [[0 for w in range(capacity + 1)] for i in range(n + 1)]

    # Llenar la tabla en un modo bottom-up
    for i in range(n + 1):
        for w in range(capacity + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif items[i-1][0] <= w:
                K[i][w] = max(items[i-1][1] + K[i-1][w - items[i-1][0]],  K[i-1][w])
            else:
                K[i][w] = K[i-1][w]

    return K[n][capacity]


# Configuración para generar datos de prueba
num_items = 50
max_weight = 100
max_value = 100
max_capacity = 1000

# Generar casos de prueba
test_cases = [
    ([(random.randint(1, max_weight), random.randint(1, max_value)) for _ in range(num_items)], random.randint(1, max_capacity))
    for _ in range(10)
]

# Ejecutar cada caso de prueba con la función knapsack
for i, (items, capacity) in enumerate(test_cases):
    dp = knapsack(items, capacity)
    _, ga_q_learn = genetic_q_learning_knapsack(items, capacity)
    _, ga = genetic_knapsack(items, capacity)
    _, pos = pos_knapsack(items, capacity)
    # q_pso = q_pso_knapsack(items, capacity)

    print(f'DP: {dp}  --  Q-GA: {ga_q_learn}  --  GA: {ga}  --  POS: {pos}\n')
