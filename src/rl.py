import hashlib
import numpy as np
from typing import List
from metawars_api import Unit, simulation, army_cost, \
    UNITY_TYPES, WEAPON_TYPES, ARMOUR_TYPES

# Definiendo el espacio de estados y acciones
action_space_size = len(UNITY_TYPES) * len(WEAPON_TYPES) * len(ARMOUR_TYPES)

print(len(UNITY_TYPES), len(WEAPON_TYPES), len(ARMOUR_TYPES))

q_table = {}  # Diccionario anidado para la tabla Q

# Parámetros del algoritmo
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Almacenar el mejor ejército encontrado
best_army = []
best_reward = 0

def state_to_army(state: int) -> List[Unit]:
    np.random.seed(state)  # Asegurar la reproducibilidad
    army_size = np.random.randint(1, 10)  
    army = []
    for _ in range(army_size):
        unit_type = np.random.choice(UNITY_TYPES)
        weapon = np.random.choice(WEAPON_TYPES)
        armour = np.random.choice(ARMOUR_TYPES)
        level = np.random.randint(1, 10)
        army.append(Unit(unit_type, weapon, armour, level))
    return army

def army_to_state(army: List[Unit]) -> int:
    army_str = ''.join([str(unit) for unit in army])
    state_hash = hashlib.md5(army_str.encode()).hexdigest()
    state = int(state_hash, 16) % (10 ** 8)
    return state

def action_to_unit(action: int) -> Unit:
    np.random.seed(action)  # Asegurar la reproducibilidad
    unit_type = np.random.choice(UNITY_TYPES)
    weapon = np.random.choice(WEAPON_TYPES)
    armour = np.random.choice(ARMOUR_TYPES)
    level = np.random.randint(1, 10)
    return Unit(unit_type, weapon, armour, level)

# Entrenamiento
for episode in range(1000):
    state = np.random.randint(0, 100)
    done = False
    while not done:
        if np.random.rand() < epsilon:
            action = np.random.randint(action_space_size)
        else:
            action = np.argmax([q_table[state].get(a, 0) for a in range(action_space_size)])
        
        unit = action_to_unit(action)
        current_army = state_to_army(state)
        new_army = current_army + [unit]
        
        left_army, right_army = new_army, []
        simulation_result = simulation(left_army, right_army)
        
        reward = sum(simulation_result)
        
        if reward > best_reward:
            best_reward = reward
            best_army = new_army
        
        new_state = army_to_state(new_army)
        
        # Inicializar el estado y la acción en la tabla Q si no están presentes
        q_table.setdefault(state, {})[action] = q_table.get(state, {}).get(action, 0)
        q_table.setdefault(new_state, {})
        
        # Obtener la acción con el máximo Q-value en el nuevo estado
        max_q_new_state = max([q_table[new_state].get(a, 0) for a in range(action_space_size)])
        
        # Actualizar la tabla Q
        q_table[state][action] = q_table[state][action] + \
                                 alpha * (reward + gamma * max_q_new_state - q_table[state][action])
        
        state = new_state
        if army_cost(new_army) > 100000:
            done = True  # Terminar el episodio si el costo del ejército es demasiado alto

def get_best_army():
    # Guardar el mejor ejército en un archivo JSON
    Unit.save_army(best_army, "best_army.json")

# Llamada a la función para guardar el mejor ejército
get_best_army()
