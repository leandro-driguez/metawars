import gymnasium as gym
import numpy as np

# Inicializa el entorno
env = gym.make("Taxi-v3")
env.reset()
env.render(mode='rgb_array')

# Verifica que el entorno se ha inicializado correctamente
assert isinstance(env, gym.Env), "El entorno no se inicializó correctamente"

# Inicializa la tabla Q
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Parámetros de aprendizaje
alpha = 0.1  # Tasa de aprendizaje
gamma = 0.6  # Factor de descuento
epsilon = 0.1  # Probabilidad de exploración

# Entrenamiento del agente
for i in range(10000):
    state, info = env.reset()
    done = False
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Exploración: elige una acción aleatoria
        else:
            action = np.argmax(q_table[state])  # Explotación: elige la acción con el mayor valor Q
        
        next_state, reward, done, _, info = env.step(action)  # Realiza la acción
        
        # Actualiza la tabla Q
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value
        
        state = next_state


# Prueba del agente entrenado
state, info = env.reset()
env.render()
# env.render(mode='human')  # Especifica el modo de renderizado 'human'
done = False
while not done:
    action = np.argmax(q_table[state])  # Elige la acción con el mayor valor Q
    next_state, reward, done, _, info = env.step(action)
    env.render(mode='human')  # Especifica el modo de renderizado 'human'
    state = next_state
