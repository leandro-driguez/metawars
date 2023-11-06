import numpy as np

class MultiArmedBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.true_arm_values = np.random.rand(n_arms)

    def pull_arm(self, arm):
        return 1 if np.random.random() < self.true_arm_values[arm] else 0

class EpsilonGreedyAgent:
    def __init__(self, n_arms, epsilon):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.arm_values = np.zeros(n_arms)
        self.arm_counts = np.zeros(n_arms)

    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_arms)
        else:
            return np.argmax(self.arm_values)

    def update_values(self, arm, reward):
        self.arm_counts[arm] += 1
        self.arm_values[arm] += (reward - self.arm_values[arm]) / self.arm_counts[arm]

# Ejemplo de uso:
n_arms  = 100
epsilon = 0.01
bandit  = MultiArmedBandit(n_arms)
agent   = EpsilonGreedyAgent(n_arms, epsilon)

# Imprime el mejor valor de las máquinas
best_arm_value = max(bandit.true_arm_values)
best_arm_index = np.argmax(bandit.true_arm_values)
print(f'El mejor valor de las máquinas es {best_arm_value} en el brazo {best_arm_index}')

n_steps = 100_000
for step in range(n_steps):
    arm = agent.select_arm()
    reward = bandit.pull_arm(arm)
    agent.update_values(arm, reward)

# Después de 10K iteraciones, realiza 30 iteraciones adicionales
correct_selections = 0
additional_steps = 30
for step in range(additional_steps):
    arm = agent.select_arm()
    if arm == best_arm_index:
        correct_selections += 1
    reward = bandit.pull_arm(arm)
    agent.update_values(arm, reward)

# if correct_selections == 0:
# print(agent.__dict__, bandit.__dict__)

print(f'Después de 10K iteraciones, el agente seleccionó el mejor brazo {correct_selections} veces de {additional_steps} iteraciones adicionales.')
