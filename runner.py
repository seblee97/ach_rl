import minigrid
import q_learner


class Runner:
    def __init__(self):
        self._environment = minigrid.MiniGrid(size=(2, 2), starting_xy=[0, 0])

        self._learner = q_learner.TabularQLearner(
            action_space=self._environment.action_space,
            state_space=self._environment.state_space,
            policy="epsilon_greedy",
            epsilon=0.01,
            alpha=0.1,
            gamma=0.9,
        )

    def train(self, num_steps: int):
        state = self._environment.agent_position
        total_rewards = 0
        for _ in range(num_steps):
            action = self._learner.select_action(state)
            reward, new_state = self._environment.step(action)
            self._learner.step(state, action, reward, new_state)
            state = new_state
            total_rewards += reward

        print(total_rewards)
