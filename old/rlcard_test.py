import rlcard
from rlcard.agents import RandomAgent

DEFAULT_GAME_CONFIG = {
    "game_num_players": 3,
}

env = rlcard.make("limit-holdem", config=DEFAULT_GAME_CONFIG)
env.set_agents([RandomAgent(num_actions=env.num_actions)])

print(env.num_actions)  # 2
print(env.num_players)  # 1
print(env.state_shape)  # [[2]]
print(env.action_shape)  # [None]

env.reset()
# print(env.run())

while True:
    obs = env.step(1)
    print(obs, env.get_payoffs())
