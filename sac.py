"""
To see Tensorboard:

tensorboard --logdir ./models
"""

import os
import yaml
import shutil
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CallbackList,
    EvalCallback,
)

from poker_env import PokerEnv
from utils.custom_callbacks import SelfPlayCallback


# setup some params
train = True
continue_training = False
training_timestamps = 10000000
current_model_version = "v16"

learning_starts = 50000
model_path = f"models/sac/{current_model_version}/3500000"

# create envs
with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
env = PokerEnv(config=config["sac-six-player"], debug=False)
eval_env = Monitor(PokerEnv(config=config["sac-six-player"]))

if train:
    # create callbacks
    log_path = f"models/sac/{current_model_version}/{training_timestamps}_log"
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_path,
        log_path=log_path,
        n_eval_episodes=100,
        eval_freq=3000,
        deterministic=False,
        render=False,
    )

    event_callback = SelfPlayCallback(
        log_path + "/last_model.zip", rolling_starts=learning_starts, n_steps=200000
    )
    callbacks = CallbackList([eval_callback, event_callback])

    # save current config
    s = "config.yaml"
    d = f"models/sac/{current_model_version}/config.yaml"
    if not os.path.exists(d):
        os.mkdir(f"models/sac/{current_model_version}")
        shutil.copy(s, d)

    # check for transfer learning
    if not continue_training:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_starts=learning_starts,
            learning_rate=0.00007,
            train_freq=(5, "episode"),
            tensorboard_log=f"models/sac/{current_model_version}/{training_timestamps}_tensorboard",
        )
    else:
        model: SAC = SAC.load(model_path, env=env)
        # optional: save replay buffer to continue train the model (caution: it's large!)
        # model.load_replay_buffer(
        #     f"models/sac/{current_model_version}/{training_timestamps}_replay_buffer"
        # )

    # train the model
    model.learn(
        total_timesteps=training_timestamps,
        log_interval=3000,
        callback=callbacks,
        reset_num_timesteps=True,
    )

    # save the model and replay buffer
    model.save(f"models/sac/{current_model_version}/{training_timestamps}")
    model.save_replay_buffer(
        f"models/sac/{current_model_version}/{training_timestamps}_replay_buffer"
    )

else:
    # test the model (deprecated, add agent in config.yaml and use poker_env.py instead)
    model = SAC.load(model_path, env=env)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action, get_all_rewards=True)
        if done:
            obs = env.reset()

            print(
                f"\nprev chips: {tuple(env.previous_chips.values())}",
                f"\nchips: {tuple([x.chips for x in env.game.players])}",
                f"\nsum chips: {sum(tuple([x.chips for x in env.game.players]))}",
                f"\nreward: {tuple(reward.values())}",
                f"\nwinners: {info}",
                f"\nbuyin history: {tuple(env.game.total_buyin_history.values())}",
                f"\nrestart times: {env.game.game_restarts}",
                "\n",
            )
