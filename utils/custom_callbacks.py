from stable_baselines3.common.callbacks import BaseCallback

from agent import RLAgent


class SelfPlayCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, model_path: str, verbose=0, n_steps=20000, rolling_starts=50000):
        super(SelfPlayCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

        self.model_path = model_path
        self.rolling_starts = rolling_starts
        self.n_steps = n_steps

        # use 4 agent slots for self-play
        self.rolling_dict = {1: 2, 2: 3, 3: 4, 4: 1}
        self.rolling_id = 1

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        # reference PokerEnv: self.training_env.envs[0].env

        if (
            self.num_timesteps >= self.rolling_starts
            and self.num_timesteps % self.n_steps == 0
        ):
            self.model.save(self.model_path)
            agent = RLAgent(
                self.training_env.envs[0].env,
                self.model_path,
                "sac",
                self.training_env.envs[0].env.num_to_action,
                self.rolling_id,
            )
            self.training_env.envs[0].env.update_opponent(self.rolling_id, agent)
            self.rolling_id = self.rolling_dict[self.rolling_id]
            
            logging_dict = {}
            for k, v in self.training_env.envs[0].env.opponents.items():
                if isinstance(v, RLAgent):
                    logging_dict[k] = f"{v.model.num_timesteps}"
            # print(
            #     "agent timesteps:",
            #     agent.model.num_timesteps,
            #     logging_dict,
            # )

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        # close the environment
        self.training_env.close()
