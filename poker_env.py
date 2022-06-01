from typing import Union

import yaml
import pprint
import numpy as np
from copy import deepcopy
from gym import spaces, Env

from gui import Window

from engine import TexasHoldEm
from engine.game.game import Player
from engine.gui.text_gui import TextGUI
from engine.game.hand_phase import HandPhase
from engine.game.action_type import ActionType
from engine.game.history import PrehandHistory
from engine.evaluator.evaluator import evaluate
from engine.game.player_state import PlayerState
from agent import RandomAgent, CrammerAgent, RLAgent
from utils.flatten import flatten_spaces, flatten_array


class PokerEnv(Env):
    # for agent training
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 120}

    def __init__(self, config: dict = None, debug: bool = False, gui: TextGUI = None):
        # poker

        with open("config.yaml") as f:
            cf = yaml.load(f, Loader=yaml.FullLoader)
            env_constants = cf["environment-constants"]
            if config is None:
                config = cf["normal-six-player"]

        self.buy_in = config["stack"]
        self.small_blind = config["small-blind"]
        self.big_blind = config["big-blind"]
        self.num_players = config["players"]
        self.opponent_config = config["opponents"]
        self.buyin_limit = config["buyin_limit"]
        self.agent_id = config["agent_id"]  # id for our own agent
        self.gui = gui
        self.use_gui = self.gui is not None
        if self.use_gui:
            self.gui.set_player_ids([self.agent_id])

        self.max_value = self.buy_in * self.num_players * (self.buyin_limit + 1) + 1
        self.debug = debug
        self.total_steps = 0
        self.total_episodes = 0

        # dictionary constants
        self.previous_action_dict = {
            x: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0} for x in range(self.num_players)
        }
        self.action_dict = deepcopy(self.previous_action_dict)
        self.num_to_action = {
            0: ActionType.CALL,
            1: ActionType.RAISE,
            2: ActionType.CHECK,
            3: ActionType.FOLD,
            4: ActionType.ALL_IN,
        }
        self.action_to_num = {
            ActionType.CALL: 0,
            ActionType.RAISE: 1,
            ActionType.CHECK: 2,
            ActionType.FOLD: 3,
            ActionType.ALL_IN: 4,
        }
        self.suit_to_int = {
            "s": 1,  # spades
            "h": 2,  # hearts
            "d": 3,  # diamonds
            "c": 4,  # clubs
        }
        self.card_num_to_int = {"T": 9, "J": 10, "Q": 11, "K": 12, "A": 13}

        self.game = TexasHoldEm(
            buyin=self.buy_in,
            buyin_limit=self.buyin_limit,
            big_blind=self.big_blind,
            small_blind=self.small_blind,
            max_players=self.num_players,
            agent_id=self.agent_id,
            add_chips_when_lose=False,
            num_to_action=self.num_to_action,
        )

        # step function
        self.current_agent_action = None
        self.previous_chips = {}
        self.default_pot_commit = {x.player_id: 0 for x in self.game.players}
        self.total_player_winnings = self.default_pot_commit.copy()
        self.total_reward = self.default_pot_commit.copy()

        # gym environment
        self.spec = None
        self.num_envs = 1

        # reward
        self.reward_multiplier = env_constants["reward_multiplier"]
        self.winner_reward_multiplier = env_constants[
            "winner_reward_multiplier"
        ]  # amplify winner's reward to encourage winning

        self.reward_range = np.array([-1, 1])
        # action space
        self.action_space = spaces.Box(
            np.array([0, 0]), np.array([3, self.max_value - 1]), (2,), dtype=np.int32
        )

        # observation space
        self.max_hand_score = env_constants["max_hand_score"]
        self.hand_score_multiplier = env_constants["hand_score_multiplier"]
        self.hand_score_reward_multiplier = env_constants[
            "hand_score_reward_multiplier"
        ]
        card_space = spaces.Tuple((spaces.Discrete(14), spaces.Discrete(5)))
        obs_space = spaces.Dict(
            {
                "actions": spaces.Tuple(
                    (
                        spaces.Tuple(
                            (
                                spaces.Discrete(5, start=-1),  # -1 for padding
                                spaces.Discrete(self.max_value),
                            )
                        ),
                    )
                    * self.num_players
                ),  # all opponents actions
                "active": spaces.MultiBinary(self.num_players),  # [0, 1, 1, 0, 1, 0]
                "chips": spaces.Tuple(
                    (spaces.Discrete(self.max_value),) * self.num_players
                ),  # every player's chips
                "community_cards": spaces.Tuple((card_space,) * 5),  # ((1, 2)) * 5
                "player_card": spaces.Tuple((card_space,) * 2),  # ((3, 2)) * 2
                "max_raise": spaces.Discrete(self.max_value),  # player.chips
                "min_raise": spaces.Discrete(self.big_blind),
                "pot": spaces.Discrete(self.max_value),  # pot.amount
                "player_stacks": spaces.Tuple(
                    (spaces.Discrete(self.max_value),) * self.num_players
                ),  # pot_commits for every player in the whole game
                "stage_bettings": spaces.Tuple(
                    (spaces.Discrete(self.max_value),) * self.num_players
                ),  # pot_commits for every player in the current stage
                "hand_score": spaces.Discrete(
                    self.max_hand_score * self.hand_score_multiplier
                ),  # our hand's score
            }
        )

        self.observation_space = flatten_spaces(obs_space)

        # opponents
        self.opponents: dict[int, Union[RandomAgent, CrammerAgent]] = {}
        self.add_opponent()

    def add_opponent(self):
        # temporarily adding all random agents
        opponents = []

        rl_agent: dict = self.opponent_config["rl-agent"]
        for name, paths in rl_agent.items():
            for path in paths:
                opponents.append(RLAgent(self, path, name, self.num_to_action))

        random_agent = self.opponent_config["random-agent"]
        for _ in range(random_agent):
            opponents.append(RandomAgent(self.game))

        crammer_agent = self.opponent_config["crammer-agent"]
        for _ in range(crammer_agent):
            opponents.append(CrammerAgent(self.game, self.num_to_action))

        reserved = False
        for i in range(len(opponents)):
            if i == self.agent_id:
                reserved = True
            if not reserved:
                self.opponents[i] = opponents[i]
                self.opponents[i].player_id = i
            else:
                self.opponents[i + 1] = opponents[i]
                self.opponents[i + 1].player_id = i + 1

    def update_opponent(
        self, player_id: int, agent: Union[RandomAgent, CrammerAgent, RLAgent]
    ):
        self.opponents[player_id] = agent

    def card_to_observation(self, card):
        card = list(str(card))
        if self.card_num_to_int.get(card[0], 0):
            card[0] = self.card_num_to_int[card[0]]
        else:
            card[0] = int(card[0]) - 1
        card[1] = self.suit_to_int[card[1]]
        return tuple(card)

    def clip(self, value, _min=-1, _max=1, round_to=3):
        return round(max(min(value, _max), _min), round_to)

    def get_winners(self):
        if self.game.hand_history[HandPhase.SETTLE]:
            winners = self.game.hand_history[HandPhase.SETTLE].pot_winners

            # player_id: (winning_chips, score, pot_id)
            return {x[1][2][0]: (x[1][0], x[1][1], x[0]) for x in winners.items()}
        return None

    def get_player_hand_score(self, player_id=None, percentage=False):
        if player_id is None:
            player_id = self.agent_id

        hand_score = self.max_hand_score // 2
        if self.game.hand_phase != HandPhase.PREHAND and self.game.players[
            player_id
        ].state not in (PlayerState.OUT, PlayerState.SKIP):
            hand_score = self.max_hand_score - evaluate(
                self.game.hands[player_id], self.game.board
            )
        # print(
        #     "hs:",
        #     hand_score,
        #     player_id,
        #     self.game.hand_phase,
        #     self.game.hands.get(player_id, 0),
        #     self.game.players[player_id].state,
        # )

        if percentage:
            return hand_score / (self.max_hand_score // 2) - 1
        return round(hand_score * self.hand_score_multiplier)

    def get_pot_commits(self):
        pot_commits = self.default_pot_commit.copy()
        stage_pot_commits = self.default_pot_commit.copy()

        for pot in self.game.pots:
            player_amount = pot.player_amounts_without_remove
            stage_amount = pot.player_amounts

            for player_id in player_amount:
                if player_id in pot_commits:
                    pot_commits[player_id] += player_amount[player_id]
                else:
                    pot_commits[player_id] = player_amount[player_id]

                if player_id in stage_pot_commits:
                    stage_pot_commits[player_id] += stage_amount.get(player_id, 0)
                else:
                    stage_pot_commits[player_id] = (
                        stage_amount[player_id] if stage_amount.get(player_id, 0) else 0
                    )
        return pot_commits, stage_pot_commits

    def get_reward(self, pot_commits: dict):
        # calculate the total number of pot commits for each player
        player_active_dict = {
            x.player_id: x.state not in (PlayerState.OUT, PlayerState.SKIP)
            for x in self.game.players
        }

        # calculate the payouts
        payouts = {
            pot_commit[0]: -1 * pot_commit[1] * (not player_active_dict[pot_commit[0]])
            for pot_commit in pot_commits.items()
        }

        # calculate winners
        winners = self.get_winners()

        # everyone but one folded
        if sum(player_active_dict.values()) == 1:
            pot_total = sum(list(map(lambda x: x.amount, self.game.pots)))
            payouts = {
                player_id: payouts[player_id]
                + player_active_dict[player_id] * (pot_total - pot_commits[player_id])
                for player_id in player_active_dict.keys()
            }

        # if last street played and still multiple players active or everyone all-in
        elif (not self.game.is_hand_running() and not self.game._is_hand_over()) or (
            sum(player_active_dict.values()) > 1 and winners is not None
        ):
            for player_id, payout in payouts.items():
                # if player folded earlier
                if payout != 0:
                    payouts[player_id] = payout

                # if player wins
                elif winners.get(player_id) is not None:
                    payouts[player_id] = (
                        self.game.players[player_id].chips
                        - self.previous_chips[player_id]
                    )  # this game chips - last game chips

                # if player stay to the end and loses
                else:
                    payouts[player_id] = -pot_commits[player_id]

        # record total winnings
        if self.game.hand_phase == HandPhase.PREHAND:
            for pid, chips in payouts.items():
                self.total_player_winnings[pid] += chips

        # calculate percentage of the player's stack
        percent_payouts = {}
        for player in self.game.players:
            player_id = player.player_id

            # if player lost all chips
            if player.chips == 0 and not self.game.is_hand_running():
                percent_payouts[player_id] = -1
            else:
                # calculate raw reward
                payout_percentage = payouts[player_id] / (
                    self.previous_chips[player_id] + 0.001
                )
                if payout_percentage > 0:
                    payout_percentage *= self.winner_reward_multiplier

                # # calculate player's hand score
                # hand_score_reward = (
                #     self.get_player_hand_score(player_id=player_id, percentage=True)
                #     * self.hand_score_reward_multiplier
                # )

                # calculate punishment agent if fold with good hands
                fold_punishment = 0
                if player_id == self.agent_id:
                    # agent_hand_score = (
                    #     hand_score_reward / self.hand_score_reward_multiplier
                    # )

                    if self.current_agent_action[0] == ActionType.FOLD:
                        # if agent_hand_score > 0:
                        #     fold_punishment = agent_hand_score
                        if self.current_agent_action[2] == HandPhase.PREFLOP:
                            fold_punishment = 0.2

                # clip raw reward to -1 to 1
                payout = self.clip(
                    payout_percentage
                    * self.reward_multiplier
                    # - hand_score_reward
                    - fold_punishment
                )

                # calculate hand score diff
                diff = 0.5 - (
                    (
                        (
                            self.game.player_hand_scores[self.agent_id]
                            - min(self.game.player_hand_scores.values())
                        )
                        / self.max_hand_score
                    )
                    * 2
                )

                # add score diff to reward
                percent_payouts[player_id] = self.clip(
                    payout + ((payout - diff) * self.hand_score_reward_multiplier)
                )
                # print(
                #     self.game.player_hand_scores,
                #     diff,
                #     (payout - diff),
                #     payout,
                #     percent_payouts[player_id],
                # )

        return percent_payouts

    def get_observations(
        self, current_player_id: int, pot_commits: dict, stage_pot_commits: dict
    ):
        # actions
        current_round_history = self.game.hand_history[self.game.hand_phase]
        if self.game.hand_phase == HandPhase.PREHAND:
            current_round_history = self.game.hand_history.get_last_history()
        # prefill action list
        actions = [(-1, 0)] * self.num_players
        if not isinstance(current_round_history, PrehandHistory):

            for player_action in current_round_history.actions:
                actions[player_action.player_id] = (
                    self.action_to_num[player_action.action_type],  # action
                    player_action.value
                    if player_action.value is not None
                    else 0,  # value
                )

        # active + chips
        active = [0] * self.num_players
        chips = [0] * self.num_players
        winners = None
        # if last street played and still multiple players active
        if not self.game.is_hand_running() and not self.game._is_hand_over():
            winners = self.game.hand_history[HandPhase.SETTLE].pot_winners
            for winner_id in winners.keys():
                active[winner_id] = 1

        for x in self.game.players:
            if winners is None:
                active[x.player_id] = int(
                    x.state not in (PlayerState.OUT, PlayerState.SKIP)
                )
            chips[x.player_id] = x.chips

        # community cards
        community_cards = [(0, 0)] * 5
        for i in range(len(self.game.board)):
            card = self.game.board[i]
            community_cards[i] = self.card_to_observation(card)

        # player hand
        player_card = ((0, 0),) * 2
        if self.game.hand_phase != HandPhase.PREFLOP:
            player_card = [
                self.card_to_observation(x) for x in self.game.hands[current_player_id]
            ]

        # update observations
        return np.array(
            flatten_array(
                [
                    tuple(actions),
                    tuple(active),
                    tuple(chips),
                    tuple(community_cards),
                    player_card,
                    self.game.players[current_player_id].chips,
                    self.game.pots[0].raised,
                    sum([x.amount for x in self.game.pots]),
                    tuple(pot_commits.values()),
                    tuple(stage_pot_commits.values()),
                    self.get_player_hand_score(),
                ]
            )
        )

    def step(self, action, format_action=True, get_all_rewards=False):
        action, val = action
        current_player: Player = list(
            filter(
                lambda x: x.player_id == self.game.current_player,
                self.game.players,
            )
        )[0]

        # convert action to ActionType
        if not isinstance(action, ActionType):
            action = round(action)
            action = self.num_to_action[action]

        # add raise value to the prev commits
        if action == ActionType.RAISE:
            if format_action:
                val += self.game.player_bet_amount(
                    current_player.player_id
                ) + self.game.chips_to_call(current_player.player_id)
            val = round(val)

            # translate action to ALL_IN
            if val >= current_player.chips:
                action = ActionType.ALL_IN
                val = None
        else:
            val = None

        # check valid action
        if not self.game.validate_move(current_player.player_id, action, val):
            action = (
                ActionType.CHECK
                if self.game.validate_move(
                    current_player.player_id, ActionType.CHECK, None
                )
                else ActionType.FOLD
            )
            val = None

        if self.debug:
            print(
                f"{str(self.game.hand_phase)[10:]}: Player {self.game.current_player}, Chips: {self.game.players[self.game.current_player].chips}, Action - {str(action)[11:].capitalize()}{f': {val}' if val else ''}"
            )
        if self.use_gui:
            self.gui.print_state(self.game)
            self.gui.print_action(self.game.current_player, action, val)

        # agent take action
        self.current_agent_action = (action, val, self.game.hand_phase)
        self.action_dict[self.game.current_player][self.action_to_num[action]] += 1
        self.game.take_action(action, val)

        done = not self.game.is_hand_running()

        # Take the other agent actions (and values) in the game.
        while self.game.current_player != self.agent_id and not done:
            observations = None
            if isinstance(self.opponents[self.game.current_player], RLAgent):
                observations = self.get_observations(
                    self.game.current_player, *self.get_pot_commits()
                )
            action, val = self.opponents[self.game.current_player].calculate_action(
                observations
            )
            if self.debug:
                print(
                    f"{str(self.game.hand_phase)[10:]}: Player {self.game.current_player}, Chips: {self.game.players[self.game.current_player].chips}, Action - {str(action)[11:].capitalize()}{f': {val}' if val else ''}"
                )

            if self.use_gui:
                self.gui.print_action(self.game.current_player, action, val)

            # opponent take action
            self.action_dict[self.game.current_player][self.action_to_num[action]] += 1
            self.game.take_action(action, val)
            done = not self.game.is_hand_running()

        if self.use_gui:
            self.gui.print_state(self.game)

        # observations
        pot_commits, stage_pot_commits = self.get_pot_commits()
        observation = self.get_observations(
            current_player.player_id, pot_commits, stage_pot_commits
        )

        # reward + info
        reward = self.get_reward(pot_commits)

        if done:
            for i, rew in reward.items():
                self.total_reward[i] += rew

        if not get_all_rewards:
            reward = reward[self.agent_id]

        info = {"winners": self.get_winners()}

        self.total_steps += 1
        if self.total_steps % 3000 == 0:

            d = {
                kv1[0]: kv1[1] - kv2[1]
                for kv1, kv2 in zip(
                    self.action_dict[self.agent_id].items(),
                    self.previous_action_dict[self.agent_id].items(),
                )
            }
            print(
                f"\nactions in 3k steps: {d}",
                f"\nactions total: {self.action_dict[self.agent_id]}",
            )

            self.previous_action_dict = deepcopy(self.action_dict)

        return observation, reward, done, info

    def render(self):
        pass

    def reset_game(self):
        self.game.reset_game()
        for x in self.game.players:
            self.previous_chips.update({x.player_id: x.chips})

    def reset(self):
        # update previous chips
        self.current_agent_action = None
        for x in self.game.players:
            self.previous_chips.update({x.player_id: x.chips})

        # initiate game engine
        while not self.game.is_hand_running():
            self.game.start_hand()
            if not self.game.is_game_running():
                self.reset_game()

        if self.use_gui:
            self.gui.print_state(self.game)

        # take opponent actions in the game
        while self.game.current_player != self.agent_id:
            observations = None
            if isinstance(self.opponents[self.game.current_player], RLAgent):
                observations = self.get_observations(
                    self.game.current_player, *self.get_pot_commits()
                )
            action, val = self.opponents[self.game.current_player].calculate_action(
                observations
            )

            if self.debug:
                print(
                    f"{str(self.game.hand_phase)[10:]}: Player {self.game.current_player}, Chips: {self.game.players[self.game.current_player].chips}, Action - {str(action)[11:].capitalize()}{f': {val}' if val else ''}"
                )
            if self.use_gui:
                self.gui.print_action(self.game.current_player, action, val)

            self.action_dict[self.game.current_player][self.action_to_num[action]] += 1
            self.game.take_action(action, val)

            while not self.game.is_hand_running():
                self.game.start_hand()
                if not self.game.is_game_running():
                    self.reset_game()

        if self.use_gui:
            self.gui.print_state(self.game)

        # calculate + return information
        current_player: Player = list(
            filter(
                lambda x: x.player_id == self.game.current_player,
                self.game.players,
            )
        )[0]

        observation = self.get_observations(
            current_player.player_id, *self.get_pot_commits()
        )

        self.total_episodes += 1

        return observation

    def close(self):
        # some cleanups
        print("Environment is closing...")

        print(
            f"\nprev chips: {tuple(self.previous_chips.values())}",
            f"\nchips: {tuple([x.chips for x in self.game.players])}",
            f"\ntotal chips in game: {sum(tuple([x.chips for x in self.game.players]))}",
            f"\ntotal winnings: {tuple(self.total_player_winnings.values())}",
            f"\ntotal reward:   {tuple(map(lambda x: round(x, 1), self.total_reward.values()))}",
            f"\nbuyin history: {tuple(self.game.total_buyin_history.values())}",
            f"\nrestart times: {self.game.game_restarts}",
            f"\nactions:\n{pprint.pformat(self.action_dict)}",
            "\n",
        )


def main(n_games=1, show_gui=True):
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # gui = TextGUI()
    

    poker = PokerEnv(config=config["sac-six-player"], debug=n_games <= 50, gui=None)
    agent = CrammerAgent(poker.game)
    agent.player_id = poker.agent_id

    # gui commands
    if(show_gui):
        gui = Window(poker)
        gui.start_state()


    # reset environment
    obs = poker.reset()

    # start step loop
    games_to_play = 0
    while 1:
        action, val = agent.calculate_action()
        #action, val = gui.accept_input()
        while not poker.game.validate_move(poker.game.current_player, action, val):
            print(f"{action} {val} is not valid for player {poker.game.current_player}")
            action, val = gui.accept_input()

        obs, reward, done, info = poker.step(
            (action, val), format_action=False, get_all_rewards=True
        )

        if done:
            if n_games <= 50:
                print(
                    f"\nprev chips: {tuple(poker.previous_chips.values())}",
                    f"\nchips: {tuple([x.chips for x in poker.game.players])}",
                    f"\nsum chips: {sum(tuple([x.chips for x in poker.game.players]))}",
                    f"\nreward: {tuple(reward.values())}",
                    f"\nhand scores: {tuple([poker.max_hand_score - x for x in poker.game.player_hand_scores.values()])}",
                    f"\nwinners: {info}",
                    f"\nbuyin history: {tuple(poker.game.total_buyin_history.values())}",
                    f"\nrestart times: {poker.game.game_restarts}",
                    "\n",
                )

            games_to_play += 1
            if games_to_play >= n_games:
                break
            obs = poker.reset()

    # include best response (https://aipokertutorial.com/agent-evaluation/)
    # probability response (compare best response winnings divided by agent winnings)

    # use player's hand score ranking + total winnings in the reward function (try to compare with best response)
    """
    score : {1: 3201, 2: 301, 3: 5031, 4: 1242...}
    max: 0, min: 7462 
    chips won: (-1522, -1000, -1141, -1195, 155, 590) 
    """
    poker.close()


if __name__ == "__main__":
    # --- about 2s for 1000 games (0.002s / game) --- #
    import pstats
    import cProfile

    with cProfile.Profile() as pr:
        main(n_games=1000)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats("profile.prof")
    # run `snakeviz profile.prof` to see stats
