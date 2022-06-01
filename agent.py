from __future__ import annotations

import random
import numpy as np
from typing import TYPE_CHECKING
from stable_baselines3 import SAC, TD3, PPO, A2C


from engine.game.action_type import ActionType

if TYPE_CHECKING:
    from poker_env import PokerEnv
    from engine.game.game import TexasHoldEm


class RandomAgent:
    def __init__(self, game: TexasHoldEm, player_id: int = None):
        self.player_id = player_id
        self.game = game

    def calculate_action(self, *arg, **kwargs):
        while 1:
            rand = random.random()
            val = None
            if rand < 0.15:
                action = ActionType.FOLD
            elif rand < 0.3:
                action = ActionType.CHECK
            elif rand < 0.4:
                action = ActionType.CALL
            else:
                action = ActionType.RAISE
                val = (
                    random.randint(2, 15)
                    + self.game.player_bet_amount(self.player_id)
                    + self.game.chips_to_call(self.player_id)
                )

                if val >= self.game.players[self.player_id].chips:
                    action = ActionType.ALL_IN
                    val = None

            if self.game.validate_move(self.player_id, action, val):
                break
        return action, val


class RLAgent:
    def __init__(
        self,
        env: PokerEnv,
        path: str,
        algorithm: str,
        num_to_action: dict = None,
        player_id: int = None,
    ):
        self.player_id = player_id
        self.game = env.game
        self.num_to_action = num_to_action
        if algorithm == "sac":
            self.model = SAC.load(
                path,
                env=env,
            )
        else:
            raise NotImplementedError
        # print(self.model.num_timesteps)
        
    def calculate_action(self, observations=None):
        if observations is None:
            raise ValueError("Observations is required for RL Agent")
        action, _states = self.model.predict(observations, deterministic=True)
        action, val = action
        action = self.num_to_action[round(action)]
        val = round(val)


        if action == ActionType.RAISE:
            val += self.game.player_bet_amount(
                self.player_id
            ) + self.game.chips_to_call(self.player_id)
            val = round(val)
            if val >= self.game.players[self.player_id].chips:
                action = ActionType.ALL_IN
                val = None
        else:
            val = None

        if not self.game.validate_move(self.player_id, action, val):
            action = (
                ActionType.CHECK
                if self.game.validate_move(self.player_id, ActionType.CHECK, None)
                else ActionType.FOLD
            )
            val = None

        return action, val


class CrammerAgent:
    def __init__(
        self,
        game: TexasHoldEm,
        player_id: int = None,
        num_to_action: dict = None,
        alpha=2,
        beta=50,
    ):
        self.player_id = player_id
        self.game = game

        if num_to_action is None:
            num_to_action = {
                0: ActionType.CALL,
                1: ActionType.RAISE,
                2: ActionType.CHECK,
                3: ActionType.FOLD,
            }
        self.num_to_action = num_to_action

        self.alpha = alpha
        self.beta = beta

    def calculate_action(self, *arg, **kwargs):
        # board = self.game.board
        # community_cards = self.game.community_cards
        # all_hands = self.game.hands
        # hand_phase = self.game.hand_phase
        # hand_history = self.game.hand_history
        # big_blind = self.game.big_blind
        curr_player_id = self.game.current_player
        curr_player_chips = self.game.players[curr_player_id].chips

        # Find all possible actions.
        # Possible actions should always include FOLD and RAISE.
        # CHECK and CALL may sometimes be invalid.
        # I don't believe CHECK and CALL can ever be simultaneously possible.
        possible_actions = []  # 3 in length.
        for action in self.num_to_action.values():
            if action.name == "RAISE":
                val = int((self.game.big_blind + curr_player_chips) / 2)
            else:
                val = None

            if self.game.validate_move(curr_player_id, action, val):
                possible_actions.append(action.name)

            # All other hand phases besides PREFLOP.
            # Okay, so during the FLOP, we know the 3 community cards.
            # Our agent will also know everyone else's cards.

            # Let's first evaluate everyone's hands. This list will be our
            # "odds" calculator for how likely it is for our CrammerAgent to win.
            # This dict can vary based on which players are active in the current
            # game.

        # player_odds:
        # {3: 3117, 4: 6530, 5: 6316, 0: 3522, 1: 6585, 2: 6528}

        hand_score = self.game.player_hand_scores

        ids = list(hand_score.keys())
        odds = list(hand_score.values())

        # Player ID with highest odds of winning.
        possible_winner_id = ids[odds.index(min(odds))]
        difference = abs(
            hand_score[possible_winner_id] - hand_score[curr_player_id]
        )  # [0, 7461]

        # Hmmm, we want to have it return some bet and val
        # given the odds of winning out of all active players.
        # We can assume there will always be >= 2 players at this time
        # point.
        # We assume the current player is an active player.
        if difference == 0:  # Our current player has the highest odds of winning.
            rand = random.random()
            val = None

            # We definitely don't want to fold (unlikely).
            if rand < 0.05:
                action = ActionType.FOLD
            elif rand < 0.4:
                if "CALL" in possible_actions:
                    action = ActionType.CALL
                else:
                    action = ActionType.CHECK
            else:
                action = ActionType.RAISE
                proportion = np.random.beta(self.alpha, self.beta, size=1)
                chips_bet = int(proportion * curr_player_chips)
                chips_to_call = self.game.chips_to_call(curr_player_id)

                raised_level = self.game._get_pot(
                    self.game.players[curr_player_id].last_pot
                ).raised
                val = max(
                    chips_to_call + self.game.player_bet_amount(curr_player_id) + 1,
                    self.game.big_blind + raised_level,
                    min(chips_bet, curr_player_chips),
                )

        else:
            # We need to check how large this difference is. Depending on its size, we need to act accordingly.
            # The temp is near 1 if the difference between curr player odds and best player odds is small.
            # The temp is near 0 if the difference is large.

            var = np.random.uniform(0.1, 1.5) * difference

            temp = 1 - (
                max(0, min(7461, (difference + var))) / (7461 + 1)
            )  # [near 1 if difference is smaller, near 0 if difference is bigger].
            temp *= np.random.uniform(0.9, 1)
            val = None

            # print("HAND SCORES:", hand_score)
            # print("VAR:", var)
            # print("DIFFERENCE:", difference)
            # print(temp)

            # Maybe we can try a more complex decision maker here.
            # log_diff = max(np.log10(1/difference), 2)  # We will find the magnitude of it.
            # # difference:
            # # [0, 1] -> within 0 and 10 -> 1
            # # [1, 2] -> within 10 and 100
            # # [2, 3] -> within 100 and 1000
            # # [3, 4] -> within 1000 and 10000

            if temp < 0.82:  # If the difference is 800 or greater (up to 7461).
                action = ActionType.FOLD
            elif temp < 0.95:  # If the difference is between 5 and 800.
                if "CALL" in possible_actions:
                    action = ActionType.CALL
                else:
                    action = ActionType.CHECK
            else:  # If the difference is less than 5 (down to 0).
                action = ActionType.RAISE
                proportion = np.random.beta(self.alpha, self.beta, size=1)
                chips_bet = int(proportion * curr_player_chips)
                chips_to_call = self.game.chips_to_call(curr_player_id)

                raised_level = self.game._get_pot(
                    self.game.players[curr_player_id].last_pot
                ).raised
                val = max(
                    chips_to_call + self.game.player_bet_amount(curr_player_id) + 1,
                    self.game.big_blind + raised_level,
                    min(chips_bet, curr_player_chips),
                )
        # # Debugging.
        # if not self.game.validate_move(curr_player_id, bet, val):
        #     print(
        #         "INVALID MOVE: ",
        #         self.game.hand_phase.name,
        #         curr_player_id,
        #         bet.name,
        #         val,
        #     )
        # else:
        #     print(self.game.hand_phase, curr_player_id, bet, val)

        if action == ActionType.RAISE:
            """
            ADDED THIS PREVIOUS POT COMMIT TO AVOID INVALID MOVE:
            """
            # print(
            #     self.game.chips_to_call(curr_player_id),
            #     self.game.player_bet_amount(curr_player_id),
            # )

            # val += self.game.player_bet_amount(
            #     curr_player_id
            # ) + self.game.chips_to_call(curr_player_id)

            if val >= curr_player_chips:
                action = ActionType.ALL_IN
                val = None

        return action, val
