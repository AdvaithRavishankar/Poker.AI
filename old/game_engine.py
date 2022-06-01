from engine import TexasHoldEm
from engine.game.action_type import ActionType


def accept_input(game, turn, player):
    args = input(
        f"Player {player} turn {turn} chips {game.players[game.current_player].chips}:"
    )

    if " " in args:
        action_str, val = args.split()
    else:
        action_str, val = args, 0
    action_str = action_str.lower()

    if action_str == "call":
        return ActionType.CALL, None
    elif action_str == "fold":
        return ActionType.FOLD, None
    elif action_str == "all-in":
        return ActionType.ALL_IN, None
    elif action_str == "raise":
        return ActionType.RAISE, float(val)
    elif action_str == "check":
        return ActionType.CHECK, None
    else:
        # always invalid
        return ActionType.RAISE, -1

def main():
    # create a game, but use our own cards
    game = TexasHoldEm(buyin=500, big_blind=5, small_blind=2, max_players=3)

    while game.is_game_running():
        game.start_hand()
        while game.is_hand_running():
            lines = []
            for i in range(len(game.pots)):
                lines.append(
                    f"Pot {i}: {game.pots[i].get_total_amount()} Board: {game.board}"
                )

            action, val = accept_input(game.hand_phase, game.current_player)
            while not game.validate_move(game.current_player, action, val):
                print(f"{action} {val} is not valid for player {game.current_player}")
                action, val = accept_input(game.hand_phase, game.current_player)

            game.take_action(action, val)
            # print(lines, game.hand_history)
            
if __name__ == "__main__":
    main()
