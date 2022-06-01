from random import shuffle
from itertools import product


def hand_generator(num_players=10):
    suits = ["c", "d", "h", "s"]
    ranks = ["2", "3", "4", "5", "4", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]

    cards = list(set(r + s for r, s in product(ranks, suits)))
    shuffle(cards)

    community = cards[:5]
    num_players_hands = []
    for i in range(num_players):
        num_players_hands.append(cards[5 + i * 2 : 7 + i * 2])

    return community, num_players_hands


if __name__ == "__main__":
    print(hand_generator())
    # result: (['6S', 'KC', '2S', '5S', '3D'], [['KD', '7S']])
