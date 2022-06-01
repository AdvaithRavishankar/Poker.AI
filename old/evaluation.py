from treys import Card
from treys import Evaluator

board = [Card.new("Jc"), Card.new("2c"), Card.new("6s"), Card.new("3c"), Card.new("8s")]
hand = [Card.new("7c"), Card.new("Jh")]
Card.new("9h")
evaluator = Evaluator()


print(Card.print_pretty_cards(board + hand))

# evaluator.evaluate(board, hand)
score = evaluator.evaluate(board, hand)

print(7463 - score, evaluator.class_to_string(evaluator.get_rank_class(score)))
