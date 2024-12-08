from deck import Deck, Card
from cribbage import Game
from typing import Tuple, List
from random import sample, choice
from scoring import greedy_throw, score
import json
from collections import defaultdict

def estimate_crib_value(game: Game, deck: Deck, throw_pair: Tuple[int, int], dealer: bool, simulations=20000) -> float:
    """ Estimate the value of a crib based on the throw pair and whether the 
        crib belongs to the dealer or non-dealer.
        
        throw_pair -- a tuple of two integers representing card ranks
        dealer -- a boolean flag indicating whether the crib belongs to the dealer
    """
    crib_score = 0
    for _ in range(simulations):
        deck.shuffle()
        opponent_hand = deck.peek(6)
        deck.shuffle()
        cut_card = deck.peek(1)[0]
        if dealer:
            # Opponent is not dealer, thus attempts to minimize the score of the cards sent to the crib
            # i.e. maximize the score of the cards kept
            keep_cards, throw_cards, cards_score = greedy_throw(game, opponent_hand, -1)
            cards_in_crib = list(throw_pair) + throw_cards
            crib_score += score(game, cards_in_crib, cut_card, True)[0]
        else:
            # Dealer attempts to maximize the score of the cards sent to the crib
            # i.e. maximize the score of both the cards kept and the cards thrown
            keep_cards, throw_cards, card_score = greedy_throw(game, opponent_hand, 1)
            cards_in_crib = list(throw_pair) + throw_cards

            crib_score += score(game, cards_in_crib, cut_card, True)[0]
    return crib_score / simulations

def generate_schell_table() -> dict:
    """ Generate a table that estimates the value of a crib based on whether 
        the crib belongs to the dealer or non-dealer. 
    """
    game = Game()
    schell_table = defaultdict(dict)
    for i in range(1, 14):
        for j in range(i, 14):
            deck = Deck(game.all_ranks(), game.all_suits(), 1)
            throw_pair = (Card(i, sample(game.all_suits(), 1)[0]), Card(j, sample(game.all_suits(), 1)[0]))
            deck.remove(throw_pair)
            schell_table[f'{i},{j}'] = {
                'dealer': estimate_crib_value(game, deck, throw_pair, True),
                'non-dealer': estimate_crib_value(game, deck, throw_pair, False)
            }
    return schell_table

if __name__ == '__main__':
    with open('schell_table.json', 'w') as f:
        json.dump(generate_schell_table(), f, indent=4)
    print('Schell table generated and saved to schell_table.json')