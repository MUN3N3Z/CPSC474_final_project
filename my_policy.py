from policy import CribbagePolicy, CompositePolicy, ThrowPolicy, PegPolicy, GreedyThrower, GreedyPegger
from cribbage import Game
from deck import Card, Deck
from typing import List, Tuple
from pegging import Pegging
from itertools import combinations
from json import load
from scoring import score
from generate_schell_table import generate_schell_table
from random import shuffle

# Generate Schell's table 
# schells_table = generate_schell_table()


class StatisticalThrower(ThrowPolicy):
    """ Statistical sampling-based keep/throw agent.
    """
    def __init__(self, game):
        super().__init__(game)
        self.deck = Deck(game.all_ranks(), game.all_suits(), 1)
        self.deck.shuffle()

    def expected_hand_score(self, hand: List[Card], remaining_cards: List[Card]) -> float:
        """ Returns the expected score of a hand based on statistical sampling of turn cards.
            hand - a list of 4 cards
        """
        return sum(score(self._game, hand, card, False)[0] for card in remaining_cards) / len(remaining_cards)

    def keep(self, hand: List[Card], scores: Tuple[int], am_dealer: bool):
        """ Returns a tuple with the best keep/throw pair based on statistical sampling(Schell's Table).
            hand - a list of 6 cards
        """
        possible_throw_pairs = list(combinations(hand, 2))
        with open('schell_table.json', 'r') as file:
            schell_lookup_table = load(file) 
            best_throw_pair = None
            best_score = 0
            final_hand = None
            remaining_cards = [card for card in self.deck._cards if card not in hand]
            for throw_pair in possible_throw_pairs:
                schell_table_keys = list(map(str, sorted([card.rank() for card in throw_pair]))) # Maintain the order of the keys
                keep_cards = [card for card in hand if card not in throw_pair]
                expected_hand_score = self.expected_hand_score(keep_cards, remaining_cards)
                if am_dealer:
                    # Dealer attempts to maximize the score of the cards sent to the crib
                    hand_and_crib_score = expected_hand_score
                    + schell_lookup_table[f'{schell_table_keys[0]},{schell_table_keys[1]}']['dealer']
                else:
                    # Non-dealer attempts to minimize the score of the cards sent to the crib
                    hand_and_crib_score = expected_hand_score
                    - schell_lookup_table[f'{schell_table_keys[0]},{schell_table_keys[1]}']['non-dealer']
                if best_score is None or hand_and_crib_score > best_score:
                    best_score = hand_and_crib_score
                    best_throw_pair = list(throw_pair)
                    final_hand = keep_cards
            return final_hand, best_throw_pair


class RuleBasedPegger(PegPolicy):
    """ Rule-based pegging agent.
    """
    def __init__(self, game: Game):
        super().__init__(game)
        self._card_weights = {
            'score_points': 1,
            'lead_low_card': 0.3,
            'lead_sum_to_15': 0.4,
            'closest_to_31': 0.01,
            'save_ace': -0.1,
            'play_ace': 0.2,
            'penalize_5': -0.1,
            'illegal_play': -1,
        }

    def peg(self, cards: List[Card], history: Pegging, turn: Card, scores: Tuple[int], am_dealer: bool):
        """ Returns the best pegging card based on the current history.
        """
        if not cards:
            return None
        card_scores = dict()
        shuffle(cards) # Shuffle cards to break ties randomly
        for card in cards:
            score = 0
            card_score = history.score(self._game, card, 0 if am_dealer else 1)
            if card_score is not None:
                score += self._card_weights['score_points'] * card_score
                if history.is_start_round() and card.rank() <= 4:
                    score += self._card_weights['lead_low_card']
                for other_card in cards: # see if combination sum can be playd i.e. <= 31
                    if other_card != card and card.rank() + other_card.rank() == 15 and history.total_points() + card.rank()  + (2 * other_card.rank()) <= 31:
                        score += self._card_weights['lead_sum_to_15']
                        break
                if history.total_points() + card.rank() >= 22 and history.total_points() + card.rank() <= 31:
                    score += self._card_weights['closest_to_31'] * (history.total_points() + card.rank()) # Penalize lower ranks
                if card.rank() == 1:
                    score += self._card_weights['play_ace'] if history.total_points() >= 29 else self._card_weights['save_ace']
                if card.rank() == 5 and history.total_points() + card.rank() < 15:
                    score += self._card_weights['penalize_5'] # Penalize playing a 5 too early
            else:
                score += self._card_weights['illegal_play']
            card_scores[card] = score
        
        # break ties by selecting card with lower rank
        best_card = max(card_scores, key=lambda card: (card_scores[card], -card.rank()))
        return_card = best_card if card_scores[best_card] != self._card_weights['illegal_play'] else None
        return return_card

class MyPolicy(CribbagePolicy):
    def __init__(self, game):
        self._policy = CompositePolicy(game, StatisticalThrower(game), RuleBasedPegger(game))

    def keep(self, hand, scores, am_dealer):
        return self._policy.keep(hand, scores, am_dealer)

    def peg(self, cards, history, turn, scores, am_dealer):
        return self._policy.peg(cards, history, turn, scores, am_dealer)



    

                                    
 