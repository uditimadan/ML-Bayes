from unittest import TestCase
from nb import *

class Testmy_naive_bayes(TestCase):
    pass

class Testmy_naive_bayes(TestCase):

    def test_fit(self):
        # Sample data

        nb = my_naive_bayes("breast-cancer.data")
        examples = [
            [10, "no", 0, 1, 1, 3, 1, 1, 1, "no"],
            [12, "yes", 0, 1, 1, 2, 1, 1, 1, "no"]
        ]
        classifications = ["recurrence", "no-recurrence"]

        nb.fit(examples, classifications)

        # Expected counts
        expected_pos_age_10 = 1
        expected_neg_age_12 = 1

        # Assert pos_examples and neg_examples for various features and values
        self.assertEqual(nb.pos_examples["age"][10], expected_pos_age_10)
        self.assertEqual(nb.neg_examples["age"][12], expected_neg_age_12)


    def test_predict(self):
        nb = my_naive_bayes("breast-cancer.data")
        examples = [[1, 'premeno', '30-34', '0-2', 'no', '3', 'right', 'left_low', 'no'],
                    [2, 'ge40', '20-24', '0-2', 'no', '1', 'left', 'left_up', 'no'],
                    [3, 'ge40', '20-24', '0-2', 'no', '1', 'right', 'left_low', 'no']]
        classifications = ['recurrence', 'no-recurrence', 'recurrence']

        nb.fit(examples, classifications)

        # Predict for a known example
        test_example = [1, 'premeno', '30-34', '0-2', 'no', '3', 'right', 'left_low', 'no']
        prediction = nb.predict([test_example])

        # Check if the prediction is correct
        self.assertEqual(prediction, ['recurrence'])

    def test_score(self):
        nb = my_naive_bayes("breast-cancer.data")
        # Identical lists
        predicted1 = ['recurrence', 'recurrence', 'no-recurrence']
        actual1 = ['recurrence', 'recurrence', 'no-recurrence']
        self.assertEqual(nb.score(predicted1, actual1), 1.0)

        # Completely different lists
        predicted2 = ['recurrence', 'recurrence', 'no-recurrence']
        actual2 = ['no-recurrence', 'no-recurrence', 'recurrence']
        self.assertEqual(nb.score(predicted2, actual2), 0.0)

        # Mostly correct lists
        predicted3 = ['recurrence', 'recurrence', 'no-recurrence']
        actual3 = ['recurrence', 'no-recurrence', 'recurrence']
        self.assertAlmostEqual(nb.score(predicted3, actual3), 0.5, places=6)


