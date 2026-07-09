import unittest
from unittest.mock import patch
from project2.lucky_game import LuckyGame, validate_name


class TestLuckyGame(unittest.TestCase):

    def setUp(self):
        self.game = LuckyGame()

    @patch('project2.lucky_game.random.randint')
    @patch('project2.lucky_game.random.sample')
    def test_init(self, mock_sample, mock_randint):
        mock_randint.return_value = 50
        mock_sample.return_value = [10, 20, 30, 40, 60, 70, 80, 90, 100]
        
        game = LuckyGame()
        
        self.assertEqual(game.lucky_number, 50)
        self.assertEqual(len(game.lucky_list), 10)
        self.assertIn(50, game.lucky_list)
        self.assertEqual(game.tries_count, 0)
        
        mock_randint.assert_called_once_with(0, 100)
        mock_sample.assert_called_once_with(range(0, 101), 9)

    def test_generate_lucky_list(self):
        result = self.game.generate_lucky_list()
        self.assertEqual(result, self.game.lucky_list)
        self.assertEqual(len(result), 10)
        self.assertIn(self.game.lucky_number, result)

    def test_shorter_lucky_list(self):
        self.game.lucky_number = 50
        self.game.lucky_list = [10, 20, 30, 40, 45, 50, 55, 60, 70, 80]
        
        # Test with player guess not in range
        player_guess = 10
        shorter_list = self.game.shorter_lucky_list(player_guess)
        
        # Should include numbers within range [40, 60] around lucky_number 50
        expected_numbers = [40, 45, 50, 55, 60]
        # Player guess (10) should be excluded if it were in range
        self.assertEqual(sorted(shorter_list), expected_numbers)
        self.assertNotIn(player_guess, shorter_list)

    def test_shorter_lucky_list_with_guess_in_range(self):
        self.game.lucky_number = 50
        self.game.lucky_list = [10, 20, 30, 40, 45, 50, 55, 60, 70, 80]
        
        # Test with player guess in range - should be excluded
        player_guess = 45
        shorter_list = self.game.shorter_lucky_list(player_guess)
        
        # Should include numbers within range [40, 60] except player_guess
        expected_numbers = [40, 50, 55, 60]
        self.assertEqual(sorted(shorter_list), expected_numbers)
        self.assertNotIn(player_guess, shorter_list)

    def test_shorter_lucky_list_edge_cases(self):
        # Test with lucky number at lower bound
        self.game.lucky_number = 5
        self.game.lucky_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15]
        
        shorter_list = self.game.shorter_lucky_list(20)  # Guess not in range
        # Range should be [0, 15], all numbers from lucky_list in this range
        expected_numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15]
        self.assertEqual(sorted(shorter_list), expected_numbers)

        # Test with lucky number at upper bound
        self.game.lucky_number = 95
        self.game.lucky_list = [85, 86, 87, 88, 89, 90, 91, 92, 93, 95, 100]
        
        shorter_list = self.game.shorter_lucky_list(10)  # Guess not in range
        # Range should be [85, 100]
        expected_numbers = [85, 86, 87, 88, 89, 90, 91, 92, 93, 95, 100]
        self.assertEqual(sorted(shorter_list), expected_numbers)

    def test_ordinal(self):
        # Test regular ordinals
        self.assertEqual(LuckyGame.ordinal(1), '1st')
        self.assertEqual(LuckyGame.ordinal(2), '2nd')
        self.assertEqual(LuckyGame.ordinal(3), '3rd')
        self.assertEqual(LuckyGame.ordinal(4), '4th')
        self.assertEqual(LuckyGame.ordinal(5), '5th')
        
        # Test teens (special cases)
        self.assertEqual(LuckyGame.ordinal(11), '11th')
        self.assertEqual(LuckyGame.ordinal(12), '12th')
        self.assertEqual(LuckyGame.ordinal(13), '13th')
        
        # Test higher numbers
        self.assertEqual(LuckyGame.ordinal(21), '21st')
        self.assertEqual(LuckyGame.ordinal(22), '22nd')
        self.assertEqual(LuckyGame.ordinal(23), '23rd')
        self.assertEqual(LuckyGame.ordinal(24), '24th')
        
        # Test larger numbers
        self.assertEqual(LuckyGame.ordinal(101), '101st')
        self.assertEqual(LuckyGame.ordinal(102), '102nd')
        self.assertEqual(LuckyGame.ordinal(103), '103rd')


class TestValidateName(unittest.TestCase):

    def test_valid_names(self):
        # Test valid name formats
        self.assertEqual(validate_name("John Doe"), "John Doe")
        self.assertEqual(validate_name("Mary Jane"), "Mary Jane")
        self.assertEqual(validate_name("Jean-Pierre Martin"), "Jean-Pierre Martin")
        self.assertEqual(validate_name("O'Connor Smith"), "O'Connor Smith")
        
        # Test with extra spaces (should be stripped)
        self.assertEqual(validate_name("  John   Doe  "), "John Doe")

    def test_invalid_names(self):
        # Test single name
        self.assertFalse(validate_name("John"))
        self.assertFalse(validate_name(""))
        
        # Test more than two names
        self.assertFalse(validate_name("John Middle Doe"))
        
        # Test with numbers
        self.assertFalse(validate_name("John123 Doe"))
        self.assertFalse(validate_name("John Doe2"))
        
        # Test with special characters (except allowed ones)
        self.assertFalse(validate_name("John@ Doe"))
        self.assertFalse(validate_name("John Doe!"))
        self.assertFalse(validate_name("John.Doe Smith"))
        
        # Test empty or whitespace only
        self.assertFalse(validate_name(""))
        self.assertFalse(validate_name("   "))

    @patch('builtins.print')
    def test_error_messages(self, mock_print):
        # Test error message for single name
        validate_name("John")
        mock_print.assert_called_with("[E] Please provide both a first and last name, separated by a space.")
        
        # Test error message for invalid characters
        validate_name("John123 Doe")
        mock_print.assert_called_with("[E] Invalid name format: Only letters, hyphens, and apostrophes are allowed.")


if __name__ == "__main__":
    unittest.main()