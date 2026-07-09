from unittest.mock import patch
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import unittest
from project2.lucky_game import Player

class TestPlayer(unittest.TestCase):

    def setUp(self):
        self.valid_name = 'John Doe'
        self.invalid_name = '123 John'
        self.valid_player_birthday = '19920904'
        self.invalid_player_birthday = 'xx324456'
        self.valid_player_age = 20
        self.invalid_player_age = 15
        # Add a valid birthday for someone under 18
        self.under_18_birthday = '20100501'

    @patch('project2.lucky_game.date')
    def test_init(self, mock_date):
        mock_today = date(2025, 9, 16)
        mock_date.today.return_value = mock_today

        # Create the player here, with the patch active
        player = Player(self.valid_name, self.valid_player_birthday)

        # Test player name
        self.assertEqual(player.player_name, self.valid_name)

        # Test player birthday
        self.assertEqual(player.player_birthdate, self.valid_player_birthday)

        # Calculate expected age for valid birthdate using relativedelta
        birth_date = datetime.strptime(self.valid_player_birthday, "%Y%m%d").date()
        age_diff = relativedelta(mock_today, birth_date)
        expected_age = age_diff.years
        
        # Assert that the calculated age matches the expected age
        self.assertEqual(player.player_age, expected_age)
    
    def test_invalid_birthday(self):
        # Test invalid player birthday
        with self.assertRaises(ValueError):
            invalid_player = Player(self.valid_name, self.invalid_player_birthday)

    @patch('project2.lucky_game.date')  # Mock the date class
    def test_calculate_age(self, mock_date):
        # Mock today's date
        mock_today = date(2025, 9, 16)
        mock_date.today.return_value = mock_today

        # Parse birthdate for expected age calculation
        birth_date = datetime.strptime(self.valid_player_birthday, "%Y%m%d").date()
        age_diff = relativedelta(mock_today, birth_date)
        expected_age = age_diff.years

        # Create the player and test the calculated age
        player = Player(self.valid_name, self.valid_player_birthday)
        self.assertEqual(player.calculate_age(), expected_age)
        self.assertEqual(player.player_age, expected_age)

    def test_validate_birthday(self):
        # Test valid birthdate (8 digits) - this should work
        valid_player = Player(self.valid_name, self.valid_player_birthday)
        self.assertEqual(len(valid_player.player_birthdate), 8)
        self.assertTrue(valid_player.validate_birthday())   
        
        # Test invalid birthdate length
        test_player = Player(self.valid_name, self.valid_player_birthday)
        test_player.player_birthdate = '199209'  
        self.assertFalse(test_player.validate_birthday())  
        
        # Test invalid birthdate with non-numeric characters
        test_player.player_birthdate = '1992-09-04'  
        self.assertFalse(test_player.validate_birthday()) 
        
        # Test invalid birthdate with invalid month
        test_player.player_birthdate = '19921301'  
        self.assertFalse(test_player.validate_birthday()) 
        
        # Test invalid birthdate with invalid day
        test_player.player_birthdate = '19920230'  
        self.assertFalse(test_player.validate_birthday())  
        
        # Test invalid year
        test_player.player_birthdate = '00000000'  
        self.assertFalse(test_player.validate_birthday())  
        
        # Test valid leap year
        valid_leap_year = Player(self.valid_name, '20200229')
        self.assertTrue(valid_leap_year.validate_birthday())  

    @patch('project2.lucky_game.date')
    def test_validate_age_over_18(self, mock_date):
        # Simulate current date
        mock_date.today.return_value = date(2025, 9, 16)
        
        # Player who is over 18
        player = Player(self.valid_name, self.valid_player_birthday)
        
        # Test if validate_age returns True for players over 18
        self.assertTrue(player.validate_age())

    @patch('project2.lucky_game.date')
    def test_validate_age_under_18(self, mock_date):
        # Simulate current date
        mock_date.today.return_value = date(2025, 9, 16)
        
        # Player who is under 18
        player = Player(self.valid_name, self.under_18_birthday)
        
        # Test if validate_age returns False for players under 18
        self.assertFalse(player.validate_age())



if __name__ == "__main__":
    unittest.main()