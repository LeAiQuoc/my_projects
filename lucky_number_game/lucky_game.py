"""
Lucky Number Game

This program implements a number guessing game where players attempt to guess a randomly generated lucky number from a list.
The player's name and birthdate are validated, ensuring only those aged 18+ can play. Players have up to 5 attempts to guess 
the lucky number, with progressively shorter lists provided after incorrect guesses. 

Classes:
---------
1. Player:
    - Stores the player's details, validates their birthdate, and calculates age to ensure eligibility (18+).
    
2. LuckyGame:
    - Generates a random lucky number and manages the game, including list generation and guess handling.

Functions:
----------
1. validate_name(name: str) -> bool:
    - Ensures the player's name is in a valid format.

Main Features:
--------------
- Players are validated based on their birthdate (format: YYYYMMDD).
- Random lucky numbers and lists are generated, and guesses are handled dynamically.
- The game offers instructions, and players can choose to play again or exit.
"""

import random
import re
from datetime import date

class Player:
    def __init__(self, player_name: str, player_birthdate: str) -> None:

        self.player_name = player_name
        self.player_birthdate = player_birthdate

        if not self.validate_birthday():
            raise ValueError('Invalid birthday')

        self.player_age = self.calculate_age()

    def calculate_age(self):
        year = int(self.player_birthdate[0:4])
        month = int(self.player_birthdate[4:6])
        day = int(self.player_birthdate[6:8])
        # Using realtime today's date to calculate age based on birthdate
        today = date.today()
        age = today.year - year - ((today.month, today.day) < (month, day))
        return age

    def validate_birthday(self):
        # Only allow 8 numbers for birthdate 
        if len(self.player_birthdate) != 8:
            print("[E] The birthdate must be in YYYYMMDD format.")
            return False
        # Slice birthdate in 3 parts. Year, month and day. print Error if not numeric.
        try:
            year = int(self.player_birthdate[0:4])
            month = int(self.player_birthdate[4:6])
            day = int(self.player_birthdate[6:8])
            date(year, month, day)
        except ValueError:
            print("[E] The birthdate must consist of numeric values only.")
            return False
        
        # Validate month and day based on the year and month (taking leap years into account)
        # For example the birthdate 20000230 is not valid, since feb have only 29 days 
        if year >= 1900 and month in range(1, 13):
            if month == 2:
                if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                    return day in range(1, 30)
                return day in range(1, 29)
            # Condition to return to the right birthdate taking lengths of days in specific months
            elif month in [4, 6, 9, 11]:
                return day in range(1, 31)
            else:
                return day in range(1, 32)
        return False

    def validate_age(self):
        # Return method if age is over or equal to 18
        return self.calculate_age() >= 18 


class LuckyGame:
    def __init__(self) -> None:
        self.lucky_number = random.randint(0, 100)
        # Return a list of 9 random numbers from a range of 0 to 100
        self.lucky_list = random.sample(range(0, 101), 9)
        # Including the lucky number to the list, now list has 10 elements
        self.lucky_list.append(self.lucky_number)
        self.tries_count = 0

    def generate_lucky_list(self):
        return self.lucky_list

    def shorter_lucky_list(self, player_guess):
        
        # We define the range around the lucky number, not around the player's guess
        min_short_num = max(0, self.lucky_number - 10)
        max_short_num = min(100, self.lucky_number + 10)

        # Iterate over the original lucky list and pick only the numbers within the range
        shorter_list = [num for num in self.lucky_list if min_short_num <= num <= max_short_num]

        # Ensure the guessed number is excluded
        if player_guess in shorter_list:
            shorter_list.remove(player_guess)

        return sorted(shorter_list) 

    # Method for number suffix
    @staticmethod
    def ordinal(n):
        if str(n)[-1] == '1' and n != 11:
            return str(n) + 'st'
        elif str(n)[-1] == '2' and n != 12:
            return str(n) + 'nd'
        elif str(n)[-1] == '3' and n != 13:
            return str(n) + 'rd'
        else:
            return str(n) + 'th'


def validate_name(name):
    # Strip leading/trailing spaces
    stripped_name = name.strip()

    # Split into first and last name, checking that there are exactly two parts
    fl_list = stripped_name.split()
    # Check if length of name is not 2
    if len(fl_list) != 2:
        print("[E] Please provide both a first and last name, separated by a space.")
        return False

    # Regular expression for valid characters (letters, hyphens, apostrophes)
    valid_name_pattern = r"^[A-Za-z'-]+$"
    # Ensure no symbols or other characters in first and last name
    for part in fl_list:
        if not re.match(valid_name_pattern, part):
            print("[E] Invalid name format: Only letters, hyphens, and apostrophes are allowed.")
            return False

    # Ensure there are no extra spaces between first and last name, and join back
    valid_name = " ".join(fl_list)

    # return the properly formatted name if everything is valid
    return valid_name



if __name__ == "__main__":
    while True:
        # Give user the options of typing 1, 2 or 3 only.
        try:
            main_menu = int(input(
                """==========================================
            WELCOME TO LUCKY NUMBER!
    ==========================================

    Please choose one of the following options:

    1. Start the Game
    2. View Rules/Instructions
    3. Exit

    ------------------------------------------
    Enter the number of your choice:
    ------------------------------------------"""))
            if main_menu not in [1,2,3]:
                # handle cases where menu input not in range
                print("[E] Invalid option. Please enter 1,2 or 3.")
                continue
        except ValueError:
            print("[E] Invalid option. Please enter 1,2 or 3")
            continue

        # Main menu options using match-case for better readability
        match main_menu:
            case 1:
                while True:
                    player_name = str(input("[?] Enter your First and Last name: ")).strip()
                    if validate_name(player_name):
                        break
                    else:
                        print("[E] The name provided is not in correct format. Correct format: 'First name' (space) 'Last name'.")

                while True:

                    player_birthday = str(input("[?] Enter your birthday in the format YYYYMMDD: ")).strip()

                     
                    player = Player(player_name, player_birthday)

                    if player.validate_birthday():
                        break
                    else:
                        print("[E] Invalid birthdate format. Please try again.")

                # If all validations passed, we start the game
                if player.validate_age():

                    print(f"Welcome, {player_name}! Let's start the game.")
                    lucky = LuckyGame()
                    max_tries = 5
                    lucky.tries_count = 0
                    print(f"Your lucky list is: {lucky.lucky_list}\n")
                    # Giving the user a maximum of 5 tries of guessing
                    while lucky.tries_count < max_tries:
                            
                        try:
                            player_input = int(input(f"[?] Pick your lucky number: "))
                            # Check if player's input is a valid number between 0 to 100
                            if not 0< player_input <= 100:
                                print(f"[E] Invalid input. The number must be between 0 and 100.")
                                continue
                            # Check if player's input is in the lucky list
                            if player_input not in lucky.lucky_list:
                                print(f"[E] Invalid guess. The number is not in the lucky list.")
                                continue               
                        # Print error if input is not numeric        
                        except ValueError:
                            print("[E] Invalid input. Please enter a number.")
                            continue
                        # If input is lucky number then print congrats message and end game    
                        if player_input == lucky.lucky_number:
                            # Count the number of tries from player
                            lucky.tries_count += 1
                            print(f"[!] Congrats! You guessed the lucky number on your {lucky.ordinal(lucky.tries_count)} try!")
                            break

                        else:
                            # If input is not correct then refer player to shorter lucky list
                            lucky.tries_count += 1
                            shorter_list = lucky.shorter_lucky_list(player_input)
                            # Shuffle the shorter list after each try
                            random.shuffle(shorter_list)

                            print(f"[!] Wrong guess! This is your {lucky.ordinal(lucky.tries_count)} attempt. New shorter list: {shorter_list}")
                        # If exceeding max count, print error message
                        if lucky.tries_count >= max_tries:
                            print(f"[E] Sorry! You've reached the maximum number of tries. The lucky number was {lucky.lucky_number}.")
                    play_again = str(input("Do you want to play again? (y/n): ")).lower()
                    if play_again == 'y':
                        lucky = LuckyGame()
                    elif play_again == 'n':
                        print("Thanks for playing! Exiting the game.")
                        break
                    else: 
                        print("[!] Invalid key, please type 'y' for yes and 'n' for no")

                else:
                    print("[E] You must be 18 years or older to play.")

            case 2:
                instructions = int(input(""" 
Welcome to the Lucky Number Game! Here’s how to play:

Start the Game: Input your full name (first and last) when prompted. Ensure there’s only one space between them.

Enter Birthdate: Provide your birthdate in the format YYYYMMDD. The year must be after 1900.

Age Verification: The game calculates your age. You must be 18 years or older to proceed.

Lucky List Generation: If you pass the age check, the computer generates a list of 9 random integers between 0 and 100.

Guess the Lucky Number: The computer will then generate a lucky number and add it to the list. You’ll have the chance to guess the lucky number from the list.

Win or Try Again: If you guess correctly, you win! If not, you will receive a new list of numbers to guess from, based on your last guess.

Good luck and have fun!
                                         
PRESS 1 TO GO BACK 
"""))
              
            case 3:
                print("Exiting the game. Byebye!")
                break