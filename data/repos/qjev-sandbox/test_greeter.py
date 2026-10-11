# Run with: python3 -m unittest discover -s data/repos/qjev-sandbox -p "test_*.py" -v
# (or: python3 -m unittest data/repos/qjev-sandbox/test_greeter.py -v)
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from greeter import Greeter


class GreeterTest(unittest.TestCase):
    def test_greets_world_by_default(self):
        self.assertEqual(Greeter().greet(), "Hello, world!")

    def test_greets_by_name(self):
        self.assertEqual(Greeter("Ada").greet(), "Hello, Ada!")

    def test_strips_blank_names(self):
        self.assertEqual(Greeter("   ").greet(), "Hello!")

    def test_shouts_the_greeting(self):
        self.assertEqual(Greeter("Ada").shout(), "HELLO, ADA!")

    def test_whispers_the_greeting(self):
        self.assertEqual(Greeter("Ada").whisper(), "hello, ada!")

    def test_farewells_always(self):
        self.assertEqual(Greeter().farewell(), "Goodbye!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
