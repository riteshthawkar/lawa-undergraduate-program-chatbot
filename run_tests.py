#!/usr/bin/env python3
"""
Test runner script for the lawa-rag-agent.
Run this script to execute all tests.
"""

import unittest
import os
import sys

# Ensure the tests directory exists
if not os.path.exists('tests'):
    os.makedirs('tests')
    # Create an empty __init__.py file in the tests directory
    with open('tests/__init__.py', 'w') as f:
        pass

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Discover and run all tests
if __name__ == '__main__':
    # Discover all tests in the tests directory
    test_suite = unittest.defaultTestLoader.discover('tests')
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with non-zero code if tests failed
    sys.exit(not result.wasSuccessful()) 