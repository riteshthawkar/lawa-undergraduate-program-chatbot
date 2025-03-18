# Tests for lawa-rag-agent

This directory contains tests for the lawa-rag-agent application.

## Running the Tests

You can run all tests using the `run_tests.py` script in the root directory:

```bash
python3 run_tests.py
```

Or you can run individual test files:

```bash
python3 -m unittest tests/test_query_rewriting.py
python3 -m unittest tests/test_integration.py
```

## Test Files

- `test_query_rewriting.py`: Tests for the query rewriting agent
- `test_integration.py`: Integration tests for the application flow

## Writing New Tests

When writing new tests, follow these guidelines:

1. Create a new test file in the `tests` directory
2. Import the necessary modules from the application
3. Use the `unittest` framework for writing tests
4. For async tests, use the `asyncio.run()` function to run the coroutine
5. Use mocks for external dependencies like OpenAI API calls

Example:

```python
import unittest
import asyncio
from unittest.mock import patch, MagicMock

from modules.some_module import some_function

class TestSomeModule(unittest.TestCase):
    def test_some_function(self):
        # Test code here
        result = some_function()
        self.assertEqual(result, expected_result)
        
    @patch('modules.some_module.external_dependency')
    async def test_async_function(self, mock_dependency):
        # Mock setup
        mock_dependency.return_value = mock_result
        
        # Test code here
        result = await async_function()
        self.assertEqual(result, expected_result)
```

## Test Coverage

To check test coverage, install the `coverage` package:

```bash
pip install coverage
```

Then run the tests with coverage:

```bash
coverage run -m unittest discover
coverage report
```

This will show you which parts of the code are covered by tests and which are not. 