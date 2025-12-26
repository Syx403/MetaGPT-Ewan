import pytest
from test import calculate_fibonacci


class TestCalculateFibonacci:
    """Test suite for calculate_fibonacci function"""
    
    def test_fibonacci_zero(self):
        """Test Fibonacci with n = 0"""
        assert calculate_fibonacci(0) == []
    
    def test_fibonacci_negative(self):
        """Test Fibonacci with negative n"""
        assert calculate_fibonacci(-1) == []
        assert calculate_fibonacci(-5) == []
    
    def test_fibonacci_one(self):
        """Test Fibonacci with n = 1"""
        assert calculate_fibonacci(1) == [0]
    
    def test_fibonacci_two(self):
        """Test Fibonacci with n = 2"""
        assert calculate_fibonacci(2) == [0, 1]
    
    def test_fibonacci_three(self):
        """Test Fibonacci with n = 3"""
        assert calculate_fibonacci(3) == [0, 1, 1]
    
    def test_fibonacci_ten(self):
        """Test Fibonacci with n = 10"""
        expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        assert calculate_fibonacci(10) == expected
    
    def test_fibonacci_fifteen(self):
        """Test Fibonacci with n = 15"""
        expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
        assert calculate_fibonacci(15) == expected
    
    def test_fibonacci_return_type(self):
        """Test that the function returns a list"""
        assert isinstance(calculate_fibonacci(5), list)
    
    def test_fibonacci_sequence_property(self):
        """Test that each element is the sum of previous two"""
        result = calculate_fibonacci(10)
        for i in range(2, len(result)):
            assert result[i] == result[i-1] + result[i-2]
    
    def test_fibonacci_length(self):
        """Test that the returned list has correct length"""
        assert len(calculate_fibonacci(5)) == 5
        assert len(calculate_fibonacci(8)) == 8
        assert len(calculate_fibonacci(0)) == 0

