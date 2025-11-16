"""
Unit Tests for Validators
"""

import pytest
from utils.validators import (
    validate_price,
    validate_size,
    validate_symbol,
    validate_timeframe,
    safe_division
)


class TestPriceValidation:
    """Tests pour validation de prix"""

    def test_valid_price(self):
        """Test prix valide"""
        assert validate_price(40000.0) is True
        assert validate_price(100.5) is True
        assert validate_price(0.00001) is True

    def test_invalid_price_zero(self):
        """Test prix zéro"""
        assert validate_price(0.0) is False

    def test_invalid_price_negative(self):
        """Test prix négatif"""
        assert validate_price(-100.0) is False

    def test_invalid_price_none(self):
        """Test prix None"""
        assert validate_price(None) is False

    def test_price_with_bounds(self):
        """Test prix avec limites"""
        # Prix dans les limites
        assert validate_price(100.0, "test", min_price=10.0, max_price=1000.0) is True

        # Prix en dessous du minimum
        assert validate_price(5.0, "test", min_price=10.0, max_price=1000.0) is False

        # Prix au dessus du maximum
        assert validate_price(2000.0, "test", min_price=10.0, max_price=1000.0) is False


class TestSizeValidation:
    """Tests pour validation de taille"""

    def test_valid_size(self):
        """Test taille valide"""
        assert validate_size(0.01) is True
        assert validate_size(1.0) is True
        assert validate_size(100.0) is True

    def test_invalid_size_zero(self):
        """Test taille zéro"""
        assert validate_size(0.0) is False

    def test_invalid_size_negative(self):
        """Test taille négative"""
        assert validate_size(-0.01) is False

    def test_size_with_bounds(self):
        """Test taille avec limites"""
        assert validate_size(0.05, min_size=0.01, max_size=1.0) is True
        assert validate_size(0.001, min_size=0.01, max_size=1.0) is False
        assert validate_size(10.0, min_size=0.01, max_size=1.0) is False


class TestSymbolValidation:
    """Tests pour validation de symbole"""

    def test_valid_symbol(self):
        """Test symbole valide"""
        assert validate_symbol('BTC/USDT') is True
        assert validate_symbol('ETH/BTC') is True
        assert validate_symbol('DOGE/USDT') is True

    def test_invalid_symbol_no_slash(self):
        """Test symbole sans slash"""
        assert validate_symbol('BTCUSDT') is False

    def test_invalid_symbol_empty(self):
        """Test symbole vide"""
        assert validate_symbol('') is False

    def test_invalid_symbol_wrong_format(self):
        """Test symbole mal formé"""
        assert validate_symbol('BTC/') is False
        assert validate_symbol('/USDT') is False


class TestTimeframeValidation:
    """Tests pour validation de timeframe"""

    def test_valid_timeframes(self):
        """Test timeframes valides"""
        valid_tfs = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
        for tf in valid_tfs:
            assert validate_timeframe(tf) is True

    def test_invalid_timeframe(self):
        """Test timeframe invalide"""
        assert validate_timeframe('10m') is False
        assert validate_timeframe('2h') is False
        assert validate_timeframe('invalid') is False
        assert validate_timeframe('') is False


class TestSafeDivision:
    """Tests pour division sécurisée"""

    def test_normal_division(self):
        """Test division normale"""
        assert safe_division(10, 2) == 5.0
        assert safe_division(100, 4) == 25.0

    def test_division_by_zero(self):
        """Test division par zéro"""
        assert safe_division(10, 0) == 0.0
        assert safe_division(100, 0, default=-1) == -1

    def test_float_division(self):
        """Test division flottante"""
        result = safe_division(1.0, 3.0)
        assert abs(result - 0.333333) < 0.0001
