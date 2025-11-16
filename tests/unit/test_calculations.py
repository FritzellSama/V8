"""
Unit Tests for Calculation Utilities
"""

import pytest
import numpy as np
import pandas as pd
from utils.calculations import (
    calculate_pnl,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_atr,
    calculate_position_size,
    calculate_kelly_criterion,
    safe_divide
)


class TestPnLCalculations:
    """Tests pour calculs de P&L"""

    def test_pnl_long_profit(self):
        """Test P&L LONG avec profit"""
        pnl, pnl_pct = calculate_pnl(
            entry_price=40000.0,
            exit_price=42000.0,
            size=0.01,
            side='long'
        )

        assert pnl == pytest.approx(20.0, rel=0.01)  # (42000-40000) * 0.01
        assert pnl_pct == pytest.approx(5.0, rel=0.01)  # 5% gain

    def test_pnl_long_loss(self):
        """Test P&L LONG avec perte"""
        pnl, pnl_pct = calculate_pnl(
            entry_price=40000.0,
            exit_price=38000.0,
            size=0.01,
            side='long'
        )

        assert pnl == pytest.approx(-20.0, rel=0.01)
        assert pnl_pct == pytest.approx(-5.0, rel=0.01)

    def test_pnl_short_profit(self):
        """Test P&L SHORT avec profit"""
        pnl, pnl_pct = calculate_pnl(
            entry_price=40000.0,
            exit_price=38000.0,
            size=0.01,
            side='short'
        )

        assert pnl == pytest.approx(20.0, rel=0.01)  # Prix baisse = profit
        assert pnl_pct == pytest.approx(5.0, rel=0.01)

    def test_pnl_short_loss(self):
        """Test P&L SHORT avec perte"""
        pnl, pnl_pct = calculate_pnl(
            entry_price=40000.0,
            exit_price=42000.0,
            size=0.01,
            side='short'
        )

        assert pnl == pytest.approx(-20.0, rel=0.01)
        assert pnl_pct == pytest.approx(-5.0, rel=0.01)

    def test_pnl_zero_movement(self):
        """Test P&L sans mouvement de prix"""
        pnl, pnl_pct = calculate_pnl(
            entry_price=40000.0,
            exit_price=40000.0,
            size=0.01,
            side='long'
        )

        assert pnl == 0.0
        assert pnl_pct == 0.0


class TestTechnicalIndicators:
    """Tests pour indicateurs techniques"""

    @pytest.fixture
    def price_series(self):
        """Série de prix de test"""
        np.random.seed(42)
        prices = [40000.0]
        for _ in range(99):
            change = np.random.randn() * 100
            prices.append(prices[-1] + change)
        return pd.Series(prices)

    def test_rsi_calculation(self, price_series):
        """Test calcul RSI"""
        rsi = calculate_rsi(price_series, period=14)

        assert len(rsi) == len(price_series)
        # RSI doit être entre 0 et 100
        assert rsi.dropna().min() >= 0
        assert rsi.dropna().max() <= 100
        # Les premières valeurs doivent être NaN
        assert pd.isna(rsi.iloc[:13]).all()

    def test_rsi_overbought(self):
        """Test RSI en zone de surachat"""
        # Prix qui monte constamment
        prices = pd.Series([i * 100 for i in range(1, 51)])
        rsi = calculate_rsi(prices, period=14)

        # RSI devrait être élevé
        assert rsi.iloc[-1] > 70

    def test_rsi_oversold(self):
        """Test RSI en zone de survente"""
        # Prix qui baisse constamment
        prices = pd.Series([50000 - i * 100 for i in range(50)])
        rsi = calculate_rsi(prices, period=14)

        # RSI devrait être bas
        assert rsi.iloc[-1] < 30

    def test_bollinger_bands(self, price_series):
        """Test calcul Bollinger Bands"""
        upper, middle, lower = calculate_bollinger_bands(
            price_series,
            period=20,
            std_dev=2.0
        )

        # Vérifier longueurs
        assert len(upper) == len(price_series)
        assert len(middle) == len(price_series)
        assert len(lower) == len(price_series)

        # Upper > Middle > Lower
        valid_idx = ~pd.isna(middle)
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()

    def test_atr_calculation(self):
        """Test calcul ATR"""
        # Créer données OHLC
        high = pd.Series([105, 110, 108, 112, 115])
        low = pd.Series([95, 100, 98, 102, 105])
        close = pd.Series([100, 105, 103, 108, 112])

        atr = calculate_atr(high, low, close, period=3)

        # ATR doit être positif
        assert (atr.dropna() > 0).all()
        # ATR représente la volatilité moyenne
        assert len(atr) == len(high)


class TestPositionSizing:
    """Tests pour calcul taille de position"""

    def test_position_size_basic(self):
        """Test calcul taille de position basique"""
        size = calculate_position_size(
            capital=10000.0,
            risk_percent=1.0,
            entry_price=40000.0,
            stop_loss=39000.0
        )

        # Risque = 100$ (1% de 10000)
        # Distance SL = 1000$ par BTC
        # Taille = 100 / 1000 = 0.1 BTC
        assert size == pytest.approx(0.1, rel=0.01)

    def test_position_size_larger_risk(self):
        """Test taille avec risque plus élevé"""
        size = calculate_position_size(
            capital=10000.0,
            risk_percent=2.0,
            entry_price=40000.0,
            stop_loss=38000.0
        )

        # Risque = 200$ (2% de 10000)
        # Distance SL = 2000$ par BTC
        # Taille = 200 / 2000 = 0.1 BTC
        assert size == pytest.approx(0.1, rel=0.01)

    def test_position_size_tight_stop(self):
        """Test taille avec stop loss serré"""
        size = calculate_position_size(
            capital=10000.0,
            risk_percent=1.0,
            entry_price=40000.0,
            stop_loss=39500.0
        )

        # Risque = 100$
        # Distance SL = 500$ par BTC
        # Taille = 100 / 500 = 0.2 BTC
        assert size == pytest.approx(0.2, rel=0.01)

    def test_position_size_short(self):
        """Test taille pour position SHORT"""
        size = calculate_position_size(
            capital=10000.0,
            risk_percent=1.0,
            entry_price=40000.0,
            stop_loss=41000.0  # SL au dessus pour short
        )

        # Distance = 1000$ par BTC
        assert size == pytest.approx(0.1, rel=0.01)


class TestKellyCriterion:
    """Tests pour Kelly Criterion"""

    def test_kelly_positive_expectation(self):
        """Test Kelly avec espérance positive"""
        kelly = calculate_kelly_criterion(
            win_rate=0.6,  # 60% de trades gagnants
            avg_win=100.0,
            avg_loss=50.0
        )

        # Kelly devrait être positif
        assert kelly > 0
        # Et raisonnable (< 100%)
        assert kelly < 1.0

    def test_kelly_negative_expectation(self):
        """Test Kelly avec espérance négative"""
        kelly = calculate_kelly_criterion(
            win_rate=0.3,  # Seulement 30% de wins
            avg_win=100.0,
            avg_loss=150.0
        )

        # Kelly devrait être négatif ou nul
        assert kelly <= 0

    def test_kelly_fifty_fifty(self):
        """Test Kelly à 50/50 avec même R:R"""
        kelly = calculate_kelly_criterion(
            win_rate=0.5,
            avg_win=100.0,
            avg_loss=100.0
        )

        # Devrait être environ 0
        assert abs(kelly) < 0.01

    def test_kelly_high_win_rate(self):
        """Test Kelly avec haut win rate"""
        kelly = calculate_kelly_criterion(
            win_rate=0.8,
            avg_win=50.0,
            avg_loss=100.0
        )

        # Même avec petit R:R, haut win rate = Kelly positif
        assert kelly > 0


class TestSafeDivision:
    """Tests pour division sécurisée"""

    def test_safe_divide_normal(self):
        """Test division normale"""
        result = safe_divide(10.0, 2.0)
        assert result == 5.0

    def test_safe_divide_by_zero(self):
        """Test division par zéro"""
        result = safe_divide(10.0, 0.0)
        assert result == 0.0  # Valeur par défaut

    def test_safe_divide_custom_default(self):
        """Test division par zéro avec valeur par défaut custom"""
        result = safe_divide(10.0, 0.0, default=-1.0)
        assert result == -1.0

    def test_safe_divide_negative(self):
        """Test division avec négatifs"""
        result = safe_divide(-10.0, 2.0)
        assert result == -5.0

    def test_safe_divide_small_denominator(self):
        """Test avec très petit dénominateur"""
        result = safe_divide(10.0, 0.0000001)
        assert result == pytest.approx(100000000, rel=0.01)
