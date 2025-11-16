"""
Unit Tests for Position Model
"""

import pytest
from datetime import datetime
from models.position import Position, ClosedTrade


class TestPosition:
    """Tests pour la classe Position"""

    def test_position_creation(self):
        """Test création d'une position"""
        position = Position(
            id='test_001',
            symbol='BTC/USDT',
            side='long',
            entry_price=40000.0,
            size=0.01
        )

        assert position.id == 'test_001'
        assert position.symbol == 'BTC/USDT'
        assert position.side == 'long'
        assert position.entry_price == 40000.0
        assert position.size == 0.01
        assert position.status == 'open'
        assert position.remaining_size == 0.01
        assert position.initial_size == 0.01

    def test_position_with_risk_levels(self):
        """Test position avec stop loss et take profits"""
        position = Position(
            id='test_002',
            symbol='ETH/USDT',
            side='short',
            entry_price=2000.0,
            size=1.0,
            stop_loss=2100.0,
            take_profits=[1900.0, 1800.0]
        )

        assert position.stop_loss == 2100.0
        assert len(position.take_profits) == 2
        assert position.take_profits[0] == 1900.0
        assert position.take_profits[1] == 1800.0

    def test_update_price_long(self):
        """Test mise à jour prix pour position LONG"""
        position = Position(
            id='test_003',
            symbol='BTC/USDT',
            side='long',
            entry_price=40000.0,
            size=0.01
        )

        # Prix monte
        position.update_price(41000.0)

        assert position.current_price == 41000.0
        assert position.unrealized_pnl > 0
        assert position.unrealized_pnl_percent > 0
        assert position.highest_price == 41000.0

    def test_update_price_short(self):
        """Test mise à jour prix pour position SHORT"""
        position = Position(
            id='test_004',
            symbol='BTC/USDT',
            side='short',
            entry_price=40000.0,
            size=0.01
        )

        # Prix baisse (profit pour short)
        position.update_price(39000.0)

        assert position.current_price == 39000.0
        assert position.unrealized_pnl > 0  # Profit car short
        assert position.lowest_price == 39000.0

    def test_partial_close(self):
        """Test fermeture partielle"""
        position = Position(
            id='test_005',
            symbol='BTC/USDT',
            side='long',
            entry_price=40000.0,
            size=0.1
        )

        # Fermer 50%
        pnl = position.partial_close(0.05, 41000.0)

        assert pnl > 0  # Profit
        assert position.remaining_size == 0.05
        assert position.status == 'partial'
        assert position.realized_pnl > 0

    def test_full_close(self):
        """Test fermeture complète"""
        position = Position(
            id='test_006',
            symbol='BTC/USDT',
            side='long',
            entry_price=40000.0,
            size=0.01
        )

        pnl = position.close(42000.0)

        assert position.status == 'closed'
        assert position.remaining_size == 0
        assert position.close_time is not None
        assert pnl > 0

    def test_stop_loss_hit_long(self):
        """Test détection stop loss pour LONG"""
        position = Position(
            id='test_007',
            symbol='BTC/USDT',
            side='long',
            entry_price=40000.0,
            size=0.01,
            stop_loss=39000.0
        )

        # Prix au dessus du SL
        position.update_price(39500.0)
        assert not position.is_stop_loss_hit()

        # Prix au niveau ou en dessous du SL
        position.update_price(39000.0)
        assert position.is_stop_loss_hit()

        position.update_price(38500.0)
        assert position.is_stop_loss_hit()

    def test_stop_loss_hit_short(self):
        """Test détection stop loss pour SHORT"""
        position = Position(
            id='test_008',
            symbol='BTC/USDT',
            side='short',
            entry_price=40000.0,
            size=0.01,
            stop_loss=41000.0
        )

        # Prix en dessous du SL
        position.update_price(40500.0)
        assert not position.is_stop_loss_hit()

        # Prix au niveau ou au dessus du SL
        position.update_price(41000.0)
        assert position.is_stop_loss_hit()

    def test_check_take_profits_long(self):
        """Test vérification des take profits pour LONG"""
        position = Position(
            id='test_009',
            symbol='BTC/USDT',
            side='long',
            entry_price=40000.0,
            size=0.01,
            take_profits=[41000.0, 42000.0, 43000.0]
        )

        # Aucun TP atteint
        position.update_price(40500.0)
        hit = position.check_take_profits()
        assert len(hit) == 0

        # Premier TP atteint
        position.update_price(41500.0)
        hit = position.check_take_profits()
        assert 0 in hit
        assert 1 not in hit

        # Deux TPs atteints
        position.update_price(42500.0)
        hit = position.check_take_profits()
        assert 0 in hit
        assert 1 in hit
        assert 2 not in hit

    def test_check_take_profits_short(self):
        """Test vérification des take profits pour SHORT"""
        position = Position(
            id='test_010',
            symbol='BTC/USDT',
            side='short',
            entry_price=40000.0,
            size=0.01,
            take_profits=[39000.0, 38000.0]
        )

        # Aucun TP atteint
        position.update_price(39500.0)
        hit = position.check_take_profits()
        assert len(hit) == 0

        # Premier TP atteint
        position.update_price(38500.0)
        hit = position.check_take_profits()
        assert 0 in hit
        assert 1 not in hit

    def test_risk_distance(self):
        """Test calcul distance au stop loss"""
        position = Position(
            id='test_011',
            symbol='BTC/USDT',
            side='long',
            entry_price=40000.0,
            size=0.01,
            stop_loss=38000.0
        )

        distance = position.get_risk_distance()
        expected = (40000.0 - 38000.0) / 40000.0 * 100  # 5%
        assert abs(distance - expected) < 0.01

    def test_reward_distance(self):
        """Test calcul distance au take profit"""
        position = Position(
            id='test_012',
            symbol='BTC/USDT',
            side='long',
            entry_price=40000.0,
            size=0.01,
            take_profits=[42000.0, 44000.0]
        )

        distance = position.get_reward_distance(0)
        expected = (42000.0 - 40000.0) / 40000.0 * 100  # 5%
        assert abs(distance - expected) < 0.01

        distance = position.get_reward_distance(1)
        expected = (44000.0 - 40000.0) / 40000.0 * 100  # 10%
        assert abs(distance - expected) < 0.01

    def test_to_dict(self):
        """Test conversion en dictionnaire"""
        position = Position(
            id='test_013',
            symbol='BTC/USDT',
            side='long',
            entry_price=40000.0,
            size=0.01,
            strategy='test_strategy'
        )

        data = position.to_dict()

        assert data['id'] == 'test_013'
        assert data['symbol'] == 'BTC/USDT'
        assert data['side'] == 'long'
        assert data['entry_price'] == 40000.0
        assert data['size'] == 0.01
        assert data['strategy'] == 'test_strategy'
        assert 'entry_time' in data
        assert 'status' in data

    def test_from_dict(self):
        """Test création depuis dictionnaire"""
        data = {
            'id': 'test_014',
            'symbol': 'ETH/USDT',
            'side': 'short',
            'entry_price': 2000.0,
            'size': 1.0,
            'stop_loss': 2100.0,
            'take_profits': [1900.0, 1800.0],
            'status': 'open',
            'strategy': 'grid_trading'
        }

        position = Position.from_dict(data)

        assert position.id == 'test_014'
        assert position.symbol == 'ETH/USDT'
        assert position.side == 'short'
        assert position.entry_price == 2000.0
        assert position.stop_loss == 2100.0
        assert len(position.take_profits) == 2

    def test_str_representation(self):
        """Test représentation string"""
        position = Position(
            id='test_015',
            symbol='BTC/USDT',
            side='long',
            entry_price=40000.0,
            size=0.01
        )

        str_repr = str(position)
        assert 'test_015' in str_repr
        assert 'LONG' in str_repr
        assert 'BTC/USDT' in str_repr


class TestClosedTrade:
    """Tests pour ClosedTrade"""

    def test_from_position(self):
        """Test création depuis Position"""
        position = Position(
            id='test_016',
            symbol='BTC/USDT',
            side='long',
            entry_price=40000.0,
            size=0.01,
            strategy='ichimoku'
        )

        closed = ClosedTrade.from_position(
            position,
            exit_price=42000.0,
            exit_reason='take_profit'
        )

        assert closed.position_id == 'test_016'
        assert closed.symbol == 'BTC/USDT'
        assert closed.side == 'long'
        assert closed.entry_price == 40000.0
        assert closed.exit_price == 42000.0
        assert closed.size == 0.01
        assert closed.pnl > 0  # Profit
        assert closed.exit_reason == 'take_profit'
        assert closed.strategy == 'ichimoku'
