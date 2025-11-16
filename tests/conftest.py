"""
Pytest Configuration and Fixtures - Quantum Trader Pro
Fixtures partagées pour tous les tests
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

# Ajouter le répertoire racine au path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


# ============================================================================
# FIXTURES DE CONFIGURATION
# ============================================================================

@pytest.fixture
def sample_config():
    """Configuration de test complète"""
    return {
        'exchange': {
            'primary': {
                'name': 'binance',
                'api_key': 'test_api_key_1234567890123456789012345678901234567890123456789012345678901234',
                'secret_key': 'test_secret_key_12345678901234567890123456789012345678901234567890123456',
                'testnet': True,
                'private_key_path': 'test-prv-key.pem',
                'timeout_seconds': 30,
                'retry_attempts': 3,
                'rate_limit_buffer': 0.1
            }
        },
        'symbols': {
            'primary': 'BTC/USDT',
            'allowed': ['BTC/USDT', 'ETH/USDT']
        },
        'timeframes': {
            'trend': '1h',
            'signal': '5m',
            'execution': '1m'
        },
        'capital': {
            'initial': 10000.0,
            'current': 10000.0
        },
        'risk': {
            'max_risk_per_trade_percent': 1.0,
            'max_daily_loss_percent': 5.0,
            'max_positions_simultaneous': 3,
            'max_positions_same_direction': 2
        },
        'execution': {
            'order_type': 'limit',
            'limit_price_offset_percent': 0.05,
            'expected_slippage_percent': 0.05,
            'max_acceptable_slippage_percent': 0.2,
            'max_retries': 3,
            'retry_delay_seconds': 2,
            'timeout_seconds': 30,
            'sor': {
                'enabled': True,
                'split_large_orders': True,
                'max_order_size_percent': 5,
                'iceberg_orders': True
            },
            'post_trade': {
                'calculate_slippage': True,
                'calculate_impact': True,
                'save_to_database': False
            }
        },
        'strategies': {
            'ichimoku_scalping': {
                'enabled': True,
                'weight': 0.4
            },
            'grid_trading': {
                'enabled': False,
                'weight': 0.3
            },
            'dca_bot': {
                'enabled': True,
                'weight': 0.3,
                'dca': {
                    'interval_hours': 1,
                    'amount_per_interval': 100.0,
                    'drop_trigger_percent': -10.0,
                    'accumulation_target': 1.0,
                    'emergency_stop_loss_percent': 50.0
                }
            }
        },
        'logging': {
            'level': 'DEBUG',
            'console_output': False,
            'colored_output': False,
            'save_to_file': False,
            'log_directory': 'logs',
            'rotation': 'daily',
            'max_file_size_mb': 10,
            'backup_count': 10,
            'format': 'text'
        },
        'live': {
            'check_interval_seconds': 10,
            'warmup_bars_required': 100,
            'auto_restart_on_error': True
        },
        'ml': {
            'enabled': False
        },
        'database': {
            'url': 'sqlite:///:memory:'
        },
        'monitoring': {
            'telegram': {
                'enabled': False,
                'bot_token': '',
                'chat_id': ''
            }
        }
    }


@pytest.fixture
def minimal_config():
    """Configuration minimale pour tests rapides"""
    return {
        'exchange': {
            'primary': {
                'api_key': 'test_key_' + '0' * 56,
                'secret_key': 'test_secret_' + '0' * 52,
                'testnet': True
            }
        },
        'symbols': {'primary': 'BTC/USDT'},
        'capital': {'initial': 1000.0},
        'risk': {
            'max_risk_per_trade_percent': 1.0,
            'max_daily_loss_percent': 5.0,
            'max_positions_simultaneous': 3
        },
        'logging': {
            'level': 'ERROR',
            'console_output': False,
            'save_to_file': False
        }
    }


# ============================================================================
# FIXTURES DE DONNÉES MARKET
# ============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Données OHLCV de test"""
    import pandas as pd
    import numpy as np

    # Créer 100 bougies de test
    np.random.seed(42)
    n_bars = 100

    timestamps = pd.date_range(
        start='2024-01-01 00:00:00',
        periods=n_bars,
        freq='5min'
    )

    # Prix simulé avec tendance haussière
    base_price = 40000.0
    prices = [base_price]
    for i in range(1, n_bars):
        change = np.random.randn() * 50 + 5  # Légère tendance haussière
        prices.append(prices[-1] + change)

    data = {
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn() * 30) for p in prices],
        'low': [p - abs(np.random.randn() * 30) for p in prices],
        'close': [p + np.random.randn() * 20 for p in prices],
        'volume': [np.random.uniform(100, 1000) for _ in range(n_bars)]
    }

    df = pd.DataFrame(data)
    df = df.set_index('timestamp')

    return df


@pytest.fixture
def sample_ticker():
    """Ticker de test"""
    return {
        'symbol': 'BTC/USDT',
        'bid': 40000.0,
        'ask': 40010.0,
        'last': 40005.0,
        'spread': 10.0,
        'spread_percent': 0.025,
        'volume': 1500000.0,
        'timestamp': datetime.now()
    }


@pytest.fixture
def sample_orderbook():
    """Orderbook de test"""
    return {
        'bids': [
            [40000.0, 1.5],
            [39995.0, 2.0],
            [39990.0, 3.0],
            [39985.0, 5.0],
            [39980.0, 10.0]
        ],
        'asks': [
            [40010.0, 1.0],
            [40015.0, 2.0],
            [40020.0, 3.0],
            [40025.0, 4.0],
            [40030.0, 8.0]
        ],
        'timestamp': datetime.now()
    }


# ============================================================================
# FIXTURES DE TRADING
# ============================================================================

@pytest.fixture
def sample_signal():
    """Signal de trading de test"""
    from strategies.base_strategy import Signal

    return Signal(
        timestamp=datetime.now(),
        action='BUY',
        symbol='BTC/USDT',
        confidence=0.85,
        entry_price=40000.0,
        stop_loss=39000.0,
        take_profit=[(41000.0, 0.5), (42000.0, 0.5)],
        size=0.01,
        reason=['test_signal'],
        metadata={'strategy': 'test'}
    )


@pytest.fixture
def sample_position():
    """Position de test"""
    from models.position import Position

    return Position(
        id='test_position_001',
        symbol='BTC/USDT',
        side='long',
        entry_price=40000.0,
        size=0.01,
        current_price=40500.0,
        stop_loss=39000.0,
        take_profits=[41000.0, 42000.0],
        strategy='test_strategy'
    )


@pytest.fixture
def sample_order_result():
    """Résultat d'ordre de test"""
    return {
        'id': 'order_123456',
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'type': 'limit',
        'status': 'closed',
        'price': 40000.0,
        'average': 40005.0,
        'amount': 0.01,
        'filled': 0.01,
        'remaining': 0.0,
        'cost': 400.05,
        'fee': {'cost': 0.4, 'currency': 'USDT'},
        'timestamp': datetime.now().timestamp() * 1000
    }


# ============================================================================
# MOCKS
# ============================================================================

@pytest.fixture
def mock_binance_client():
    """Mock du client Binance"""
    client = MagicMock()

    # Configurer les retours par défaut
    client.symbol = 'BTC/USDT'
    client.base = 'BTC'
    client.quote = 'USDT'
    client.testnet = True
    client.is_connected = True

    # Mock des méthodes
    client.get_ticker.return_value = {
        'symbol': 'BTC/USDT',
        'bid': 40000.0,
        'ask': 40010.0,
        'last': 40005.0,
        'spread': 10.0,
        'volume': 1500000.0
    }

    client.get_balance.return_value = {
        'base': {'free': 0.1, 'used': 0.0, 'total': 0.1},
        'quote': {'free': 10000.0, 'used': 0.0, 'total': 10000.0}
    }

    client.fetch_ohlcv.return_value = [
        [1704067200000, 40000.0, 40100.0, 39900.0, 40050.0, 100.0],
        [1704067500000, 40050.0, 40150.0, 40000.0, 40100.0, 120.0],
        [1704067800000, 40100.0, 40200.0, 40050.0, 40180.0, 110.0],
    ]

    client.test_connectivity.return_value = True
    client.reconnect.return_value = True

    client.create_order.return_value = {
        'id': 'mock_order_001',
        'status': 'closed',
        'filled': 0.01,
        'average': 40005.0
    }

    return client


@pytest.fixture
def mock_logger():
    """Mock du logger"""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.debug = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.critical = MagicMock()
    logger.trade_opened = MagicMock()
    logger.trade_closed = MagicMock()
    logger.daily_summary = MagicMock()
    return logger


# ============================================================================
# HELPERS
# ============================================================================

@pytest.fixture
def temp_env_file(tmp_path):
    """Crée un fichier .env temporaire"""
    env_file = tmp_path / '.env'
    env_content = """
BINANCE_API_KEY=test_api_key_1234567890123456789012345678901234567890123456789012345678901234
BINANCE_SECRET_KEY=test_secret_1234567890123456789012345678901234567890123456789012345678901234
BINANCE_TESTNET=true
INITIAL_CAPITAL=1000
SYMBOL=BTC/USDT
MAX_RISK_PER_TRADE=1.0
MAX_DAILY_LOSS=5.0
MAX_POSITIONS=3
"""
    env_file.write_text(env_content.strip())
    return env_file


@pytest.fixture
def temp_config_yaml(tmp_path):
    """Crée un fichier config.yaml temporaire"""
    import yaml

    config_file = tmp_path / 'config.yaml'
    config_data = {
        'exchange': {
            'primary': {
                'name': 'binance',
                'api_key': '',
                'secret_key': '',
                'testnet': True
            }
        },
        'symbols': {'primary': 'BTC/USDT'},
        'capital': {'initial': 1000.0},
        'risk': {
            'max_risk_per_trade_percent': 1.0,
            'max_daily_loss_percent': 5.0,
            'max_positions_simultaneous': 3
        },
        'logging': {
            'level': 'INFO',
            'console_output': False,
            'save_to_file': False
        }
    }

    config_file.write_text(yaml.dump(config_data))
    return config_file
