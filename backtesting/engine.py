"""
Backtest Engine - Quantum Trader Pro
Moteur de backtesting pour tester les stratégies sur données historiques
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from utils.logger import setup_logger
from utils.calculations import calculate_rsi, calculate_bollinger_bands, calculate_atr

logger = setup_logger('BacktestEngine')


@dataclass
class BacktestTrade:
    """Représente un trade dans le backtest"""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl: float = 0.0
    pnl_percent: float = 0.0
    fees: float = 0.0
    status: str = 'open'  # 'open', 'closed', 'stopped'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class BacktestResult:
    """Résultats complets du backtest"""
    # Performance
    total_return: float = 0.0
    total_return_percent: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0

    # Trades
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Timing
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    duration_days: int = 0

    # Capital
    initial_capital: float = 0.0
    final_capital: float = 0.0

    # Détails
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)


class BacktestEngine:
    """
    Moteur de backtesting pour tester les stratégies sur données historiques.

    Features:
    - Simulation réaliste avec slippage et fees
    - Support stop-loss et take-profit
    - Calcul métriques avancées (Sharpe, Sortino, Drawdown)
    - Génération de rapports détaillés
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.001,  # 0.1% par défaut (Binance)
        slippage: float = 0.0005,  # 0.05% slippage
        risk_per_trade: float = 0.02  # 2% du capital par trade
    ):
        """
        Initialise le moteur de backtest

        Args:
            initial_capital: Capital initial
            fee_rate: Taux de frais par trade
            slippage: Slippage simulé
            risk_per_trade: Risque par trade en % du capital
        """
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.risk_per_trade = risk_per_trade

        self.capital = initial_capital
        self.trades: List[BacktestTrade] = []
        self.open_positions: List[BacktestTrade] = []
        self.equity_curve: List[float] = [initial_capital]

        logger.info(f"BacktestEngine initialisé avec {initial_capital} USDT")

    def run(
        self,
        data: pd.DataFrame,
        strategy: Callable[[pd.DataFrame, int], Optional[Dict[str, Any]]],
        symbol: str = "BTC/USDT"
    ) -> BacktestResult:
        """
        Exécute le backtest sur les données historiques

        Args:
            data: DataFrame OHLCV avec colonnes ['open', 'high', 'low', 'close', 'volume']
            strategy: Fonction de stratégie qui retourne un signal
            symbol: Paire de trading

        Returns:
            Résultats du backtest
        """
        logger.info(f"Démarrage backtest sur {len(data)} bougies")

        # Reset état
        self.capital = self.initial_capital
        self.trades = []
        self.open_positions = []
        self.equity_curve = [self.initial_capital]

        # Vérifier données
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"DataFrame doit contenir: {required_columns}")

        # Boucle principale du backtest
        for i in range(len(data)):
            current_bar = data.iloc[i]
            current_time = data.index[i] if isinstance(data.index, pd.DatetimeIndex) else datetime.now()

            # 1. Vérifier stop-loss/take-profit sur positions ouvertes
            self._check_stops(current_bar, current_time)

            # 2. Obtenir signal de la stratégie
            signal = strategy(data, i)

            # 3. Exécuter le signal
            if signal:
                self._execute_signal(signal, current_bar, current_time, symbol)

            # 4. Mettre à jour equity curve
            equity = self._calculate_equity(current_bar['close'])
            self.equity_curve.append(equity)

        # Fermer positions restantes
        if self.open_positions:
            final_bar = data.iloc[-1]
            final_time = data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else datetime.now()
            self._close_all_positions(final_bar['close'], final_time)

        # Calculer résultats
        result = self._calculate_results(data)

        logger.info(f"Backtest terminé: {result.total_return_percent:.2f}% return, "
                   f"{result.win_rate:.1f}% win rate")

        return result

    def _execute_signal(
        self,
        signal: Dict[str, Any],
        bar: pd.Series,
        timestamp: datetime,
        symbol: str
    ) -> None:
        """Exécute un signal de trading"""
        action = signal.get('action', '')

        if action == 'buy':
            # Fermer shorts existants
            self._close_positions_by_side('short', bar['close'], timestamp)

            # Ouvrir long
            if not self._has_position('long'):
                self._open_position(
                    symbol=symbol,
                    side='long',
                    price=bar['close'],
                    timestamp=timestamp,
                    stop_loss=signal.get('stop_loss'),
                    take_profit=signal.get('take_profit')
                )

        elif action == 'sell':
            # Fermer longs existants
            self._close_positions_by_side('long', bar['close'], timestamp)

            # Ouvrir short (si supporté)
            if signal.get('open_short', False) and not self._has_position('short'):
                self._open_position(
                    symbol=symbol,
                    side='short',
                    price=bar['close'],
                    timestamp=timestamp,
                    stop_loss=signal.get('stop_loss'),
                    take_profit=signal.get('take_profit')
                )

        elif action == 'close':
            # Fermer toutes les positions
            self._close_all_positions(bar['close'], timestamp)

    def _open_position(
        self,
        symbol: str,
        side: str,
        price: float,
        timestamp: datetime,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> None:
        """Ouvre une nouvelle position"""
        # Appliquer slippage
        if side == 'long':
            entry_price = price * (1 + self.slippage)
        else:
            entry_price = price * (1 - self.slippage)

        # Calculer taille position basée sur risque
        position_value = self.capital * self.risk_per_trade * 10  # Leverage implicite
        position_value = min(position_value, self.capital * 0.95)  # Max 95% du capital

        quantity = position_value / entry_price

        # Calculer fees
        fees = position_value * self.fee_rate
        self.capital -= fees

        trade = BacktestTrade(
            entry_time=timestamp,
            exit_time=None,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            exit_price=None,
            quantity=quantity,
            fees=fees,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status='open'
        )

        self.open_positions.append(trade)
        logger.debug(f"Position ouverte: {side} {quantity:.6f} @ {entry_price:.2f}")

    def _close_position(
        self,
        position: BacktestTrade,
        exit_price: float,
        timestamp: datetime,
        status: str = 'closed'
    ) -> None:
        """Ferme une position"""
        # Appliquer slippage
        if position.side == 'long':
            actual_exit = exit_price * (1 - self.slippage)
        else:
            actual_exit = exit_price * (1 + self.slippage)

        position.exit_price = actual_exit
        position.exit_time = timestamp
        position.status = status

        # Calculer P&L
        if position.side == 'long':
            pnl = (actual_exit - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - actual_exit) * position.quantity

        # Soustraire fees de sortie
        exit_fees = actual_exit * position.quantity * self.fee_rate
        pnl -= exit_fees
        position.fees += exit_fees

        position.pnl = pnl
        position.pnl_percent = (pnl / (position.entry_price * position.quantity)) * 100

        # Mettre à jour capital
        self.capital += pnl

        # Déplacer vers trades fermés
        self.open_positions.remove(position)
        self.trades.append(position)

        logger.debug(f"Position fermée: {position.side} @ {actual_exit:.2f}, "
                    f"P&L: {pnl:.2f} ({position.pnl_percent:.2f}%)")

    def _close_positions_by_side(self, side: str, price: float, timestamp: datetime) -> None:
        """Ferme toutes les positions d'un côté"""
        positions_to_close = [p for p in self.open_positions if p.side == side]
        for position in positions_to_close:
            self._close_position(position, price, timestamp)

    def _close_all_positions(self, price: float, timestamp: datetime) -> None:
        """Ferme toutes les positions ouvertes"""
        for position in list(self.open_positions):
            self._close_position(position, price, timestamp)

    def _has_position(self, side: str) -> bool:
        """Vérifie si une position est ouverte"""
        return any(p.side == side for p in self.open_positions)

    def _check_stops(self, bar: pd.Series, timestamp: datetime) -> None:
        """Vérifie les stop-loss et take-profit"""
        for position in list(self.open_positions):
            if position.side == 'long':
                # Stop loss
                if position.stop_loss and bar['low'] <= position.stop_loss:
                    self._close_position(position, position.stop_loss, timestamp, 'stopped')
                # Take profit
                elif position.take_profit and bar['high'] >= position.take_profit:
                    self._close_position(position, position.take_profit, timestamp, 'closed')
            else:  # short
                # Stop loss
                if position.stop_loss and bar['high'] >= position.stop_loss:
                    self._close_position(position, position.stop_loss, timestamp, 'stopped')
                # Take profit
                elif position.take_profit and bar['low'] <= position.take_profit:
                    self._close_position(position, position.take_profit, timestamp, 'closed')

    def _calculate_equity(self, current_price: float) -> float:
        """Calcule l'equity actuelle"""
        equity = self.capital

        for position in self.open_positions:
            if position.side == 'long':
                unrealized = (current_price - position.entry_price) * position.quantity
            else:
                unrealized = (position.entry_price - current_price) * position.quantity
            equity += unrealized

        return equity

    def _calculate_results(self, data: pd.DataFrame) -> BacktestResult:
        """Calcule les métriques de performance"""
        result = BacktestResult()

        # Dates
        if isinstance(data.index, pd.DatetimeIndex):
            result.start_date = data.index[0].to_pydatetime()
            result.end_date = data.index[-1].to_pydatetime()
            result.duration_days = (result.end_date - result.start_date).days

        # Capital
        result.initial_capital = self.initial_capital
        result.final_capital = self.capital
        result.total_return = self.capital - self.initial_capital
        result.total_return_percent = (result.total_return / self.initial_capital) * 100

        # Trades
        result.trades = self.trades
        result.total_trades = len(self.trades)

        if result.total_trades > 0:
            winning = [t for t in self.trades if t.pnl > 0]
            losing = [t for t in self.trades if t.pnl <= 0]

            result.winning_trades = len(winning)
            result.losing_trades = len(losing)
            result.win_rate = (result.winning_trades / result.total_trades) * 100

            if winning:
                result.average_win = sum(t.pnl for t in winning) / len(winning)
                result.largest_win = max(t.pnl for t in winning)

            if losing:
                result.average_loss = abs(sum(t.pnl for t in losing) / len(losing))
                result.largest_loss = abs(min(t.pnl for t in losing))

            # Profit factor
            total_wins = sum(t.pnl for t in winning) if winning else 0
            total_losses = abs(sum(t.pnl for t in losing)) if losing else 1
            result.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Equity curve et drawdown
        result.equity_curve = self.equity_curve

        # Calculer drawdown
        peak = self.initial_capital
        drawdowns = []

        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            drawdowns.append(drawdown)

        result.drawdown_curve = drawdowns
        result.max_drawdown = max(drawdowns) if drawdowns else 0
        result.max_drawdown_percent = (result.max_drawdown / self.initial_capital) * 100

        # Sharpe et Sortino
        if len(self.equity_curve) > 1:
            returns = pd.Series(self.equity_curve).pct_change().dropna()

            if len(returns) > 0 and returns.std() > 0:
                # Sharpe (annualisé, assume 365 jours de trading)
                result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(365)

                # Sortino (seulement downside risk)
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0 and downside_returns.std() > 0:
                    result.sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(365)

        return result

    def generate_report(self, result: BacktestResult) -> str:
        """Génère un rapport texte des résultats"""
        report = []
        report.append("=" * 60)
        report.append("RAPPORT DE BACKTEST - Quantum Trader Pro")
        report.append("=" * 60)

        if result.start_date and result.end_date:
            report.append(f"Période: {result.start_date.strftime('%Y-%m-%d')} à "
                         f"{result.end_date.strftime('%Y-%m-%d')} ({result.duration_days} jours)")

        report.append(f"\nCAPITAL:")
        report.append(f"  Initial: {result.initial_capital:.2f} USDT")
        report.append(f"  Final: {result.final_capital:.2f} USDT")
        report.append(f"  Retour: {result.total_return:.2f} USDT ({result.total_return_percent:.2f}%)")

        report.append(f"\nPERFORMANCE:")
        report.append(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        report.append(f"  Sortino Ratio: {result.sortino_ratio:.2f}")
        report.append(f"  Max Drawdown: {result.max_drawdown:.2f} USDT ({result.max_drawdown_percent:.2f}%)")

        report.append(f"\nTRADES:")
        report.append(f"  Total: {result.total_trades}")
        report.append(f"  Gagnants: {result.winning_trades} ({result.win_rate:.1f}%)")
        report.append(f"  Perdants: {result.losing_trades}")
        report.append(f"  Profit Factor: {result.profit_factor:.2f}")

        if result.total_trades > 0:
            report.append(f"\n  Gain moyen: {result.average_win:.2f} USDT")
            report.append(f"  Perte moyenne: {result.average_loss:.2f} USDT")
            report.append(f"  Plus gros gain: {result.largest_win:.2f} USDT")
            report.append(f"  Plus grosse perte: {result.largest_loss:.2f} USDT")

        report.append("=" * 60)

        return "\n".join(report)


# Stratégies de test intégrées
def simple_sma_strategy(data: pd.DataFrame, index: int) -> Optional[Dict[str, Any]]:
    """
    Stratégie simple SMA crossover pour tests

    Args:
        data: DataFrame OHLCV
        index: Index actuel

    Returns:
        Signal ou None
    """
    if index < 50:  # Besoin de données historiques
        return None

    # Calculer SMAs
    close = data['close'].iloc[:index+1]
    sma_fast = close.rolling(20).mean().iloc[-1]
    sma_slow = close.rolling(50).mean().iloc[-1]

    sma_fast_prev = close.rolling(20).mean().iloc[-2]
    sma_slow_prev = close.rolling(50).mean().iloc[-2]

    current_price = close.iloc[-1]

    # Signal de croisement
    if sma_fast > sma_slow and sma_fast_prev <= sma_slow_prev:
        return {
            'action': 'buy',
            'stop_loss': current_price * 0.98,  # 2% stop loss
            'take_profit': current_price * 1.04  # 4% take profit
        }
    elif sma_fast < sma_slow and sma_fast_prev >= sma_slow_prev:
        return {
            'action': 'sell'
        }

    return None


def rsi_strategy(data: pd.DataFrame, index: int) -> Optional[Dict[str, Any]]:
    """
    Stratégie RSI oversold/overbought

    Args:
        data: DataFrame OHLCV
        index: Index actuel

    Returns:
        Signal ou None
    """
    if index < 14:  # Besoin de 14 périodes pour RSI
        return None

    close = data['close'].iloc[:index+1].tolist()
    rsi = calculate_rsi(close, period=14)

    if rsi is None:
        return None

    current_price = close[-1]

    # RSI oversold -> Buy
    if rsi < 30:
        return {
            'action': 'buy',
            'stop_loss': current_price * 0.97,
            'take_profit': current_price * 1.05
        }
    # RSI overbought -> Sell
    elif rsi > 70:
        return {
            'action': 'sell'
        }

    return None


__all__ = [
    'BacktestEngine',
    'BacktestTrade',
    'BacktestResult',
    'simple_sma_strategy',
    'rsi_strategy'
]
