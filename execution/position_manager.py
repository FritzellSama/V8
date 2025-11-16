"""
Position Manager - Quantum Trader Pro
Gestion complète des positions ouvertes avec tracking et monitoring
"""

from typing import Dict, List, Optional
from datetime import datetime
from utils.logger import setup_logger
from models.position import Position, PositionSide, PositionStatus


class PositionManager:
    """
    Gestionnaire de positions avec:
    - Tracking positions ouvertes
    - Calcul PnL en temps réel
    - Monitoring stop-loss/take-profit
    - Statistiques et reporting
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le gestionnaire de positions
        
        Args:
            config: Configuration complète du bot
        """
        self.config = config
        self.logger = setup_logger('PositionManager')
        
        # Positions
        self.open_positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        # Limites
        risk_config = config.get('risk', {})
        self.max_positions = risk_config.get('max_positions_simultaneous', 3)
        self.max_same_direction = risk_config.get('max_positions_same_direction', 2)
        
        # Statistiques
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        
        self.logger.info("✅ Position Manager initialisé")
    
    def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        stop_loss: Optional[float] = None,
        take_profits: Optional[List[float]] = None,
        strategy: str = "",
        order_id: str = ""
    ) -> Optional[Position]:
        """
        Ouvre une nouvelle position

        Args:
            symbol: Paire de trading
            side: 'long' ou 'short'
            entry_price: Prix d'entrée
            size: Taille de la position
            stop_loss: Niveau de stop loss
            take_profits: Liste des niveaux de TP (prix)
            strategy: Nom de la stratégie
            order_id: ID de l'ordre d'entrée

        Returns:
            Position créée ou None si impossible
        """

        # Vérifier limites
        if not self._can_open_position(side):
            self.logger.warning(
                f"⚠️ Impossible d'ouvrir position: limites atteintes"
            )
            return None

        # Créer position
        position_id = f"{symbol}_{side}_{int(datetime.now().timestamp())}"

        position = Position(
            id=position_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            size=size,
            stop_loss=stop_loss,
            take_profits=take_profits if take_profits else [],
            strategy=strategy,
            entry_order_id=order_id
        )

        # Ajouter aux positions ouvertes
        self.open_positions[position_id] = position

        self.logger.trade_opened(
            symbol=symbol,
            side=side,
            size=size,
            price=entry_price,
            order_id=order_id
        )

        return position
    
    def close_position(
        self,
        position_id: str,
        close_price: float,
        reason: str = "manual",
        order_id: str = ""
    ) -> Optional[Position]:
        """
        Ferme une position
        
        Args:
            position_id: ID de la position
            close_price: Prix de sortie
            reason: Raison de fermeture
            order_id: ID de l'ordre de sortie
        
        Returns:
            Position fermée ou None
        """
        
        if position_id not in self.open_positions:
            self.logger.warning(f"⚠️ Position {position_id} introuvable")
            return None
        
        position = self.open_positions[position_id]
        
        # Fermer position
        position.close(close_price)
        position.exit_order_ids.append(order_id)
        
        # Déplacer vers closed
        del self.open_positions[position_id]
        self.closed_positions.append(position)
        
        # MAJ statistiques
        self.total_trades += 1
        self.total_pnl += position.pnl
        
        if position.pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Log
        if position.pnl > 0:
            self.logger.trade_closed(
                symbol=position.symbol,
                side=position.side,
                size=position.size,
                entry_price=position.entry_price,
                exit_price=close_price,
                pnl=position.pnl,
                pnl_percent=position.pnl_percent
            )
        else:
            self.logger.error(
                f"📉 Position fermée à perte: {position.symbol} "
                f"{position.side.upper()} | "
                f"PnL: ${position.pnl:.2f} ({position.pnl_percent:.2f}%)"
            )
        
        # Limiter historique (garder 500 dernières)
        if len(self.closed_positions) > 500:
            self.closed_positions = self.closed_positions[-500:]
        
        return position
    
    def partial_close_position(
        self,
        position_id: str,
        size_to_close: float,
        close_price: float,
        reason: str = "take_profit",
        order_id: str = ""
    ) -> Optional[Position]:
        """
        Ferme partiellement une position
        
        Args:
            position_id: ID de la position
            size_to_close: Taille à fermer
            close_price: Prix de sortie
            reason: Raison (ex: 'take_profit_1')
            order_id: ID de l'ordre
        
        Returns:
            Position modifiée ou None
        """
        
        if position_id not in self.open_positions:
            return None
        
        position = self.open_positions[position_id]
        
        # Fermeture partielle
        position.partial_close(size_to_close, close_price)
        position.exit_order_ids.append(order_id)
        
        self.logger.info(
            f"📊 Position partiellement fermée: {position.symbol} "
            f"({size_to_close}/{position.initial_size}) | "
            f"Raison: {reason} | "
            f"PnL partiel: ${position.realized_pnl:.2f}"
        )
        
        # Si complètement fermée, déplacer vers closed
        if position.status == 'closed':
            del self.open_positions[position_id]
            self.closed_positions.append(position)
            self.total_trades += 1
            self.total_pnl += position.pnl
            
            if position.pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
        
        return position
    
    def update_position_price(self, position_id: str, new_price: float):
        """
        Met à jour le prix d'une position
        
        Args:
            position_id: ID de la position
            new_price: Nouveau prix
        """
        if position_id in self.open_positions:
            self.open_positions[position_id].update_price(new_price)
    
    def update_all_positions(self, prices: Dict[str, float]):
        """
        Met à jour toutes les positions avec nouveaux prix
        
        Args:
            prices: Dict {symbol: price}
        """
        for position in self.open_positions.values():
            if position.symbol in prices:
                position.update_price(prices[position.symbol])
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """Récupère une position par ID"""
        return self.open_positions.get(position_id)
    
    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Récupère toutes les positions pour un symbole"""
        return [
            pos for pos in self.open_positions.values()
            if pos.symbol == symbol
        ]
    
    def get_all_open_positions(self) -> List[Position]:
        """Retourne toutes les positions ouvertes"""
        return list(self.open_positions.values())
    
    def get_positions_count(self) -> Dict[str, int]:
        """Compte les positions par direction"""
        long_count = sum(1 for p in self.open_positions.values() if p.side == 'long')
        short_count = sum(1 for p in self.open_positions.values() if p.side == 'short')
        
        return {
            'total': len(self.open_positions),
            'long': long_count,
            'short': short_count
        }
    
    def _can_open_position(self, side: str) -> bool:
        """Vérifie si on peut ouvrir une nouvelle position"""
        
        counts = self.get_positions_count()
        
        # Vérifier limite totale
        if counts['total'] >= self.max_positions:
            return False
        
        # Vérifier limite par direction
        side_count = counts.get(side, 0)
        if side_count >= self.max_same_direction:
            return False
        
        return True
    
    def get_total_exposure(self) -> Dict[str, float]:
        """Calcule l'exposition totale"""
        
        total_long = sum(
            p.size * p.current_price
            for p in self.open_positions.values()
            if p.side == 'long'
        )
        
        total_short = sum(
            p.size * p.current_price
            for p in self.open_positions.values()
            if p.side == 'short'
        )
        
        return {
            'long': total_long,
            'short': total_short,
            'net': total_long - total_short,
            'gross': total_long + total_short
        }
    
    def get_unrealized_pnl(self) -> float:
        """Calcule le PnL non réalisé total"""
        return sum(
            p.unrealized_pnl
            for p in self.open_positions.values()
        )
    
    def get_realized_pnl(self) -> float:
        """Calcule le PnL réalisé total (incluant positions fermées)"""
        return sum(
            p.realized_pnl
            for p in self.closed_positions
        )
    
    def get_statistics(self) -> Dict:
        """
        Retourne les statistiques complètes
        
        Returns:
            Dict avec toutes les stats
        """
        
        # Win rate
        win_rate = 0
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
        
        # Moyennes
        avg_win = 0
        avg_loss = 0
        
        if self.winning_trades > 0:
            winning_pnls = [p.pnl for p in self.closed_positions if p.pnl > 0]
            avg_win = sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0
        
        if self.losing_trades > 0:
            losing_pnls = [abs(p.pnl) for p in self.closed_positions if p.pnl < 0]
            avg_loss = sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0
        
        # Profit factor
        profit_factor = 0
        total_wins = sum(p.pnl for p in self.closed_positions if p.pnl > 0)
        total_losses = abs(sum(p.pnl for p in self.closed_positions if p.pnl < 0))
        
        if total_losses > 0:
            profit_factor = total_wins / total_losses
        
        # Exposure
        exposure = self.get_total_exposure()
        
        # Durée moyenne
        avg_duration = 0
        if self.closed_positions:
            avg_duration = sum(p.duration_minutes for p in self.closed_positions) / len(self.closed_positions)
        
        return {
            'open_positions': len(self.open_positions),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'unrealized_pnl': self.get_unrealized_pnl(),
            'realized_pnl': self.get_realized_pnl(),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_duration_minutes': avg_duration,
            'exposure_long': exposure['long'],
            'exposure_short': exposure['short'],
            'exposure_net': exposure['net'],
            'exposure_gross': exposure['gross']
        }
    
    def get_positions_summary(self) -> List[Dict]:
        """Retourne un résumé des positions ouvertes"""
        return [pos.to_dict() for pos in self.open_positions.values()]
    
    def close_all_positions(self, close_price_map: Dict[str, float], reason: str = "emergency"):
        """
        Ferme toutes les positions (urgence)
        
        Args:
            close_price_map: Dict {symbol: price}
            reason: Raison de fermeture
        """
        self.logger.warning(f"⚠️ Fermeture de toutes les positions: {reason}")
        
        position_ids = list(self.open_positions.keys())
        
        for position_id in position_ids:
            position = self.open_positions[position_id]
            close_price = close_price_map.get(position.symbol)
            
            if close_price:
                self.close_position(position_id, close_price, reason)
