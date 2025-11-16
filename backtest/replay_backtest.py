"""
Replay Backtest - Quantum Trader Pro
Système de backtesting qui utilise le code de production (main.py) avec des données historiques
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta
from typing import Dict, Optional
import pandas as pd

from config import ConfigLoader
from core.virtual_binance_client import VirtualBinanceClient
from data.data_loader import DataLoader
from core.binance_client import BinanceClient
from utils.logger import setup_logger

class ReplayBacktest:
    """
    Backtester qui:
    1. Charge les données historiques
    2. Crée un VirtualBinanceClient
    3. Lance main.py en mode replay
    4. Avance le temps bougie par bougie
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise le replay backtest

        Args:
            config_path: Chemin vers config.yaml
        """
        self.logger = setup_logger('ReplayBacktest')
        self.logger.info("=" * 70)
        self.logger.info("🔄 QUANTUM TRADER PRO - REPLAY BACKTEST")
        self.logger.info("=" * 70)

        # Charger config
        try:
            self.config_loader = ConfigLoader(config_path)
            self.config = self.config_loader.config
            self.backtest_config = self.config.get('backtest', {})
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement config: {e}")
            sys.exit(1)

        # Paramètres backtest
        data_config = self.backtest_config.get('data', {})
        self.start_date = data_config.get('start_date', '2023-01-01')
        self.end_date = data_config.get('end_date', '2024-11-08')
        self.warmup_bars = data_config.get('warmup_bars', 100)

        # État
        self.historical_data = {}
        self.virtual_client = None
        self.bot = None

    def load_historical_data(self):
        """Charge les données historiques depuis Binance"""

        self.logger.info(f"📥 Chargement données: {self.start_date} → {self.end_date}")

        # Utiliser un vrai client juste pour charger les données
        temp_client = BinanceClient(self.config)
        data_loader = DataLoader(temp_client, self.config)

        # Charger pour chaque timeframe configuré
        timeframes_config = self.config.get('timeframes', {})
        timeframes = [
            timeframes_config.get('trend', '1h'),
            timeframes_config.get('signal', '5m'),
            timeframes_config.get('micro', '1m')
        ]

        for tf in timeframes:
            try:
                self.logger.info(f"📥 Chargement {tf}...")

                # Charger les données historiques
                df = data_loader.load_historical_data(
                    symbol=self.config['symbols']['primary'],
                    timeframe=tf,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    limit=2000  # Plus de données pour le backtest
                )

                if df is not None and len(df) > 0:
                    self.historical_data[tf] = df
                    self.logger.info(f"✅ {len(df)} bougies chargées pour {tf}")
                else:
                    self.logger.warning(f"⚠️ Aucune donnée pour {tf}")

            except Exception as e:
                self.logger.error(f"❌ Erreur chargement {tf}: {e}")
                raise

        if not self.historical_data:
            raise Exception("Aucune donnée historique chargée")

        return self.historical_data

    def prepare_virtual_client(self):
        """Prépare le client virtuel avec les données"""

        self.logger.info("🔧 Préparation du client virtuel...")

        # Créer client virtuel
        self.virtual_client = VirtualBinanceClient(self.config)

        # Charger les données
        self.virtual_client.load_historical_data(self.historical_data)

        self.logger.info("✅ Client virtuel prêt")

        return self.virtual_client

    def run_replay(self):
        """
        Lance le replay en utilisant main.py
        Avance le temps bougie par bougie
        """

        self.logger.info("🔄 Démarrage du replay...")

        # Import ici pour éviter les imports circulaires
        from main import QuantumTraderBot

        # Créer le bot avec le client virtuel
        self.bot = QuantumTraderBot(config_path=None)

        # REMPLACER le client réel par le virtuel
        self.bot.client = self.virtual_client
        self.bot.data_loader.client = self.virtual_client
        self.bot.market_data.client = self.virtual_client
        self.bot.trade_executor.client = self.virtual_client
        self.bot.trade_executor.order_executor.client = self.virtual_client

        # Désactiver le warmup initial (on a déjà les données)
        # On va faire notre propre boucle de trading

        # Obtenir la timeframe principale (plus petite pour plus de précision)
        main_tf = min(self.historical_data.keys(),
                     key=lambda x: self.virtual_client._timeframe_to_minutes(x))
        df_main = self.historical_data[main_tf]

        self.logger.info(f"📊 Timeframe principale: {main_tf} ({len(df_main)} bougies)")
        self.logger.info("🔄 Démarrage boucle de trading...")

        # Boucle principale: avancer bougie par bougie
        total_bars = len(df_main)

        for i in range(self.warmup_bars, total_bars):
            try:
                # Mettre à jour l'index pour toutes les timeframes
                for tf in self.historical_data.keys():
                    # Calculer l'index correspondant pour cette timeframe
                    ratio = self.virtual_client._timeframe_to_minutes(main_tf) / self.virtual_client._timeframe_to_minutes(tf)
                    self.virtual_client.current_index[tf] = int(i * ratio)

                # Obtenir le timestamp courant
                current_bar = df_main.iloc[i]
                current_time = current_bar.name
                current_price = current_bar['close']

                # Avancer le temps du client virtuel
                if not self.virtual_client.advance_time(current_time):
                    self.logger.warning("⚠️ Fin des données atteinte")
                    break

                # Log périodique
                if i % 100 == 0:
                    progress = (i - self.warmup_bars) / (total_bars - self.warmup_bars) * 100
                    balance = self.virtual_client.virtual_balance
                    self.logger.info(
                        f"📊 Progress: {progress:.1f}% | "
                        f"Date: {current_time.strftime('%Y-%m-%d %H:%M')} | "
                        f"Prix: ${current_price:.2f} | "
                        f"Balance: ${balance:.2f}"
                    )

                # === LOGIQUE DE TRADING (comme main.py) ===

                # 1. Récupérer les données de marché
                # Remplace TOUTE cette boucle par:

                market_data = {}
                timeframes_cfg = self.config['timeframes']
                tf_mapping = {
                    'trend': timeframes_cfg.get('trend', '1h'),
                    'signal': timeframes_cfg.get('signal', '5m'),
                    'micro': timeframes_cfg.get('micro', '1m')
                }

                # Itérer sur les NOMS seulement (trend, signal, micro)
                for tf_name in ['trend', 'signal', 'micro']:
                    actual_tf = tf_mapping[tf_name]  # '1h', '5m', '1m'

                    ohlcv = self.virtual_client.get_ohlcv(
                        self.config['symbols']['primary'],
                        actual_tf,
                        limit=200
                    )

                    # Convertir en DataFrame
                    df_tf = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df_tf['timestamp'] = pd.to_datetime(df_tf['timestamp'], unit='ms')
                    df_tf.set_index('timestamp', inplace=True)

                    market_data[actual_tf] = df_tf  # Clé = '1h', '5m', '1m'

                # 2. Générer signaux
                signals = self.bot.strategy_manager.generate_all_signals(market_data)

                # 3. Filtrer conflits
                filtered_signals = self.bot.strategy_manager.filter_conflicting_signals(signals)

                # 4. Exécuter signaux
                for signal in filtered_signals:
                    try:
                        position = self.bot.trade_executor.execute_signal(signal)
                        if position:
                            self.logger.info(f"✅ Position ouverte: {position.id}")
                    except Exception as e:
                        self.logger.error(f"❌ Erreur exécution signal: {e}")

                # 5. Monitorer positions existantes
                self.bot.trade_executor.monitor_positions()

            except Exception as e:
                current_time = current_bar.name if hasattr(locals().get('current_bar'), 'name') else 'unknown'
                self.logger.error(f"❌ Erreur à la bougie {i} ({current_time}): {e}")
                import traceback
                traceback.print_exc()
                continue

        # Fin du backtest
        self.logger.info("🏁 Replay terminé")
        self._print_results()

    def _print_results(self):
        """Affiche les résultats du backtest"""

        self.logger.info("=" * 70)
        self.logger.info("📊 RÉSULTATS DU REPLAY BACKTEST")
        self.logger.info("=" * 70)

        # Stats du client virtuel
        stats = self.virtual_client.get_statistics()

        initial_balance = float(self.backtest_config.get('simulation', {}).get('initial_balance', 1000))
        final_balance = stats['final_balance']
        pnl = final_balance - initial_balance
        pnl_pct = (pnl / initial_balance * 100) if initial_balance > 0 else 0

        self.logger.info(f"💰 Balance initiale: ${initial_balance:.2f}")
        self.logger.info(f"💰 Balance finale: ${final_balance:.2f}")
        self.logger.info(f"📈 PnL Total: ${pnl:.2f} ({pnl_pct:+.2f}%)")
        self.logger.info("")

        # Stats du trade executor
        if self.bot and hasattr(self.bot, 'trade_executor'):
            status = self.bot.trade_executor.get_status()
            pos_stats = status.get('position_manager', {})

            total_trades = pos_stats.get('total_trades', 0)
            winning = pos_stats.get('winning_trades', 0)
            losing = pos_stats.get('losing_trades', 0)
            win_rate = pos_stats.get('win_rate', 0)
            total_pnl = pos_stats.get('total_pnl', 0)

            self.logger.info(f"📊 Total trades: {total_trades}")
            self.logger.info(f"✅ Winning: {winning}")
            self.logger.info(f"❌ Losing: {losing}")
            self.logger.info(f"🎯 Win Rate: {win_rate:.2f}%")
            self.logger.info(f"💵 Total PnL: ${total_pnl:.2f}")

        self.logger.info("=" * 70)

def main():
    """Point d'entrée"""

    print("\n" + "=" * 70)
    print("🔄 QUANTUM TRADER PRO - REPLAY BACKTEST")
    print("=" * 70 + "\n")

    try:
        # Créer le backtest
        backtest = ReplayBacktest()

        # Charger données
        backtest.load_historical_data()

        # Préparer client virtuel
        backtest.prepare_virtual_client()

        # Lancer le replay
        backtest.run_replay()

        print("\n✅ Backtest terminé avec succès\n")

    except KeyboardInterrupt:
        print("\n⚠️ Backtest interrompu par l'utilisateur\n")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Erreur: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
