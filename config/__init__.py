"""
Configuration Module - Quantum Trader Pro
Chargement et validation de la configuration avec gestion des clés API
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import sys

class ConfigError(Exception):
    """Erreur de configuration"""
    pass

class ConfigLoader:
    """Chargeur de configuration avec validation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or Path(__file__).parent / 'config.yaml'
        self.env_path = Path(__file__).parent.parent / '.env'
        
        # Charger .env
        self._load_env()
        
        # Charger YAML
        self.config = self._load_yaml()
        
        # Injecter variables d'environnement
        self._inject_env_vars()
        
        # Valider configuration
        self._validate()
    
    def _load_env(self):
        """Charge les variables d'environnement depuis .env"""
        if not self.env_path.exists():
            print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║                    ⚠️  FICHIER .env MANQUANT                     ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Le fichier .env est OBLIGATOIRE pour configurer les clés API.   ║
║                                                                   ║
║  📋 INSTRUCTIONS:                                                ║
║  1. Copiez .env.example vers .env                                ║
║  2. Éditez .env avec vos clés API                                ║
║  3. Relancez le programme                                        ║
║                                                                   ║
║  🔑 CLÉS TESTNET BINANCE:                                        ║
║  → https://testnet.binance.vision/                               ║
║  → Créez un compte et générez vos clés API                       ║
║                                                                   ║
║  ⚠️  NE PAS utiliser vos clés de production!                     ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
            """)
            sys.exit(1)
        
        load_dotenv(self.env_path)
        print("✅ Fichier .env chargé")
    
    def _load_yaml(self) -> Dict[str, Any]:
        """Charge configuration depuis YAML"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ Configuration chargée depuis {self.config_path}")
            return config
        except FileNotFoundError:
            raise ConfigError(f"❌ Fichier config.yaml introuvable: {self.config_path}")
        except yaml.YAMLError as e:
            raise ConfigError(f"❌ Erreur parsing YAML: {e}")
    
    def _inject_env_vars(self):
        """Injecte les variables d'environnement dans la config"""
        
        # BINANCE API
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        private_key_path = os.getenv('BINANCE_PRIVATE_KEY_PATH', 'test-prv-key.pem')
        testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
        
        if not api_key:
            print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║                 ⚠️  CLÉS API MANQUANTES                            ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Les clés API Binance ne sont pas configurées dans .env           ║
║                                                                   ║
║  📋 POUR TESTNET (RSA):                                           ║
║  1. Générez les clés RSA:                                         ║
║     openssl genrsa -out test-prv-key.pem 2048                     ║
║     openssl rsa -in test-prv-key.pem -pubout -out test-pub-key.pem║
║                                                                   ║
║  2. Enregistrez test-pub-key.pem sur:                             ║
║     https://testnet.binance.vision/                               ║
║                                                                   ║
║  3. Mettez l'API Key dans .env:                                   ║
║     BINANCE_API_KEY=votre_api_key_testnet                         ║
║     BINANCE_PRIVATE_KEY_PATH=test-prv-key.pem                     ║
║     BINANCE_TESTNET=true                                          ║
║                                                                   ║
║  📋 POUR PRODUCTION (HMAC):                                       ║
║     BINANCE_API_KEY=votre_api_key                                 ║
║     BINANCE_SECRET_KEY=votre_secret_key                         b ║
║     BINANCE_TESTNET=false                                       b ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
            """)
            sys.exit(1)
        
        # Pour testnet, pas besoin de secret_key mais de private_key_path
        if testnet and not os.path.exists(private_key_path):
            if not secret_key:  # Si pas de fichier PEM, on peut essayer avec secret (compatibilité)
                print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║              ⚠️  CONFIGURATION TESTNET RSA                       ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Le testnet Binance nécessite maintenant RSA au lieu de HMAC!    ║
║                                                                   ║
║  Fichier clé privée non trouvé: {private_key_path:<33} ║
║                                                                   ║
║  Générez les clés avec:                                          ║
║  openssl genrsa -out test-prv-key.pem 2048                       ║
║  openssl rsa -in test-prv-key.pem -pubout -out test-pub-key.pem  ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
                """)
        
        # Pour production, on a besoin du secret_key
        if not testnet and not secret_key:
            print("❌ Secret Key requis pour le mode production!")
            sys.exit(1)
        
        # Injecter dans config
        self.config['exchange']['primary']['api_key'] = api_key
        self.config['exchange']['primary']['secret_key'] = secret_key
        self.config['exchange']['primary']['private_key_path'] = private_key_path
        self.config['exchange']['primary']['testnet'] = testnet
        
        # Symbol
        if symbol := os.getenv('SYMBOL'):
            self.config['symbols']['primary'] = symbol
        
        # Capital
        if initial_capital := os.getenv('INITIAL_CAPITAL'):
            self.config['capital']['initial'] = float(initial_capital)
        
        # Risk
        if max_risk := os.getenv('MAX_RISK_PER_TRADE'):
            self.config['risk']['max_risk_per_trade_percent'] = float(max_risk)
        
        if max_daily_loss := os.getenv('MAX_DAILY_LOSS'):
            self.config['risk']['max_daily_loss_percent'] = float(max_daily_loss)
        
        if max_positions := os.getenv('MAX_POSITIONS'):
            self.config['risk']['max_positions_simultaneous'] = int(max_positions)
        
        # Telegram
        if telegram_enabled := os.getenv('TELEGRAM_ENABLED'):
            self.config['monitoring']['telegram']['enabled'] = telegram_enabled.lower() == 'true'
        
        if telegram_token := os.getenv('TELEGRAM_BOT_TOKEN'):
            self.config['monitoring']['telegram']['bot_token'] = telegram_token
        
        if telegram_chat_id := os.getenv('TELEGRAM_CHAT_ID'):
            self.config['monitoring']['telegram']['chat_id'] = telegram_chat_id
        
        # ML
        if ml_enabled := os.getenv('ML_ENABLED'):
            self.config['ml']['enabled'] = ml_enabled.lower() == 'true'
        
        # Database
        if db_url := os.getenv('DATABASE_URL'):
            self.config['database']['url'] = db_url
        
        # Logging
        if log_level := os.getenv('LOG_LEVEL'):
            self.config['logging']['level'] = log_level
        
        print("✅ Variables d'environnement injectées")
    
    def _validate(self):
        """Valide la configuration"""
        errors = []
        
        # Validation Exchange
        exchange = self.config.get('exchange', {}).get('primary', {})
        testnet = exchange.get('testnet')

        if not exchange.get('api_key'):
            errors.append("❌ API Key manquante")
        
        if not exchange.get('secret_key'):
            if testnet :
                print("✅ Ok")
            else:
                errors.append("❌ Secret Key manquante")
        
        # Vérifier longueur des clés (Binance keys are ~64 chars)
        if exchange.get('api_key') and len(exchange['api_key']) < 20:
            errors.append("⚠️  API Key semble invalide (trop courte)")
        
        if exchange.get('secret_key') and len(exchange['secret_key']) < 20:
            errors.append("⚠️  Secret Key semble invalide (trop courte)")
        
        # Validation Capital
        capital = self.config.get('capital', {})
        if capital.get('initial', 0) < 100:
            errors.append("⚠️  Capital initial < 100 USDT (risqué)")
        
        # Validation Risk
        risk = self.config.get('risk', {})
        if risk.get('max_risk_per_trade_percent', 0) > 5:
            errors.append("⚠️  Risk per trade > 5% (très risqué!)")
        
        if risk.get('max_daily_loss_percent', 0) > 20:
            errors.append("⚠️  Max daily loss > 20% (extrêmement risqué!)")
        
        # Validation Symbols
        symbol = self.config.get('symbols', {}).get('primary')
        if not symbol:
            errors.append("❌ Symbol principal manquant")
        elif '/' not in symbol:
            errors.append(f"❌ Symbol invalide: {symbol} (format: BASE/QUOTE)")
        
        if errors:
            print("\n" + "="*70)
            print("❌ ERREURS DE CONFIGURATION")
            print("="*70)
            for error in errors:
                print(f"  {error}")
            print("="*70)
            
            if any("❌" in e for e in errors):
                print("\n⛔ Erreurs critiques détectées. Arrêt du programme.")
                sys.exit(1)
            else:
                print("\n⚠️  Warnings détectés. Continuez à vos risques et périls.")
                input("Appuyez sur Entrée pour continuer...")
        else:
            print("✅ Configuration validée")
    
    def get(self) -> Dict[str, Any]:
        """Retourne la configuration complète"""
        return self.config
    
    def display_summary(self):
        """Affiche un résumé de la configuration"""
        exchange = self.config['exchange']['primary']
        capital = self.config['capital']
        risk = self.config['risk']
        symbol = self.config['symbols']['primary']
        
        testnet_indicator = "🧪 TESTNET" if exchange['testnet'] else "⚠️  PRODUCTION"
        
        print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║                QUANTUM TRADER PRO - CONFIGURATION                 ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  📊 MODE: {testnet_indicator:<52} ║
║  💰 Capital Initial: ${capital['initial']:<38.2f} ║
║  🎯 Symbol: {symbol:<49} ║
║  📈 Risk per Trade: {risk['max_risk_per_trade_percent']:<43.1f}% ║
║  🛡️  Max Daily Loss: {risk['max_daily_loss_percent']:<42.1f}% ║
║  📦 Max Positions: {risk['max_positions_simultaneous']:<45} ║
║                                                                   ║
║  🔑 API Key: {exchange['api_key'][:20]}...{exchange['api_key'][-4:]:<21} ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
        """)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def load_config(config_path: Optional[str] = None, display_summary: bool = True) -> Dict[str, Any]:
    """
    Charge et retourne la configuration

    Args:
        config_path: Chemin vers config.yaml (optionnel)
        display_summary: Afficher le résumé de configuration

    Returns:
        Dictionnaire de configuration
    """
    loader = ConfigLoader(config_path)
    if display_summary:
        loader.display_summary()
    return loader.get()


# Cache pour singleton paresseux
_config_cache: Optional[Dict[str, Any]] = None
_config_loader_cache: Optional[ConfigLoader] = None


def get_config() -> Dict[str, Any]:
    """
    Récupère la configuration (chargement paresseux)

    Charge la configuration une seule fois et la met en cache.
    Équivalent de l'ancien CONFIG global mais en lazy loading.

    Returns:
        Dictionnaire de configuration
    """
    global _config_cache, _config_loader_cache

    if _config_cache is None:
        _config_loader_cache = ConfigLoader()
        _config_loader_cache.display_summary()
        _config_cache = _config_loader_cache.get()

    return _config_cache


def reset_config_cache() -> None:
    """
    Reset le cache de configuration

    Utile pour les tests ou rechargement dynamique.
    """
    global _config_cache, _config_loader_cache
    _config_cache = None
    _config_loader_cache = None


# NOTE: Plus de chargement automatique à l'import!
# Utilisez ConfigLoader() directement ou appelez get_config() pour lazy loading.

# Export
__all__ = ['load_config', 'get_config', 'reset_config_cache', 'ConfigLoader', 'ConfigError']
