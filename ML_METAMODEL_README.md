# 🧠 ML Meta-Model Architecture - Quantum Trader Pro

## 📋 Vue d'ensemble

Cette architecture implémente un **système de sélection dynamique de stratégies** basé sur le Machine Learning. Le Meta-Model analyse le contexte marché en temps réel et sélectionne intelligemment quelle(s) stratégie(s) utiliser.

## 🎯 Philosophie

Au lieu d'avoir des poids **fixes** pour chaque stratégie, le système:
1. ✅ **Adapte** les stratégies au contexte marché
2. ✅ **Apprend** de la performance historique
3. ✅ **Désactive** automatiquement les stratégies sous-performantes
4. ✅ **Optimise** en continu les allocations

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    QUANTUM TRADER PRO                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   STRATEGY MANAGER                          │
│  - Coordonne toutes les stratégies                          │
│  - Intègre le ML Meta-Model                                 │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌──────────────────────┐    ┌──────────────────────────────┐
│  CLASSIC STRATEGIES  │    │      ML META-MODEL           │
│  - Ichimoku          │    │  🧠 Cerveau décisionnel      │
│  - Grid Trading      │    │                              │
│  - DCA Bot           │    └──────────┬───────────────────┘
│  - ML Strategy       │               │
└──────────┬───────────┘               │
           │                           │
           │        ┌──────────────────┴──────────────────┐
           │        ▼                                     ▼
           │  ┌────────────────────┐    ┌────────────────────────────┐
           │  │ Market Context     │    │ Performance Tracker        │
           │  │ Analyzer           │    │ - Win rate par stratégie   │
           │  │ - Trend            │    │ - Profit factor            │
           │  │ - Volatility       │    │ - Sharpe ratio             │
           │  │ - Volume           │    │ - Performance par contexte │
           │  │ - Momentum         │    │ - Losing/Winning streaks   │
           │  └────────────────────┘    └────────────────────────────┘
           │                  │                         │
           └──────────────────┴─────────────────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │  SIGNAL SELECTION      │
                  │  - Poids dynamiques    │
                  │  - Confidence ajustée  │
                  │  - Filtrage intelligent│
                  └────────────────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │   TRADE EXECUTOR       │
                  └────────────────────────┘
```

## 📦 Nouveaux Fichiers

### 1. **`ml_models/strategy_performance_tracker.py`**
**Rôle:** Suit la performance de chaque stratégie en temps réel

**Fonctionnalités:**
- Enregistre chaque trade (entry, exit, PnL)
- Calcule métriques par stratégie (win rate, profit factor, sharpe)
- Fenêtre glissante (ex: 7 derniers jours)
- Performance par contexte marché
- Détection de losing streaks
- Recommandation de désactivation

**Méthodes clés:**
```python
tracker.record_trade(strategy_name, signal_time, entry, exit, pnl, action, context)
tracker.get_strategy_metrics(strategy_name)  # → {win_rate, sharpe, etc.}
tracker.should_disable_strategy(strategy_name)  # → True/False
tracker.get_best_strategy_for_context(market_context)  # → strategy_name
```

### 2. **`ml_models/meta_model.py`**
**Rôle:** Le cerveau qui sélectionne dynamiquement les stratégies

**Modes de sélection:**
1. **`winner_takes_all`**: Choisit UNE seule stratégie (la meilleure)
2. **`weighted_ensemble`**: Combine toutes avec poids dynamiques
3. **`context_adaptive`**: Sélectionne selon le contexte marché

**Fonctionnalités:**
- Analyse contexte marché (trend, volatility, volume, momentum)
- Calcule score pour chaque stratégie
- Ajuste confidence des signaux selon poids
- Enregistre décisions pour analyse
- Désactive automatiquement stratégies sous-performantes

**Méthodes clés:**
```python
meta_model.analyze_market_context(data)  # → {trend, volatility, volume}
meta_model.select_strategies(all_signals, context)  # → {strategy: weight}
meta_model.adjust_signal_confidence(signal, strategy, weight)
```

### 3. **`strategies/ml_strategy.py`**
**Rôle:** Stratégie 100% ML (XGBoost + LSTM + Ensemble)

**Fonctionnalités:**
- Utilise `FeatureEngineering` pour features techniques
- Utilise `EnsembleModel` pour prédictions
- Génère signaux BUY/SELL avec confidence
- SL/TP basés sur ATR
- Minimum confidence et agreement configurable

**Workflow:**
```
Data → Feature Engineering → Ensemble Model → Signal
          (70+ features)     (XGBoost+LSTM)   (BUY/SELL)
```

### 4. **`strategies/strategy_manager.py`** (MODIFIÉ)
**Ajouts:**
- Intégration du Meta-Model
- Méthode `_meta_model_selection()` pour sélection intelligente
- Méthode `record_trade_result()` pour feedback au tracker
- Support de la ML Strategy

## 🔧 Configuration

Ajouter dans `config.yaml`:

```yaml
# ============================================================================
# MACHINE LEARNING
# ============================================================================
ml:
  # Meta-Model (Sélection dynamique de stratégies)
  meta_model:
    enabled: true
    selection_mode: weighted_ensemble  # winner_takes_all, weighted_ensemble, context_adaptive
    min_confidence: 0.6
    context_weight: 0.5  # Balance entre contexte (0.5) et performance (0.5)
    performance_window_days: 7  # Fenêtre glissante
  
  # Ensemble Model
  models:
    ensemble:
      method: weighted  # voting, weighted, stacking
      weights: [0.4, 0.4, 0.2]  # XGBoost, LSTM, autres
      min_confidence: 0.7
      min_agreement: 0.6

# ============================================================================
# STRATEGIES
# ============================================================================
strategies:
  # Stratégies existantes...
  ichimoku_scalping:
    enabled: true
    weight: 0.25
  
  grid_trading:
    enabled: true
    weight: 0.25
  
  dca_bot:
    enabled: true
    weight: 0.25
  
  # ✨ NOUVELLE: ML Strategy
  ml_strategy:
    enabled: true
    weight: 0.25
    models_path: ./ml_models/saved_models  # Path vers modèles entraînés
```

## 🚀 Utilisation

### Mode Production & Backtest

Le système fonctionne **identique** en production et en backtest:

```python
# Génération de signaux (pareil en prod et backtest)
all_signals = strategy_manager.generate_all_signals(data)

# Filtrage intelligent avec Meta-Model
filtered_signals = strategy_manager.filter_conflicting_signals(
    all_signals,
    data  # ← Nécessaire pour contexte marché
)

# Exécution
for signal in filtered_signals:
    trade_executor.execute_signal(signal)
```

### Feedback de Performance

Après chaque trade fermé:

```python
# Enregistrer résultat pour le tracker
strategy_manager.record_trade_result(
    strategy_name=position.strategy,
    signal_time=position.entry_time,
    entry_price=position.entry_price,
    exit_price=exit_price,
    exit_time=datetime.now(),
    pnl=position.pnl,
    action=position.side,
    market_context=current_context
)
```

Le tracker met à jour les métriques et le Meta-Model adapte automatiquement.

## 📊 Exemple de Fonctionnement

### Scénario 1: Marché en Tendance Haussière

```
Contexte: trend=bullish, volatility=low, volume=high
Performance tracker:
  - Ichimoku: win_rate=72%, sharpe=1.8 (excellent en trend)
  - Grid: win_rate=45%, sharpe=0.3 (mauvais en trend)
  - DCA: win_rate=55%, sharpe=0.9 (neutre)
  - ML: win_rate=68%, sharpe=1.5 (bon)

Meta-Model décision (weighted_ensemble):
  - Ichimoku: 45% ✅ (performance excellente)
  - Grid: 5% ⚠️ (sous-performe)
  - DCA: 20%
  - ML: 30%
```

### Scénario 2: Marché en Range

```
Contexte: trend=neutral, volatility=low, volume=normal
Performance tracker:
  - Ichimoku: win_rate=42%, sharpe=-0.2 (faux signaux)
  - Grid: win_rate=78%, sharpe=2.1 (excellent en range!)
  - DCA: win_rate=51%, sharpe=0.6
  - ML: win_rate=62%, sharpe=1.2

Meta-Model décision (context_adaptive):
  - Grid: 70% ✅ (spécialisée pour range)
  - DCA: 15%
  - ML: 15%
  - Ichimoku: DÉSACTIVÉE ❌ (losing streak)
```

## 🎯 Avantages

1. **Adaptation Automatique**
   - Le système s'adapte sans intervention manuelle
   - Désactive les stratégies sous-performantes
   - Booste les stratégies qui marchent

2. **Robustesse**
   - Diversification intelligente
   - Pas de dépendance à une seule stratégie
   - Résilience aux changements de marché

3. **Performance Optimale**
   - Toujours utiliser la meilleure approche
   - Évite les trades perdants
   - Maximise le profit factor

4. **Production = Backtest**
   - Même code en backtest et prod
   - Si ça marche en backtest, ça marchera en prod
   - Pas de divergence

## 📈 Métriques de Suivi

Le système expose plusieurs métriques:

```python
# Stats du Meta-Model
stats = strategy_manager.get_meta_model_stats()
# → {
#     'enabled': True,
#     'selection_mode': 'weighted_ensemble',
#     'decision_stats': {...},
#     'performance_tracker': {...}
# }

# Performance individuelle des stratégies
perf = strategy_manager.get_all_performance_stats()
# → {
#     'ichimoku': {win_rate, total_pnl, sharpe, ...},
#     'grid': {...},
#     'dca': {...},
#     'ml': {...}
# }
```

## 🔮 Évolutions Futures

1. **Reinforcement Learning**
   - Le Meta-Model pourrait être un RL agent
   - Apprend directement de l'environnement
   - Maximise PnL long-terme

2. **Détection de Régime**
   - Identifier automatiquement bull/bear/range
   - Switcher entre modes optimisés

3. **Auto-tuning**
   - Optimiser automatiquement les paramètres
   - A/B testing des stratégies
   - Evolution génétique des configs

## ✅ Checklist d'Intégration

- [x] ✅ `strategy_performance_tracker.py` créé
- [x] ✅ `meta_model.py` créé
- [x] ✅ `ml_strategy.py` créé
- [x] ✅ `strategy_manager.py` modifié
- [ ] ⏳ Ajouter config ML dans `config.yaml`
- [ ] ⏳ Tester en backtest
- [ ] ⏳ Entraîner modèles ML
- [ ] ⏳ Valider en production

## 🚦 Prochaines Étapes

1. **Ajouter la config ML** dans `config.yaml`
2. **Tester le backtest** avec Meta-Model activé
3. **Analyser les résultats** - voir quelles stratégies sont sélectionnées
4. **Affiner les seuils** (min_confidence, weights, etc.)
5. **Entraîner les modèles ML** pour activer ml_strategy
6. **Déployer en production** une fois validé

---

**Architecture by:** Quantum Trader Pro Team
**Date:** 2025-11-15
**Version:** 4.0 - ML Meta-Model Edition 🧠
