"""
Thread Safety Manager - Quantum Trader Pro
Gestion de la concurrence et synchronisation des threads
"""

import threading
from typing import Dict, Any, Optional
from contextlib import contextmanager
from utils.logger import setup_logger

class ThreadSafetyManager:
    """
    Gestionnaire de sécurité des threads pour éviter les race conditions
    """
    
    def __init__(self):
        self.logger = setup_logger('ThreadSafetyManager')
        
        # Locks pour différentes ressources
        self.locks = {
            'orders': threading.RLock(),  # Pour placer/annuler ordres
            'positions': threading.RLock(),  # Pour gérer positions
            'balance': threading.RLock(),  # Pour mise à jour balance
            'state': threading.RLock(),  # Pour sauvegarde état
            'market_data': threading.RLock(),  # Pour données marché
            'strategy': threading.RLock(),  # Pour signaux stratégie
        }
        
        # Compteurs thread-safe
        self._counters = {}
        self._counter_lock = threading.Lock()
        
        self.logger.info("✅ Thread Safety Manager initialisé")
    
    @contextmanager
    def lock(self, resource: str):
        """
        Context manager pour lock une ressource
        
        Usage:
            with thread_manager.lock('orders'):
                # Code critique pour ordres
                pass
        """
        if resource not in self.locks:
            self.logger.warning(f"Ressource inconnue: {resource}, création d'un nouveau lock")
            self.locks[resource] = threading.RLock()
        
        lock = self.locks[resource]
        
        try:
            lock.acquire()
            yield
        finally:
            lock.release()
    
    def synchronized_method(self, resource: str):
        """
        Décorateur pour synchroniser une méthode entière
        
        Usage:
            @thread_manager.synchronized_method('orders')
            def place_order(self, ...):
                pass
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                with self.lock(resource):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def increment_counter(self, name: str, value: int = 1) -> int:
        """
        Incrémente un compteur de manière thread-safe
        
        Args:
            name: Nom du compteur
            value: Valeur à ajouter
            
        Returns:
            Nouvelle valeur du compteur
        """
        with self._counter_lock:
            if name not in self._counters:
                self._counters[name] = 0
            self._counters[name] += value
            return self._counters[name]
    
    def get_counter(self, name: str) -> int:
        """Récupère la valeur d'un compteur"""
        with self._counter_lock:
            return self._counters.get(name, 0)
    
    def reset_counter(self, name: str):
        """Reset un compteur"""
        with self._counter_lock:
            self._counters[name] = 0

class OrderQueue:
    """
    File d'attente thread-safe pour les ordres
    """
    
    def __init__(self, max_size: int = 100):
        self.queue = []
        self.max_size = max_size
        self.lock = threading.RLock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)
    
    def put(self, order: Dict, timeout: Optional[float] = None) -> bool:
        """
        Ajoute un ordre à la queue
        
        Args:
            order: Ordre à ajouter
            timeout: Timeout en secondes
            
        Returns:
            True si ajouté avec succès
        """
        with self.not_full:
            while len(self.queue) >= self.max_size:
                if not self.not_full.wait(timeout):
                    return False
            
            self.queue.append(order)
            self.not_empty.notify()
            return True
    
    def get(self, timeout: Optional[float] = None) -> Optional[Dict]:
        """
        Récupère un ordre de la queue
        
        Args:
            timeout: Timeout en secondes
            
        Returns:
            Ordre ou None si timeout
        """
        with self.not_empty:
            while not self.queue:
                if not self.not_empty.wait(timeout):
                    return None
            
            order = self.queue.pop(0)
            self.not_full.notify()
            return order
    
    def size(self) -> int:
        """Retourne la taille de la queue"""
        with self.lock:
            return len(self.queue)
    
    def clear(self):
        """Vide la queue"""
        with self.lock:
            self.queue.clear()
            self.not_full.notify_all()

# Instance globale
thread_manager = ThreadSafetyManager()

__all__ = ['ThreadSafetyManager', 'OrderQueue', 'thread_manager']
