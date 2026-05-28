import time
import threading
from typing import Dict, Any, Optional

class MemoryCache:
    def __init__(self, ttl_seconds: int = 1800):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def set(self, key: str, value: Any) -> None:
        """Stores a value in the cache with a timestamp."""
        with self._lock:
            self._cache[key] = {
                "value": value,
                "expires_at": time.time() + self._ttl
            }

    def get(self, key: str) -> Optional[Any]:
        """Retrieves a cached value, automatically handling expiration."""
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None
            
            if time.time() > entry["expires_at"]:
                del self._cache[key]
                return None
                
            return entry["value"]

    def delete(self, key: str) -> None:
        """Deletes a key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]

    def cleanup(self) -> None:
        """Purges expired items from cache."""
        now = time.time()
        with self._lock:
            expired_keys = [k for k, v in self._cache.items() if now > v["expires_at"]]
            for k in expired_keys:
                del self._cache[k]

# Global cache instance
upload_cache = MemoryCache()
