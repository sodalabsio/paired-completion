from sqlitedict import SqliteDict
from typing import Union
import dill

class ValueCache:
    # Cached values go from (model_name, text) -> value
    # @staticmethod
    def __init__(self, name):
        self.name = name
        self.db = SqliteDict("./" + name + ".db", autocommit=True)

    def get(self, key: str, allow_miss=True) -> Union[float, None]:
        # Get the value from the cache
        if key in self.db:
            return self.db[key]
        
        if allow_miss:
            return None
        
        raise Exception("Cache miss on:", key)
    
    def set(self, key: str, value: float):
        # print("Setting cache value {} = {}".format(key, value))
        self.db[key] = value

    def register(self):
        def decorator(f):
            # Create a wrapper that checks the cache first. if the cache hits, return the value
            # Otherwise, call the function and cache the result
            def wrapper(*args, **kwargs):
                if "temperature" in kwargs and kwargs["temperature"] > 0.0:
                    # Don't cache when the temperature is nonzero
                    return f(*args, **kwargs)
                
                # Get the key by concatenating the __str__ of the arguments
                argstr = "||".join([str(arg) for arg in args])
                kwargstr = "||".join([str(kwarg) for kwarg in kwargs.items()])
                # fnstr = dill.dumps(f)
                key = self.name + "->" + argstr + "||" + kwargstr

                # Check the cache
                cached_value = self.get(key)
                if cached_value is not None:
                    return cached_value

                # Call the function
                value = f(*args, **kwargs)

                # Cache the value and return it
                self.set(key, value)
                return value

            return wrapper
        return decorator
