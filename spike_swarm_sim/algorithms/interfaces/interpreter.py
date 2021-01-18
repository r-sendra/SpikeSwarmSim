from functools import wraps
language_dict = {}

def GET(query):
    # @wraps()
    def _GET(func):
        query_elements = ['GET'] + query.split(':')
        aux_dict = language_dict
        for i, elem in enumerate(query_elements):
            if elem not in aux_dict.keys():
                aux_dict[elem] = {}
            if i < len(query_elements) - 1:
                aux_dict = aux_dict[elem]
            else:
                aux_dict[elem] = func.__name__   
        @wraps(func)
        def wrapper(*args, **kwargs):      
            return func(*args, **kwargs)
        return wrapper
    return _GET 

def SET(query):
    def _SET(func):
        query_elements = ['SET'] + query.split(':')
        aux_dict = language_dict
        for i, elem in enumerate(query_elements):
            if elem not in aux_dict.keys():
                aux_dict[elem] = {}
            if i < len(query_elements) - 1:
                aux_dict = aux_dict[elem]
            else:
                aux_dict[elem] = func.__name__
        @wraps(func)        
        def wrapper(*args, **kwargs):      
            return func(*args, **kwargs)
        return wrapper
    return _SET

def LEN(query):
    def _LEN(func):
        query_elements = ['LEN'] + query.split(':')
        aux_dict = language_dict
        for i, elem in enumerate(query_elements):
            if elem not in aux_dict.keys():
                aux_dict[elem] = {}
            if i < len(query_elements) - 1:
                aux_dict = aux_dict[elem]
            else:
                aux_dict[elem] = func.__name__
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return _LEN

def INIT(query):
    def _INIT(func):
        query_elements = ['INIT'] + query.split(':')
        aux_dict = language_dict
        for i, elem in enumerate(query_elements):
            if elem not in aux_dict.keys():
                aux_dict[elem] = {}
            if i < len(query_elements) - 1:
                aux_dict = aux_dict[elem]
            else:
                aux_dict[elem] = func.__name__
        @wraps(func)            
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return _INIT