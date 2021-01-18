import numpy as np
from functools import wraps
from .interpreter import language_dict

class GeneticInterface:
    def __init__(self, neural_net):
        self.neural_net = neural_net 

    def submit_query(self, query, primitive='GET', **kwargs):
        query_hierarchy = [primitive] + query.split(':')
        query_status = language_dict 
        for query_elem in query_hierarchy[:-1]:
            assert query_elem in query_status
            query_status = query_status[query_elem]
        func = getattr(getattr(self.neural_net, query_hierarchy[1]), query_status)\
               if query_hierarchy[1] not in ['decoding', 'encoding', ]\
               else getattr(self.neural_net, query_status)
        return func(query_hierarchy[-1], **kwargs)

    def toGenotype(self, queries, min_vals, max_vals):
        genotype = np.hstack([self.submit_query(query, primitive='GET',\
                   min_val=min_val, max_val=max_val) \
                   for query, max_val, min_val in zip(queries, max_vals, min_vals)])
        return genotype

    def fromGenotype(self, queries, genotype, min_vals, max_vals):
        counter = 0
        for query, max_val, min_val in zip(queries, max_vals, min_vals):
            segment_len = self.submit_query(query, primitive='LEN')
            genotype_segment = genotype[counter:counter+segment_len].copy()
            self.submit_query(query, primitive='SET', data=genotype_segment, min_val=min_val, max_val=max_val)
            counter += segment_len
    
    def initGenotype(self, queries, min_vals, max_vals):
        for query, max_val, min_val in zip(queries, max_vals, min_vals):
            self.submit_query(query, primitive='INIT', min_val=min_val, max_val=max_val)
