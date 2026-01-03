# Classes de données pour CLEVR
import re
import torch_geometric

class ClevrData(torch_geometric.data.Data):
    """Classe de données pour CLEVR avec support des edge_attr"""

    def __init__(self, src=None, **kwargs):
        super(ClevrData, self).__init__(**kwargs)
        self.src = src  # 'Gs' | 'Gt'

    def __cat_dim__(self, key, value):
        """Retourne la dimension pour la concaténation"""
        return -1 if bool(re.search('(index|face)', key)) else 0

    def __inc__(self, key, value):
        """Incrémente pour edge_index et edge_attr"""
        return self.num_nodes if bool(re.search('(index|face)', key)) else 0
