import logging 
from contextlib import contextmanager
from typing import Any
from ..api import VectorDB, DBCaseConfig
from shapelets.storage import RecordStore, KnnOptions, MetricType
from shapelets.data import DataType

log = logging.getLogger(__name__)
service = RecordStore.start()
knnopt = KnnOptions()
knnopt.include_record = True
knnopt.include_embedding = False

class ShapeletsClient(VectorDB):
    """Shapelets client for VectorDB. 
    """ 

    def __init__(
            self,
            dim: int,
            db_config: dict,
            db_case_config: DBCaseConfig,
            drop_old: bool = False,
            **kwargs
        ):
        #self.db_config = db_config
        #self.db_config["host"] = "127.0.0.1"
        #self.db_config["port"] = 8500
        #self.case_config = db_case_config
        self.collection_name = 'example'
        service.create_catalog(self.collection_name, {'embedding':DataType.embedding(dim, MetricType.Cosine)})
        
    @contextmanager
    def init(self) -> None:
        """ create and destory connections to database.
        """
        yield
        self.client = None
        self.collection = None

    def ready_to_search(self) -> bool:
        pass

    def ready_to_load(self) -> bool:
        pass

    def optimize(self) -> None:
        pass

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> (int, Exception):
        """Insert embeddings into the database.

        Args:
            embeddings(list[list[float]]): list of embeddings
            metadata(list[int]): list of metadata
            kwargs: other arguments

        Returns:
            (int, Exception): number of embeddings inserted and exception if any
        """
        ids=[str(i) for i in metadata]
        #metadata = [{"id": int(i)} for i in metadata] 

        index = service.open_catalog(self.collection_name)
        loader = index.create_loader()
        if len(embeddings) > 0:
            vectors_per_request = 5000
            for i in range(0,len(embeddings), vectors_per_request):
                data = [{'embedding': emb, 'id':id} for emb, id in zip(embeddings[i:i+vectors_per_request],ids[i:i+vectors_per_request])]
                for entry in data:
                    loader.append({'id':int(entry['id'])},{'embedding':entry['embedding']})
            loader.finalize()     
        return len(embeddings), None
    
    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Search embeddings from the database.
        Args:
            embedding(list[float]): embedding to search
            k(int): number of results to return
            kwargs: other arguments

        Returns:
            Dict {ids: list[list[int]], 
                    embedding: list[list[float]] 
                    distance: list[list[float]]}
        """
        index = service.open_catalog(self.collection_name)
        knnResult = index.knn(query, k, options = knnopt)
        return [r.record['id'] for r in knnResult]
        
    

