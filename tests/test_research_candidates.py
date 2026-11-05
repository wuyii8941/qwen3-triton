import torch
from benchmarks.research_candidates import partition_parameters,tensor_storage_bytes


class TiedModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(16,8)
        self.head=torch.nn.Linear(8,16,bias=False);self.head.weight=self.embedding.weight
        self.hidden=torch.nn.Linear(8,8,bias=False);self.norm=torch.nn.LayerNorm(8)
    def get_input_embeddings(self):return self.embedding
    def get_output_embeddings(self):return self.head


def test_muon_partition_keeps_tied_embedding_and_norm_in_auxiliary_adam():
    model=TiedModel();hidden,aux=partition_parameters(model)
    assert [id(p) for p in hidden]==[id(model.hidden.weight)]
    assert id(model.embedding.weight) in {id(p) for p in aux}
    assert len(hidden)+len(aux)==len(list(model.parameters()))
    assert not {id(p) for p in hidden}&{id(p) for p in aux}


def test_storage_counter_deduplicates_views():
    value=torch.arange(8,dtype=torch.float32)
    assert tensor_storage_bytes(dict(first=value,view=value[:2],scalar=1))==32
