# -*- coding: utf-8 -*-
import pandas as pd

def sub_alignment_with_head(part_entities, alignment):
    print("index {0}".format(4))
    if not isinstance(part_entities, set):
        part_entities = set(part_entities)
    bucket = []
    for e1, e2 in alignment:
        if e1 in part_entities:
            bucket.append((e1, e2))
    return bucket

def sub_alignment_with_tail(part_entities, alignment):
    if not isinstance(part_entities, set):
        part_entities = set(part_entities)
    bucket = []
    for e1, e2 in alignment:
        if e2 in part_entities:
            bucket.append((e1, e2))
    return bucket

def get_neighbours(conn_df, part_entities, max_hop_k):
    added_entity_set = set(part_entities)
    neighbours_list = []
    ent_df = pd.Series(data=part_entities).to_frame("ent")
    # The following part is applied to obtain the k hop neighbors.
    for step in range(0, max_hop_k):
        tmp_triple_df = ent_df.merge(conn_df, how="inner", left_on="ent", right_on="h") 
        # The shape of 'tmp_triple_df' is (N, 3). The elements in the first two columns are same. The name of columns is "ent      h      t"
        tmp_triple_df2 = ent_df.merge(conn_df, how="inner", left_on="ent", right_on="t")
        # The shape of 'tmp_triple_df2' is (N, 3). The elements in first and third columns are same. The name of columns is "ent      h      t"
        new_hop_entities = set(tmp_triple_df["t"].tolist()).difference(added_entity_set) 
        # Can help us find the difference between two sets. The elements of the returned collection are contained 
        # in the first collection, but not in the second collection.
        new_hop_entities2 = set(tmp_triple_df2["h"].tolist()).difference(added_entity_set)
        new_hop_entities.update(new_hop_entities2)
        added_entity_set.update(new_hop_entities)
        if len(new_hop_entities) == 0:
            break
        neighbours_list.append(new_hop_entities)
        ent_df = pd.Series(data=list(new_hop_entities)).to_frame("ent")
    return neighbours_list
