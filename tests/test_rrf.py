from app.kb.search.rrf import reciprocal_rank_fusion


def test_rrf_combines_and_ranks():
    dense = [{"chunk_id": "a", "text": "A"}, {"chunk_id": "b", "text": "B"}]
    sparse = [{"chunk_id": "b", "text": "B"}, {"chunk_id": "c", "text": "C"}]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    ids = [f["chunk_id"] for f in fused]
    assert ids[0] == "b"  # present in both lists -> highest fused score
    assert set(ids) == {"a", "b", "c"}
    assert all("rrf_score" in f for f in fused)


def test_rrf_ignores_empty_list():
    dense = [{"chunk_id": "a", "text": "A"}]
    fused = reciprocal_rank_fusion([dense, []])
    assert [f["chunk_id"] for f in fused] == ["a"]
