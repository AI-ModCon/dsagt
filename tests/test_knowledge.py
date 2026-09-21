def test_list_collections_names_purpose_keys_and_count(tmp_path):
    """Every collection carries its purpose, its chunks' metadata keys, and a
    count; dsagt's own collections are listed before their first write."""
    import json as _json

    from dsagt.knowledge import KnowledgeBase

    index = tmp_path / "kb_index"
    coll = index / "code_use"
    coll.mkdir(parents=True)
    (coll / "chroma_ids.json").write_text("[]")
    with open(coll / "chunks.jsonl", "w") as fh:
        for i in range(3):
            fh.write(
                _json.dumps(
                    {
                        "text": f"Code: fastp {i}",
                        "metadata": {
                            "code_name": "fastp",
                            "session_id": "s1",
                            "return_code": 0,
                        },
                    }
                )
                + "\n"
            )
    kb = KnowledgeBase(index, default_embedder="local")
    listed = {c["name"]: c for c in kb.list_collections()}
    assert listed["code_use"]["chunk_count"] == 3
    assert listed["code_use"]["metadata_keys"] == [
        "code_name",
        "return_code",
        "session_id",
    ]
    assert "execution record" in listed["code_use"]["description"]
    # Not written yet; listed with its purpose so the agent can find it.
    assert listed["codes"]["chunk_count"] == 0
    assert "search_registry" in listed["codes"]["description"]
    assert listed["session_memory"]["metadata_keys"] == []


def test_a_removed_entry_leaves_the_others_findable(tmp_path):
    """An id is opaque, so removing one entry does not move the rest.

    A search hit is read as a position in the chunk list and merged with the
    BM25 leg's positions, so the id list and the chunk list have to describe
    the same rows after a removal.
    """
    import numpy as np

    from dsagt.knowledge import ChromaIndex

    coll = tmp_path / "codes"
    coll.mkdir(parents=True)
    index = ChromaIndex(collection_name="codes", persist_dir=coll)
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=np.float32,
    )
    index.add(vectors, metadatas=[{"code_name": n} for n in ("a", "b", "c")])
    assert index._ids == ["0", "1", "2"]

    index.remove({0})
    # The survivors keep their ids and become positions 0 and 1.
    assert index._ids == ["1", "2"]
    _, hits = index.search(np.array([0.0, 1.0], dtype=np.float32), 1)
    assert list(hits) == [0]  # 'b', now the first chunk
    _, hits = index.search(np.array([-1.0, 0.0], dtype=np.float32), 1)
    assert list(hits) == [1]  # 'c', now the second

    # A later add cannot reuse a live id.
    index.add(np.array([[0.0, -1.0]], dtype=np.float32), metadatas=[{"code_name": "d"}])
    assert index._ids == ["1", "2", "3"]


def test_delete_entries_rewrites_the_chunk_file_and_counts(tmp_path, monkeypatch):
    """delete_entries drops the matching rows from every leg of the store."""
    import numpy as np

    from dsagt.knowledge import KnowledgeBase

    index_dir = tmp_path / "kb_index"
    kb = KnowledgeBase(index_dir, default_embedder="local")
    monkeypatch.setattr(
        kb._store,
        "embed",
        lambda texts: np.eye(max(len(texts), 2), dtype=np.float32)[: len(texts)],
    )
    kb.add_entries(
        texts=["code a", "code b"],
        collection="codes",
        metadatas=[{"code_name": "a"}, {"code_name": "b"}],
    )
    assert kb.delete_entries("codes", {"code_name": "a"}) == 1
    assert kb.delete_entries("codes", {"code_name": "a"}) == 0

    import json as _json

    rows = [
        _json.loads(line)
        for line in (index_dir / "codes" / "chunks.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [r["metadata"]["code_name"] for r in rows] == ["b"]
    assert _json.loads((index_dir / "codes" / "chroma_ids.json").read_text()) == ["1"]
