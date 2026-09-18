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
