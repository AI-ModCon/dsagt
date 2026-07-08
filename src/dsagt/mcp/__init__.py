"""DSAGT MCP server package — the single merged ``dsagt-server``.

The MCP tool surface is split by concern across sibling modules:

* :mod:`dsagt.mcp.registry_tools` — tool registry + execution + provenance
* :mod:`dsagt.mcp.knowledge_tools` — knowledge-base retrieval
* :mod:`dsagt.mcp.memory_tools` — explicit memory
* :mod:`dsagt.mcp.skill_tools` — skill search / install / sources

:mod:`dsagt.mcp.server` composes all four under one ``Server("dsagt")`` and
owns the ``dsagt-server`` entry point + shared-KB startup.
"""
