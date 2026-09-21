"""DSAGT MCP server package: the single merged ``dsagt-server``.

The MCP tool surface is split by concern across sibling modules:

* :mod:`dsagt.mcp.registry_tools`: the code registry and provenance
* :mod:`dsagt.mcp.knowledge_tools`: knowledge-base retrieval
* :mod:`dsagt.mcp.memory_tools`: explicit memory
* :mod:`dsagt.mcp.skill_tools`: skill search, install, and sources

:mod:`dsagt.mcp.server` composes all four under one ``Server("dsagt")`` and
owns the ``dsagt-server`` entry point and the shared-KB startup.
"""
