"""
DSAgt observability: first-party span emission over the MLflow store.

DSAGT writes every trace to one store, ``sqlite:///<pdir>/mlflow.db`` by
default.  Two emission paths share that one store, because MLflow's API
separates them:

  * The live tracer (``mlflow.start_span``): first-party DSAGT spans emitted
    while the MCP server or dsagt-run runs.  Uses MLflow's active-span context
    for auto-nesting and the ``obs`` proxy.  Each trace's root is tagged
    ``dsagt.source`` with the MCP tool category the agent invoked
    (memory / skill / knowledge / registry), or ``execution`` for dsagt-run,
    set at the dispatch boundary, so the UI can filter this debugging view
    apart from agent traces and group it by concern.
  * The replay sink (``MLflowSink``, ``mlflow.start_span_no_context``): a
    finished agent ``traces.Trace`` written after the fact with the
    transcript's original timestamps.  ``start_span`` cannot backdate (it has
    no ``start_time_ns`` parameter), so replay uses ``start_span_no_context``;
    live uses ``start_span``, because ``start_span_no_context`` establishes no
    active span, and the ``obs`` proxy and auto-nesting depend on one.

Layout (top to bottom)
----------------------
  setup        find_project_config · resolve_tracking_uri · init_tracing
  live tracer  open_span ─┬─ traced       (decorate a function)
                          ├─ child_span   (open a sub-span)
                          └─ obs          (annotate the open span)
               tagging:   open_span(source=…) calls _attach_trace_metadata
                          (dsagt.source set on the trace's root only)
               factories: kb_* · registry_*
               log_execution_trace  (one code.execute trace from an
                            execution record, backdated to the run)
  replay sink  MLflowSink  (Trace to backdated spans; a traces.TraceCollector
                            consumer)

``traced`` and ``child_span`` open a span; ``obs`` annotates whichever span is
currently open.  All three no-op when tracing was never initialized, so
business code never imports MLflow and never branches on whether tracing is on.
"""

from __future__ import annotations

import functools
import getpass
import hashlib
import inspect
import logging
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

logger = logging.getLogger(__name__)

# Module-level state. _initialized guards every span helper: without a tracking
# store + experiment set (init_tracing), mlflow.start_span would write to
# MLflow's default location, so the helpers no-op until init_tracing succeeds.
# DSAGT runs one process per project, so the session id is a process-wide
# constant set once at startup and tagged onto each internal trace for grouping.
_initialized = False
_default_session_id: str | None = None
_default_agent: str | None = None


# ===========================================================================
# Setup: the project, the store, and the MLflow configuration
# ===========================================================================


def find_project_config() -> tuple[Path | None, dict | None]:
    """Read ``./.dsagt/config.yaml`` from cwd.

    Returns ``(cwd, parsed_config)``, or ``(None, None)`` when cwd is not a
    project directory.  Only cwd is checked: services that need this run
    with cwd equal to the project directory by contract, so any other cwd is
    a misconfigured caller and fails at once.

    Resolving a project name to its directory from an arbitrary cwd (the
    user CLI's ``dsagt info <name>``) is the registry's job
    (``~/dsagt-projects/projects.yaml``, ``session.project_dir``).  This
    helper is for services running inside the project.
    """
    cwd = Path.cwd().resolve()
    candidate = cwd / ".dsagt" / "config.yaml"
    if not candidate.exists():
        return None, None
    try:
        import yaml

        return cwd, yaml.safe_load(candidate.read_text()) or {}
    except Exception as e:
        logger.debug("could not parse %s: %s", candidate, e)
        return cwd, None


def resolve_tracking_uri(config: dict | None) -> str:
    """Compute the MLflow tracking URI for DSAGT self-logging.

    ``MLFLOW_TRACKING_URI`` in the environment wins when set: MLflow's own
    convention, and the one variable ``agents._mcp_env_block`` writes into
    every per-agent MCP config, so a value exported before ``dsagt init``
    reaches the CLI, the MCP server and its ``dsagt-run`` children alike.  That
    is how a project logs to a shared tracking server.  Credentials for such a
    server (``MLFLOW_TRACKING_TOKEN``, ``_USERNAME``, ``_PASSWORD``, or
    ``MLFLOW_TRACKING_API_KEY`` for an ``X-API-Key`` gateway, see
    :class:`ApiKeyHeaderProvider`) are read from the shell and are never
    written to disk.

    Otherwise ``sqlite:///<project_dir>/mlflow.db``, the serverless default.
    ``project_dir`` comes from the resolved config (injected by
    ``session.load_config``), falling back to cwd for in-project callers.  The
    MLflow client accepts a ``sqlite:`` URI directly (creating and migrating
    the database on first use), so self-logging needs no listener.  SQLite is
    MLflow's supported serverless backend; the filesystem store (``file:``,
    ``./mlruns``) is deprecated as of Feb 2026.  Spans and their metadata go
    to the sqlite store; MLflow keeps a span's large inputs and outputs, and
    the model record a trace is attached to, as files under the experiment's
    artifact location, ``<project>/mlruns/``.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    cfg = config or {}
    pdir = cfg.get("project_dir")
    base = Path(pdir).resolve() if pdir else Path.cwd().resolve()
    return f"sqlite:///{base / 'mlflow.db'}"


class ApiKeyHeaderProvider:
    """Send ``X-API-Key`` on every MLflow request.

    The key is ``MLFLOW_TRACKING_API_KEY``; the header is sent only when the
    variable is set.  A tracking server behind an API gateway (Kong answers ``WWW-Authenticate:
    Key``) authenticates on that header alone, and the MLflow client produces
    only the Bearer form of ``MLFLOW_TRACKING_TOKEN`` and Basic auth.
    Registered under the ``mlflow.request_header_provider`` entry point in
    ``pyproject.toml``, so MLflow loads it in every process of this
    environment (the CLI, the MCP server and its ``dsagt-run`` children)
    without an import on dsagt's side.  MLflow duck-types the provider
    (``in_context`` and ``request_headers``), so this class imports nothing
    from mlflow and the module's cold start is unchanged.  The key is read
    from the shell on each request and never written to disk.
    """

    def in_context(self) -> bool:
        return bool(os.environ.get("MLFLOW_TRACKING_API_KEY"))

    def request_headers(self) -> dict[str, str]:
        return {"X-API-Key": os.environ["MLFLOW_TRACKING_API_KEY"]}


EXPERIMENT_DESCRIPTION = "DSAgt (DataSmith Agent) AI-assisted data pipeline builder"


def experiment_name(config: dict | None) -> str:
    """The MLflow experiment this project logs to.

    ``mlflow.experiment`` in ``.dsagt/config.yaml`` when set; otherwise
    ``dsagt-<8 hex>`` from a hash of the project directory.  The project name
    is the wrong default on a shared tracking server: ``demo`` collides across
    users, and the experiment list there belongs to everyone.  The directory
    hash is stable for the life of the project (``dsagt mv`` changes it) and
    distinct per user because home directories are.  The readable project name
    is carried on the experiment's description and ``dsagt.project`` tag
    (:func:`_ensure_experiment`).
    """
    cfg = config or {}
    name = (cfg.get("mlflow") or {}).get("experiment")
    if name:
        return name
    pdir = cfg.get("project_dir")
    base = Path(pdir).resolve() if pdir else Path.cwd().resolve()
    return f"dsagt-{hashlib.sha1(str(base).encode()).hexdigest()[:8]}"


def _version_model_name() -> str:
    """The LoggedModel that stands for this dsagt release in an experiment.

    MLflow's trace table fills its *Version* column from ``mlflow.modelId``,
    which must reference a LoggedModel, MLflow 3's record of which version of
    the application produced a trace.  ``set_active_model(name=…)`` creates
    or reuses one per experiment and stamps every trace of the process,
    replayed agent turns included.  Model names may not contain ``.``.
    """
    from dsagt import __version__

    return f"dsagt-{__version__.replace('.', '_')}"


def _current_user() -> str | None:
    """The local user for the trace table's *User* column (``mlflow.trace.user``).

    A gateway may stamp its own identity under ``mlflow.user``; the UI reads
    only ``mlflow.trace.user``.  ``None`` where the process has no login
    identity."""
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return None


def _quiet_mlflow_chatter() -> None:
    """Silence MLflow's INFO messages about routine setup.

    ``set_experiment`` and ``set_active_model`` log "Experiment … does not
    exist. Creating", "LoggedModel … creating one" and "Active model is set to
    …" at INFO, the last on every ``dsagt-run``.  They go to stderr, and an
    agent that captures a code's stderr reads them as the code's output.
    Warnings and errors are still logged.
    """
    logging.getLogger("mlflow.tracking.fluent").setLevel(logging.WARNING)
    # "Flushing the async trace logging queue before program exit" at INFO on
    # every dsagt-run exit, and "Creating initial MLflow database tables" on
    # the first process to open a project's store; an agent that redirects a
    # code's stderr into its output file gets them appended to the JSON.
    for name in (
        "mlflow.tracing.export.async_export_queue",
        "mlflow.store.db.utils",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def _bound_remote_retries(tracking_uri: str) -> None:
    """Cap the MLflow client's retry budget against an http(s) store.

    The client defaults (7 retries at backoff 2, 120 s per request) mean a
    hung server stalls a single call for minutes, and the unattended paths
    (``dsagt-run`` cold start, the periodic pass) make several.  Tracing is
    best-effort; a few seconds is the most it may cost a tool call.
    ``setdefault``, so an explicit setting in the environment wins.
    """
    if tracking_uri.startswith(("http://", "https://")):
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "20")


def _ensure_experiment(name: str, project: str) -> None:
    """Select the experiment; on first creation, describe it and tag the project.

    ``set_experiment`` returns the experiment, so the tags cost nothing once
    they exist, and a description edited by hand on the server is left alone.
    """
    import mlflow

    existing = mlflow.get_experiment_by_name(name)
    if existing is not None and existing.lifecycle_stage == "deleted":
        # `set_experiment` refuses a name that exists in the deleted state,
        # and the name is deterministic for this project, so the project
        # cannot start again until the experiment is restored or renamed.
        # The error says which.
        raise RuntimeError(
            f"MLflow experiment {name!r} (id {existing.experiment_id}) exists "
            f"in the deleted state on {mlflow.get_tracking_uri()}. Restore it "
            "there, or set `mlflow.experiment` in .dsagt/config.yaml to a new "
            "name."
        )
    exp = mlflow.set_experiment(name)
    if "dsagt.project" not in exp.tags:
        mlflow.set_experiment_tag(
            "mlflow.note.content", f"{EXPERIMENT_DESCRIPTION}; project: {project}"
        )
        mlflow.set_experiment_tag("dsagt.project", project)


def init_tracing(
    service_name: str,
    mlflow_url: str | None = None,
    session_id: str | None = None,
) -> None:
    """Point MLflow at the project's store and experiment.

    After this, every ``mlflow.start_span`` (from ``@traced`` and
    ``child_span`` in the MCP server and dsagt-run) is written to the store
    :func:`resolve_tracking_uri` names, ``sqlite:///<pdir>/mlflow.db`` by
    default.  The session id, passed by the MCP server at startup, tags
    internal traces for grouping.

    The ``mlflow_url`` and ``session_id`` keyword arguments exist for tests,
    where the caller supplies known values directly.

    Never raises.  When cwd is not a dsagt project dir, or the store cannot be
    reached or the experiment used (deleted on a shared server, refused key,
    server down), logs the cause and no-ops so the process runs untraced:
    one-shot tools and tests outside a project, and a server whose tools must
    keep working regardless.
    """
    global _initialized, _default_session_id, _default_agent

    if _initialized:
        if session_id:
            _default_session_id = session_id
        return

    cfg_pdir, cfg = find_project_config()
    project_name = (cfg or {}).get("project")
    if not project_name:
        logger.warning(
            "%s: no .dsagt/config.yaml with a 'project' in cwd (%s); tracing "
            "disabled for this process.",
            service_name,
            Path.cwd(),
        )
        return

    _default_session_id = session_id
    _default_agent = cfg.get("agent")
    resolve_cfg = dict(cfg)
    resolve_cfg["project_dir"] = str(cfg_pdir)
    if mlflow_url is None:
        mlflow_url = resolve_tracking_uri(resolve_cfg)
    experiment = experiment_name(resolve_cfg)

    import mlflow

    try:
        _quiet_mlflow_chatter()
        _bound_remote_retries(mlflow_url)
        mlflow.set_tracking_uri(mlflow_url)
        _ensure_experiment(experiment, project_name)
        mlflow.set_active_model(name=_version_model_name())
    except (
        Exception
    ) as e:  # noqa: BLE001  # a store problem must not take the server down
        # The MCP tools serve regardless of tracing: a server that cannot
        # reach or use its store still serves the agent, untraced, with the
        # cause on the log so the operator can fix the store.
        logger.error(
            "%s: tracing disabled; cannot use experiment %r at %s: %s",
            service_name,
            experiment,
            mlflow_url,
            e,
        )
        return
    _initialized = True
    logger.info(
        "init_tracing: service=%s mlflow=%s experiment=%s project=%s session=%s",
        service_name,
        mlflow_url,
        experiment,
        project_name,
        _default_session_id or "<none>",
    )


# ===========================================================================
# Live tracer: first-party debug spans, emitted as code runs
# ===========================================================================

# ----- the span primitive -----


@contextmanager
def open_span(name: str, span_type: str | None = None, source: str | None = None):
    """Open a span on the MLflow store.

    The single tracing code path: ``mlflow.start_span`` nests under whatever
    span is already active (its own context model), so child spans nest
    without bookkeeping here.  Yields ``None`` when tracing was never
    initialized (one-shot tools and tests outside a project), so the helpers
    below become no-ops.

    ``source`` is set only on the categorization root of a trace (the MCP
    dispatch span and ``tool.execute``) and tags the whole trace
    ``dsagt.source`` for the debug-view filter
    (:func:`_attach_trace_metadata`).  Inner spans pass ``source=None`` and
    inherit the root's tag.
    """
    if not _initialized:
        yield None
        return
    import mlflow
    from mlflow.entities import SpanType

    with mlflow.start_span(name=name, span_type=span_type or SpanType.UNKNOWN) as span:
        _attach_trace_metadata(source)
        yield span


# ----- attribute-value helpers -----


def _coerce_attr(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce_attr(v) for v in value]
    return str(value)


def truncate(value: str, limit: int = 256) -> str:
    """Truncate a string for span attributes.

    Span backends (and the MLflow trace UI) handle short attribute values
    much better than multi-megabyte stdout/stderr blobs.  The full payload
    is in ``trace_archive/<record_id>.json``; the span carries a head and a
    character count so a reader of the UI can tell what happened.
    """
    if value is None:
        return ""
    if len(value) <= limit:
        return value
    head = limit - 32
    return value[:head] + f"... [+{len(value) - head} chars]"


REDACTED_KEYS = frozenset(
    {
        "headers",
        "authorization",
        "auth",
        "api_key",
        "api-key",
        "apikey",
        "x-api-key",
        "token",
        "access_token",
        "password",
        "secret",
    }
)
# Credential shapes inside free text (a recorded command carrying
# `-H "Authorization: Bearer …"`, a URL with `?api_key=…`, a printed config
# with `"api_key": "…"`), which a key-name match cannot catch.  Anchored so
# ordinary prose survives: `Bearer`/`Basic` only after `Authorization:`, and
# a key label only when its value looks like a token (16+ token characters),
# so "a basic example", "the bearer of bad news" and "max_token: 5" pass
# through untouched.  This masks the common shapes.
_SECRET_IN_TEXT = re.compile(
    r"(?i)"
    r"(authorization\s*[=:]\s*(?:bearer|basic)\s+)[^\s&\"']+"
    r"|((?:api[_-]?key|access[_-]?token|secret[_-]?key|password|token|secret)"
    r"\"?\s*[=:]\s*\"?)[A-Za-z0-9._\-]{16,}"
)


def _mask(m: "re.Match[str]") -> str:
    return (m.group(1) or m.group(2)) + "[redacted]"


def bound(value: Any, limit: int = 4096) -> Any:
    """Shrink a tool argument or result to what a span may safely carry.

    The dispatch shell records every call's raw arguments and result on the
    trace root, and both are agent-controlled: a ``kb_ingest`` argument or a
    ``kb_search`` result can carry an ``Authorization`` bearer or an API key
    from a document, and ``kb_search`` returns whole chunk texts.  Anything
    set on a span is written verbatim into ``mlflow.db`` (MLflow truncates
    only the UI preview), and ``dsagt traces`` then serves it in a browser.
    Credential-bearing keys are replaced outright, credential shapes inside
    strings are masked, and every string leaf is cut to ``limit`` so structure
    survives for the UI while the store holds a preview.
    """
    if isinstance(value, dict):
        return {
            k: "[redacted]" if str(k).lower() in REDACTED_KEYS else bound(v, limit)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [bound(v, limit) for v in value]
    if isinstance(value, str):
        return truncate(_SECRET_IN_TEXT.sub(_mask, value), limit)
    return value


# ----- trace tagging: the dsagt.source debug filter + session grouping -----
#
# dsagt.source names the MCP tool category that was invoked, one of
# {memory, skill, knowledge, registry} (the four concern modules of
# dsagt-server), plus ``execution`` for dsagt-run's out-of-process code runs.
# It is assigned at the entry point: the MCP dispatch shell maps each tool to
# the concern that owns it and stamps the category on the trace's root span,
# so ``search_skills`` calling into ``kb.search`` is tagged ``skill`` (the
# tool the agent called), and the ``knowledge`` subsystem that did the work
# leaves no tag of its own.  Inner spans inherit the root's tag; agent traces
# carry no ``dsagt.source`` at all, which is what lets the MLflow UI filter
# the debug view in or out.


def _attach_trace_metadata(source: str | None) -> None:
    """Tag the current trace as a DSAGT-internal (debug) trace.

    Called from :func:`open_span` on a categorization root (``source`` set):

    - ``dsagt.source`` tag: the MCP tool category or ``execution``; the UI's
      debug-view filter reads it.
    - ``dsagt.agent`` metadata: the agent platform that drove this session, so
      a shared store can be split by agent, the comparison the smoke test
      exists for.  Metadata, because that is where ``MLflowSink`` puts it on
      agent traces and where ``dsagt info`` reads it; one place.
    - ``mlflow.trace.session`` metadata: groups this process's internal traces
      under one session (reserved MLflow key, drives the native session filter).
    - ``mlflow.trace.user`` metadata: the local user, the reserved key behind
      the trace table's *User* column.
    - ``mlflow.modelId`` (set process-wide by :func:`init_tracing` through
      ``set_active_model``): the dsagt release, behind the *Version* column.
    - ``dsagt.version`` metadata: which dsagt produced the trace.  On a shared
      server holding months of traces from many installs, nothing else says;
      MLflow's own ``mlflow.source.git.*`` are empty because the process runs
      in the project directory, which is no checkout.  ``MLflowSink`` stamps
      the same key on agent traces.

    No-op for inner spans (``source is None``); they inherit the root's tag.
    """
    if not source:
        return
    from dsagt import __version__

    metadata = {"dsagt.version": __version__}
    if user := _current_user():
        metadata["mlflow.trace.user"] = user
    if _default_agent:
        metadata["dsagt.agent"] = _default_agent
    if _default_session_id:
        metadata["mlflow.trace.session"] = _default_session_id
    import mlflow

    mlflow.update_current_trace(tags={"dsagt.source": source}, metadata=metadata)


# ----- annotate the active span -----


class _ActiveSpanProxy:
    """Annotate the open span; ``traced`` and ``child_span`` open one.

    ``obs.set("hits", 5)`` from inside a ``@traced`` body attaches to that
    body's span.  It exists so business code (knowledge.py, provenance.py,
    registry_tools.py) never imports MLflow and never branches on whether
    tracing is on: when no span is active (tracing disabled, or a call site
    outside any traced block) every method does nothing.

    The process-wide singleton is exported as ``obs``.
    """

    def set(self, key: str, value: Any) -> None:
        span = self._current()
        if span is not None and value is not None:
            span.set_attribute(key, _coerce_attr(value))

    def set_many(self, attrs: Mapping[str, Any]) -> None:
        span = self._current()
        if span is None:
            return
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(k, _coerce_attr(v))

    def event(self, name: str, **attrs: Any) -> None:
        span = self._current()
        if span is None:
            return
        from mlflow.entities import SpanEvent

        clean_attrs = {k: v for k, v in attrs.items() if v is not None}
        span.add_event(SpanEvent(name, attributes=clean_attrs))

    def set_inputs(self, inputs: Any) -> None:
        """Populate the trace's ``request`` field for the MLflow trace UI."""
        span = self._current()
        if span is None or inputs is None:
            return
        span.set_inputs(inputs)

    def set_outputs(self, outputs: Any) -> None:
        """Populate the trace's ``response`` field for the MLflow trace UI."""
        span = self._current()
        if span is None or outputs is None:
            return
        span.set_outputs(outputs)

    def set_status(self, status: str) -> None:
        """Mark the span ``"ERROR"`` (or ``"OK"``), the trace state the UI
        filters on and ``dsagt info`` counts.  A failure a tool returns as a
        value (a ``{"status": "error"}`` result, a non-zero exit) is otherwise
        indistinguishable from success in the store."""
        span = self._current()
        if span is None:
            return
        span.set_status(status)

    @staticmethod
    def _current():
        """Return the active MLflow span, or ``None`` when no span is open."""
        if not _initialized:
            return None
        import mlflow

        return mlflow.get_current_active_span()


obs = _ActiveSpanProxy()


# ----- open a span: decorator + context manager -----


def traced(
    span_name: str,
    *,
    capture: Iterable[str] = (),
    extract_return: Mapping[str, Callable[[Any], Any]] | None = None,
) -> Callable:
    """Wrap a function in an MLflow span.

    Parameters
    ----------
    span_name
        Span name. Should be ``"<service>.<operation>"`` (e.g. ``"kb.search"``).
    capture
        Names of arguments to copy into span attributes. Looked up by name
        against the function signature, so positional and keyword args both
        work.
    extract_return
        Optional mapping of attribute name to a function applied to the return
        value to extract that attribute (``{"hits": lambda r: len(r)}``).

    Behavior
    --------
    * Captures the configured args as attributes (skipping ``None``).
    * Always sets ``duration_ms``.
    * Tags the trace ``dsagt.source`` + session for the debug view.
    * Records exceptions and sets ERROR status (MLflow auto-records the
      exception on context exit); re-raises after recording.
    """
    capture = tuple(capture)
    extract_return = dict(extract_return or {})

    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with open_span(span_name) as span:
                if span is None:
                    return fn(*args, **kwargs)

                # Capture configured arguments as span attributes.  Looked up
                # by name against the function signature so positional and
                # keyword args both work.  TypeError on bind_partial means a
                # decorator higher in the stack mangled the signature; in
                # that rare case we just skip arg capture rather than crash.
                if capture:
                    try:
                        bound = sig.bind_partial(*args, **kwargs)
                    except TypeError:
                        bound = None
                    if bound is not None:
                        bound.apply_defaults()
                        for name in capture:
                            if name in bound.arguments:
                                value = bound.arguments[name]
                                if value is not None:
                                    span.set_attribute(name, _coerce_attr(value))

                start = time.perf_counter()
                try:
                    result = fn(*args, **kwargs)
                except Exception:
                    span.set_status("ERROR")
                    raise
                finally:
                    span.set_attribute(
                        "duration_ms", round((time.perf_counter() - start) * 1000, 3)
                    )

                # Extract return-value-derived attributes.  A buggy extractor
                # lambda must NOT crash the instrumented function, but should
                # be visible at DEBUG so a developer running with --verbose
                # can spot a silently-missing span attribute caused by a
                # broken extract_return mapping.
                for attr_name, extractor in extract_return.items():
                    try:
                        value = extractor(result)
                    except Exception as e:
                        logger.debug(
                            "extract_return[%r] failed (%s: %s); "
                            "attribute will be missing from span",
                            attr_name,
                            type(e).__name__,
                            e,
                        )
                        continue
                    if value is not None:
                        span.set_attribute(attr_name, _coerce_attr(value))

                return result

        return wrapper

    return decorator


@contextmanager
def child_span(name: str, *, span_type: str | None = None, **attrs: Any):
    """Open a child span with arbitrary attributes.

    Use this from inside a ``@traced`` method to break a method into sub-phases
    (e.g. embed / index_search inside kb.search).  Prefer the typed
    factories below when one exists for your operation.
    """
    with open_span(name, span_type=span_type) as span:
        if span is None:
            yield None
            return
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(k, _coerce_attr(v))
        yield span


# ----- named span factories -----
#
# Every span name DSAgt emits has a factory here. Business modules call the
# factory, never mlflow.start_span directly, so span names, attribute schemas,
# and span types stay in one file.

# Knowledge base spans.


def kb_embed_span(backend: str | None, model: str | None, n_texts: int):
    """Span around an embedding call.

    Used for both query embedding (kb.search) and chunk embedding (kb.ingest,
    kb.append, kb.add_entries).  Backend-agnostic: ``backend`` is ``"api"``
    for the HTTP embedder or ``"local"`` for the ONNX model.
    """
    from mlflow.entities import SpanType

    return child_span(
        "kb.embed",
        span_type=SpanType.EMBEDDING,
        backend=backend,
        model=model,
        n_texts=n_texts,
    )


def kb_index_search_span(vector_db: str | None, k: int, filtered: bool):
    """Span around an underlying vector index search call."""
    from mlflow.entities import SpanType

    return child_span(
        "kb.index_search",
        span_type=SpanType.RETRIEVER,
        vector_db=vector_db,
        k=k,
        filtered=filtered,
    )


# Registry spans.  Only the deliberate, infrequent registry operations are
# instrumented: search_registry and search_skills are high-frequency and
# low-information per call, and the agent-side LLM trace already records that
# they were invoked.


def registry_save_code_span(code_name: str | None):
    """Span around ``save_code_spec``."""
    from mlflow.entities import SpanType

    return child_span(
        "registry.save_code_spec", span_type=SpanType.TOOL, code_name=code_name
    )


def registry_install_deps_span(packages: list[str] | None):
    """Span around an ``install_dependencies`` call."""
    from mlflow.entities import SpanType

    return child_span(
        "registry.install_dependencies",
        span_type=SpanType.TOOL,
        package_count=len(packages) if packages else 0,
        # The first few package names identify the call in the UI; the full
        # list is in the LLM call record.
        packages_preview=", ".join(packages[:5]) if packages else "",
    )


def registry_reconstruct_pipeline_span(fmt: str | None):
    """Span around a ``reconstruct_pipeline`` call."""
    from mlflow.entities import SpanType

    return child_span(
        "registry.reconstruct_pipeline", span_type=SpanType.TOOL, format=fmt or "bash"
    )


# Code execution spans (dsagt-run).


def log_execution_trace(record: dict) -> str | None:
    """Log one ``dsagt-run`` execution record as a ``code.execute`` trace.

    The span is backdated to the record's start and end, so the trace can be
    written after the run, from another process, without the run paying for
    the store.  A top-level, categorization-root span: the agent's shell
    spawns ``dsagt-run`` in its own process tree, so this trace stands alone
    in the store.  It carries ``record_id`` (correlates to the
    ``trace_archive`` record) and ``code_name``, and is tagged
    ``dsagt.source=execution``, a bucket distinct from the four MCP tool
    categories because these are code runs in the user's environment.  The
    span holds previews: the command, and stdout and stderr cut to about
    4 KB, since the full output is in the record on the machine that ran the
    code, and that file is the only full copy when the store is a shared
    server other people read.  Returns the trace id, or ``None`` when tracing
    is not initialized.
    """
    if not _initialized:
        return None
    import mlflow
    from mlflow.entities import SpanEvent, SpanStatusCode, SpanType
    from mlflow.tracing.trace_manager import InMemoryTraceManager

    from dsagt import __version__

    execution = record["execution"]
    command = execution["exact_command"]
    stdout = execution.get("stdout", "")
    stderr = execution.get("stderr", "")
    return_code = execution["return_code"]
    start_ns = _iso_to_ns(execution["timestamp_start"])
    end_ns = _iso_to_ns(execution["timestamp_end"])
    duration_ms = execution.get("duration_ms", round((end_ns - start_ns) / 1e6, 3))

    attributes = {
        "record_id": record["record_id"],
        "code_name": record["code_name"],
        "exit_code": return_code,
        "duration_ms": duration_ms,
        "n_input_files": len(execution.get("input_files", [])),
        "n_output_files": len(execution.get("output_files", [])),
        "command": truncate(" ".join(command), 256),
        "stdout_len": len(stdout),
        "stderr_len": len(stderr),
    }
    if stderr.strip():
        attributes["stderr_truncated"] = truncate(stderr, 256)
    span = mlflow.start_span_no_context(
        name="code.execute",
        span_type=SpanType.TOOL,
        start_time_ns=start_ns,
        inputs=bound(
            {
                "code": record["code_name"],
                "command": list(command),
                "input_files": execution.get("input_files", []),
            }
        ),
        attributes=attributes,
    )
    span.set_outputs(
        bound(
            {
                "exit_code": return_code,
                "duration_ms": duration_ms,
                "stdout": truncate(stdout, 4096),
                "stderr": truncate(stderr, 4096) if stderr else "",
                "output_files": execution.get("output_files", []),
            }
        )
    )
    if return_code != 0:
        span.add_event(
            SpanEvent(
                "code_failed", timestamp=end_ns, attributes={"exit_code": return_code}
            )
        )
        span.set_status(SpanStatusCode.ERROR)

    metadata = {"dsagt.version": __version__}
    if user := _current_user():
        metadata["mlflow.trace.user"] = user
    if _default_agent:
        metadata["dsagt.agent"] = _default_agent
    session = record.get("session_id") or _default_session_id
    if session:
        metadata["mlflow.trace.session"] = session
    with InMemoryTraceManager.get_instance().get_trace(span.trace_id) as in_mem:
        in_mem.info.trace_metadata = {**in_mem.info.trace_metadata, **metadata}
        in_mem.info.tags = {**in_mem.info.tags, "dsagt.source": "execution"}
    span.end(end_time_ns=end_ns)
    return span.trace_id


def _iso_to_ns(timestamp: str) -> int:
    from datetime import datetime

    return int(datetime.fromisoformat(timestamp).timestamp() * 1e9)


# ===========================================================================
# Replay sink: a finished agent Trace to backdated MLflow spans
# ===========================================================================

_S_PER_NS = 1e9


def _to_ns(epoch_s: float | None) -> int | None:
    return int(epoch_s * _S_PER_NS) if epoch_s is not None else None


def _stamp_usage(span, usage: dict | None) -> None:
    """Set MLflow's chat-usage attribute from a normalized usage dict.

    MLflow sums this attribute across every span of a trace into
    ``mlflow.trace.tokenUsage``, so it is set on whichever span represents
    the LLM call: an LLM span, or the tool span that a tool-calling message
    collapses into under the autolog-parity layout.  ``input_tokens`` already
    counts cached tokens (``traces._usage``); the cache breakdown is kept as
    plain attributes for the per-span view.
    """
    if not usage:
        return
    from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey

    inp = usage.get("input_tokens") or 0
    out = usage.get("output_tokens") or 0
    span.set_attribute(
        SpanAttributeKey.CHAT_USAGE,
        {
            TokenUsageKey.INPUT_TOKENS: inp,
            TokenUsageKey.OUTPUT_TOKENS: out,
            TokenUsageKey.TOTAL_TOKENS: inp + out,
        },
    )
    for key in ("cache_read_input_tokens", "cache_write_input_tokens"):
        if usage.get(key):
            span.set_attribute(key, usage[key])


class MLflowSink:
    """Render a :class:`~dsagt.traces.Trace` into MLflow spans (a trace consumer).

    The agent half of observability: ``@traced`` and ``obs`` emit DSAGT's own
    debug spans live, and this replays a finished transcript's
    :class:`~dsagt.traces.Trace` after the fact into the same store.  It uses
    ``mlflow.start_span_no_context``, the one API that accepts an explicit
    ``parent_span`` and a backdated ``start_time_ns``, and mirrors the span
    conventions of MLflow's own ``claude_code`` autolog so foreign traces
    render identically in the Chat UI: an AGENT root, ``llm`` children
    carrying ``message.format="anthropic"`` and ``mlflow.chat.tokenUsage``,
    and ``tool_<name>`` children.  Agent traces carry no ``dsagt.source`` tag,
    so they stay in the normal view, separate from the internal debug traces.

    A session ``Trace`` carries one AGENT subtree per turn; the sink emits one
    MLflow trace per AGENT root, matching the per-prompt granularity autolog's
    Stop hook produces.  MLflow mints its own trace and span ids, so each
    trace is tagged ``dsagt.trace_id = <trace_id>:<root span_id>`` (a stable
    per-turn idempotency key).

    A consumer of :class:`~dsagt.traces.TraceCollector`: ``name`` keys its own
    ack file (``.dsagt/trace_acks_mlflow.json``); ``write`` logs the trace.
    Spans are plain dicts (the ``traces`` module docstring describes them), so
    this reads them directly.
    """

    name = "mlflow"

    def __init__(self, tracking_uri: str, experiment: str):
        self._uri = tracking_uri
        self._experiment = experiment

    def write(self, trace) -> list[str]:
        """Log every turn subtree; return the MLflow trace id of each."""
        import mlflow

        _quiet_mlflow_chatter()
        _bound_remote_retries(self._uri)
        mlflow.set_tracking_uri(self._uri)
        mlflow.set_experiment(self._experiment)
        # The CLI catch-up path (`dsagt traces` / `dsagt info`) reaches here
        # without init_tracing, so the version model is activated here as well;
        # create-or-reuse, one call per write.
        mlflow.set_active_model(name=_version_model_name())

        children: dict[str, list] = {}
        for span in trace.spans:
            if span["parent_id"] is not None:
                children.setdefault(span["parent_id"], []).append(span)

        trace_ids = []
        for root in trace.spans:
            if root["parent_id"] is None:
                trace_ids.append(
                    self._emit_subtree(root, children.get(root["span_id"], []), trace)
                )
        return trace_ids

    def _emit_subtree(self, root, children, trace) -> str:
        """Emit one MLflow trace for an AGENT ``root`` and its direct children."""
        import mlflow
        from mlflow.entities import SpanType
        from mlflow.tracing.constant import (
            SpanAttributeKey,
            TraceMetadataKey,
        )
        from mlflow.tracing.trace_manager import InMemoryTraceManager

        kind_to_type = {
            "AGENT": SpanType.AGENT,
            "LLM": SpanType.LLM,
            "TOOL": SpanType.TOOL,
            "OTHER": SpanType.UNKNOWN,
        }

        ml_root = mlflow.start_span_no_context(
            name=root["name"],
            span_type=kind_to_type[root["kind"]],
            inputs={"prompt": root["attributes"].get("prompt", "")},
            start_time_ns=_to_ns(root["start_time"]),
        )

        for span in children:
            if span["kind"] == "LLM":
                child = mlflow.start_span_no_context(
                    name=span["name"],
                    parent_span=ml_root,
                    span_type=SpanType.LLM,
                    start_time_ns=_to_ns(span["start_time"]),
                    inputs={
                        "model": span["model"] or "unknown",
                        "messages": span["request"],
                    },
                    attributes={
                        "model": span["model"] or "unknown",
                        SpanAttributeKey.MESSAGE_FORMAT: "anthropic",
                    },
                )
                _stamp_usage(child, span["usage"])
                child.set_outputs(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": span["response"],
                    }
                )
            else:  # TOOL / OTHER
                child = mlflow.start_span_no_context(
                    name=span["name"],
                    parent_span=ml_root,
                    span_type=kind_to_type[span["kind"]],
                    start_time_ns=_to_ns(span["start_time"]),
                    inputs=span["attributes"].get("input", {}),
                    attributes={
                        "tool_name": span["attributes"].get("tool_name"),
                        "tool_id": span["attributes"].get("tool_id"),
                    },
                )
                child.set_outputs({"result": span["attributes"].get("result", "")})
                _stamp_usage(child, span["usage"])
            child.end(end_time_ns=_to_ns(span["end_time"]))

        # Trace-level metadata: session correlation + the per-turn canonical id
        # (idempotency key) + request/response previews for the trace list.
        try:
            mgr = InMemoryTraceManager.get_instance()
            with mgr.get_trace(ml_root.trace_id) as in_mem:
                from dsagt import __version__

                meta = {
                    TraceMetadataKey.TRACE_SESSION: trace.session_id,
                    "dsagt.trace_id": f"{trace.trace_id}:{root['span_id']}",
                    "dsagt.agent": trace.agent,
                    "dsagt.version": __version__,
                }
                if user := _current_user():
                    meta[TraceMetadataKey.TRACE_USER] = user
                in_mem.info.trace_metadata = {**in_mem.info.trace_metadata, **meta}
                if prompt := root["attributes"].get("prompt"):
                    in_mem.info.request_preview = str(prompt)[:1000]
                if response := root["attributes"].get("response"):
                    in_mem.info.response_preview = str(response)[:1000]
        except Exception as e:  # noqa: BLE001
            logger.warning("MLflowSink: could not stamp trace metadata: %s", e)

        ml_root.set_outputs({"response": root["attributes"].get("response", "")})
        ml_root.end(end_time_ns=_to_ns(root["end_time"]))
        return ml_root.trace_id
