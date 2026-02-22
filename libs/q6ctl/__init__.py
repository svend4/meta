"""q6ctl — Оркестратор Q6: реестр, контекст, пайплайны.

Использование:
  python -m libs.q6ctl.q6cli list modules
  python -m libs.q6ctl.q6cli run SC-1
  python -m libs.q6ctl.q6cli pipe hexpack:ring hexcrypt:sbox
  python -m libs.q6ctl.q6cli ctx new my_session
"""
from .registry import (
    MODULES, CLUSTERS, SUPERCLUSTERS,
    get_module, get_cluster, get_supercluster,
    modules_in_cluster, cluster_of_module,
    all_module_names, all_cluster_ids, all_supercluster_ids,
    json_ready_modules,
)
from .context import (
    create as ctx_create,
    load as ctx_load,
    save as ctx_save,
    write_key, read_key, has_key, keys as ctx_keys,
    list_contexts,
)
from .pipeline import (
    PipelineStep, parse_steps,
    run as pipeline_run,
    run_step,
    dry_run,
    run_supercluster,
    dry_run_supercluster,
    available_pipelines,
)
