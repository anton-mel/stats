from pie_evals.node.engines.shape import context_tokens_for, serve_envelope, workload_concurrency
from pie_evals.schema import WorkloadSpec


def w(id, kind, **params):
    return WorkloadSpec(id=id, kind=kind, params=params)


def test_envelope_takes_the_longest_context_and_the_highest_concurrency():
    cells = [
        w("ss", "single_stream", prefill=128, decode=64),
        w("c8", "concurrency", concurrency=8, num_requests=32, prefill=128, decode=128),
        w("lc", "long_context", prefill=2048, decode=128),
    ]
    env = serve_envelope(cells)
    assert workload_concurrency(env) == 8
    assert context_tokens_for(env) == 2048 + 128


def test_envelope_is_the_widest_workload_when_it_already_covers_the_rest():
    cells = [w("ss", "single_stream", prefill=128, decode=64), w("c256", "concurrency", concurrency=256, num_requests=1024, prefill=128, decode=128)]
    assert serve_envelope(cells).id == "c256"


def test_envelope_keeps_a_long_context_shape_so_its_prompt_stays_a_shared_prefix():
    from pie_evals.node.workloads.synthetic import common_args_for

    cells = [w("c8", "concurrency", concurrency=8, num_requests=32, prefill=128, decode=128), w("lc", "long_context", prefill=8192, decode=128)]
    env = serve_envelope(cells)
    assert str(env.kind) == "long_context" and workload_concurrency(env) == 8
    args = common_args_for(env)
    assert max(len(a) for a in args) < 64 * 1024
