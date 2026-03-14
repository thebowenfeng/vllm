#!/bin/bash
#
# Launch a vLLM multi-node cluster using the native vLLM installation (no Docker).
#
# Supports two backends:
#   - ray:                    Ray manages the cluster. Start head on one machine,
#                             workers on others, then run `vllm serve` once from
#                             the head node.
#   - mp (or multiprocessing): No Ray required. Run this script on every node
#                             with the appropriate --node-rank. vLLM coordinates
#                             directly via torch.distributed (NCCL/Gloo over TCP).
#
# ─── RAY MODE ────────────────────────────────────────────────────────────────
# 1. On the head node:
#    bash run_cluster_native.sh \
#         --head \
#         --head-node-ip <HEAD_NODE_IP> \
#         --backend ray
#
# 2. On each worker node:
#    bash run_cluster_native.sh \
#         --worker \
#         --head-node-ip <HEAD_NODE_IP> \
#         --node-ip <THIS_NODE_IP> \
#         --backend ray
#
# 3. Then, from the head node, start vLLM (once the cluster is up):
#    vllm serve <model> \
#         --tensor-parallel-size <gpus_per_node> \
#         --pipeline-parallel-size <num_nodes> \
#         --distributed-executor-backend ray \
#         --max-num-seqs 2
#
# ─── MULTIPROCESSING MODE ────────────────────────────────────────────────────
# 1. On the head node (node-rank 0):
#    bash run_cluster_native.sh \
#         --head \
#         --head-node-ip <HEAD_NODE_IP> \
#         --backend multiprocessing \
#         --nnodes <total_nodes> \
#         --tensor-parallel-size <gpus_per_node> \
#         --pipeline-parallel-size <num_nodes> \
#         --model <model_name_or_path> \
#         [extra vllm serve args...]
#
# 2. On each worker node (node-rank 1, 2, ...):
#    bash run_cluster_native.sh \
#         --worker \
#         --head-node-ip <HEAD_NODE_IP> \
#         --node-ip <THIS_NODE_IP> \
#         --node-rank <RANK> \
#         --backend multiprocessing \
#         --nnodes <total_nodes> \
#         --tensor-parallel-size <gpus_per_node> \
#         --pipeline-parallel-size <num_nodes> \
#         --model <model_name_or_path> \
#         [extra vllm serve args...]
#
# ─── SAME-MACHINE MULTIPLE WORKERS ──────────────────────────────────────────
# To run multiple workers on the same machine (e.g. for testing), supply
# different --node-ip values that resolve to different IPs on the machine
# (e.g. different loopback aliases or bound interface IPs). For Ray mode,
# you can also run workers in separate terminals on the same host — Ray will
# register them as separate processes on the same node, but vLLM only needs
# one Ray node per machine for typical use. For true multi-worker-per-host
# simulation, use multiprocessing mode with --nnodes matching the number of
# workers you want.
#
# ─── NOTES ───────────────────────────────────────────────────────────────────
# - All nodes must have vLLM and Ray (if using Ray backend) installed.
# - All nodes must be reachable at the IP addresses you supply.
# - RAY_PORT (default 6379) and VLLM_MASTER_PORT (default 29500) must be open.
# - Traffic is unencrypted — use a private network segment.
# - Keep terminal sessions open; closing them stops the node.

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
NODE_TYPE=""
HEAD_NODE_IP=""
NODE_IP=""
NODE_RANK=""
BACKEND="ray"
NNODES=""
RAY_PORT=6379
VLLM_MASTER_PORT=29500
# Max concurrent sequences — set to 2+ so the scheduler can pipeline two
# simultaneous requests instead of serialising them behind each other.
MAX_NUM_SEQS=2
# Remaining args after known flags are forwarded verbatim to `vllm serve`
VLLM_EXTRA_ARGS=()

# ─── Argument parsing ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --head)
            NODE_TYPE="head"
            shift
            ;;
        --worker)
            NODE_TYPE="worker"
            shift
            ;;
        --head-node-ip)
            HEAD_NODE_IP="$2"
            shift 2
            ;;
        --node-ip)
            NODE_IP="$2"
            shift 2
            ;;
        --node-rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --ray-port)
            RAY_PORT="$2"
            shift 2
            ;;
        --master-port)
            VLLM_MASTER_PORT="$2"
            shift 2
            ;;
        --max-num-seqs)
            MAX_NUM_SEQS="$2"
            shift 2
            ;;
        *)
            # Everything else is forwarded to `vllm serve`
            VLLM_EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ─── Validation ──────────────────────────────────────────────────────────────
if [[ -z "${NODE_TYPE}" ]]; then
    echo "Error: must specify --head or --worker"
    exit 1
fi

if [[ -z "${HEAD_NODE_IP}" ]]; then
    echo "Error: --head-node-ip is required"
    exit 1
fi

if [[ "${BACKEND}" != "ray" && "${BACKEND}" != "multiprocessing" && "${BACKEND}" != "mp" ]]; then
    echo "Error: --backend must be 'ray' or 'multiprocessing' (or 'mp')"
    exit 1
fi

# Normalise "multiprocessing" → "mp" (vLLM only accepts the short form)
if [[ "${BACKEND}" == "multiprocessing" ]]; then
    BACKEND="mp"
fi

if [[ "${NODE_TYPE}" == "worker" && -z "${NODE_IP}" ]]; then
    echo "Error: --node-ip is required for worker nodes (must be unique per worker)"
    exit 1
fi

if [[ "${BACKEND}" == "multiprocessing" && "${NODE_TYPE}" == "worker" && -z "${NODE_RANK}" ]]; then
    echo "Error: --node-rank is required for worker nodes in multiprocessing mode"
    exit 1
fi

if [[ "${BACKEND}" == "multiprocessing" && -z "${NNODES}" ]]; then
    echo "Error: --nnodes is required in multiprocessing mode"
    exit 1
fi

# Warn when pipeline parallelism is used on a single node.
# With PP>1 on one node, concurrent requests must advance through all pipeline
# stages in lock-step — one request stalls the other mid-generation.
# For single-node multi-GPU setups, prefer --tensor-parallel-size=<num_gpus>
# (pure TP) which has no pipeline bubbles and supports true concurrent decoding.
#
# Technical note: vLLM's --async-scheduling flag (which reduces PP bubbles)
# is silently ignored when pipeline_parallel_size > 1 — the executor code
# gates it on `pp_size <= 1` — so there is no workaround within vLLM today.
if [[ "${NNODES}" == "1" ]]; then
    # Extract --pipeline-parallel-size from VLLM_EXTRA_ARGS if set
    PP_SIZE=1
    for i in "${!VLLM_EXTRA_ARGS[@]}"; do
        if [[ "${VLLM_EXTRA_ARGS[$i]}" == "--pipeline-parallel-size" || \
              "${VLLM_EXTRA_ARGS[$i]}" == "-pp" ]]; then
            PP_SIZE="${VLLM_EXTRA_ARGS[$((i+1))]}"
        fi
    done
    if [[ "${PP_SIZE}" -gt 1 ]]; then
        echo ""
        echo "  WARNING: --pipeline-parallel-size=${PP_SIZE} with --nnodes=1 detected."
        echo "  Pipeline parallelism on a single node causes concurrent requests to"
        echo "  advance in lock-step through pipeline stages, making them appear stuck."
        echo "  Recommendation: use --tensor-parallel-size=<num_gpus> instead and"
        echo "  omit --pipeline-parallel-size (or set it to 1)."
        echo ""
    fi
fi

# ─── Derived values ──────────────────────────────────────────────────────────
# Head node IP is this node's IP when running --head
if [[ "${NODE_TYPE}" == "head" && -z "${NODE_IP}" ]]; then
    NODE_IP="${HEAD_NODE_IP}"
fi

export VLLM_HOST_IP="${NODE_IP}"

echo "============================================================"
echo "  vLLM Native Cluster Launcher"
echo "  Node type       : ${NODE_TYPE}"
echo "  Backend         : ${BACKEND}"
echo "  Head node IP    : ${HEAD_NODE_IP}"
echo "  This node IP    : ${NODE_IP}"
[[ -n "${NODE_RANK}" ]] && echo "  Node rank       : ${NODE_RANK}"
[[ -n "${NNODES}" ]]    && echo "  Total nodes     : ${NNODES}"
echo "  Max num seqs    : ${MAX_NUM_SEQS}"
echo "============================================================"

# Build the scheduling-related flags to inject into vllm serve
# NOTE: --async-scheduling is intentionally omitted here.
# vLLM silently ignores it when pipeline_parallel_size > 1 (the condition
# `pp_size <= 1 && async_scheduling` in multiproc_executor.py means async
# scheduling only applies to non-pipeline-parallel setups). Passing it with
# PP>1 is misleading and has no effect.
SCHEDULING_ARGS=("--max-num-seqs" "${MAX_NUM_SEQS}")

# ─── RAY BACKEND ─────────────────────────────────────────────────────────────
if [[ "${BACKEND}" == "ray" ]]; then

    # Check Ray is available
    if ! command -v ray &>/dev/null; then
        echo "Error: 'ray' not found on PATH. Install with: pip install 'ray[cgraph]'"
        exit 1
    fi

    if [[ "${NODE_TYPE}" == "head" ]]; then
        echo "Starting Ray HEAD node on ${NODE_IP}:${RAY_PORT} ..."
        echo ""
        echo "  Once all workers have joined, start vLLM from this node with:"
        echo "    vllm serve <model> \\"
        echo "      --tensor-parallel-size <gpus_per_node> \\"
        echo "      --pipeline-parallel-size <num_nodes> \\"
        echo "      --distributed-executor-backend ray \\"
        echo "      --max-num-seqs ${MAX_NUM_SEQS}"
        echo ""
        ray start \
            --head \
            --node-ip-address="${NODE_IP}" \
            --port="${RAY_PORT}" \
            --block
    else
        echo "Starting Ray WORKER node, connecting to ${HEAD_NODE_IP}:${RAY_PORT} ..."
        ray start \
            --address="${HEAD_NODE_IP}:${RAY_PORT}" \
            --node-ip-address="${NODE_IP}" \
            --block
    fi

# ─── MULTIPROCESSING BACKEND ─────────────────────────────────────────────────
elif [[ "${BACKEND}" == "mp" ]]; then

    # Check vllm is available
    if ! command -v vllm &>/dev/null; then
        echo "Error: 'vllm' not found on PATH. Ensure vLLM is installed."
        exit 1
    fi

    if [[ "${NODE_TYPE}" == "head" ]]; then
        echo "Starting vLLM HEAD node (node-rank 0) ..."
        echo "Worker nodes should connect to ${HEAD_NODE_IP}:${VLLM_MASTER_PORT}"
        echo ""
        # Head node: node-rank 0, no --headless flag (it drives serving)
        vllm serve \
            --distributed-executor-backend mp \
            --nnodes "${NNODES}" \
            --node-rank 0 \
            --master-addr "${HEAD_NODE_IP}" \
            --master-port "${VLLM_MASTER_PORT}" \
            "${SCHEDULING_ARGS[@]}" \
            "${VLLM_EXTRA_ARGS[@]}"
    else
        echo "Starting vLLM WORKER node (node-rank ${NODE_RANK}) ..."
        echo "Connecting to head at ${HEAD_NODE_IP}:${VLLM_MASTER_PORT} ..."
        echo ""
        # Worker nodes: supply their rank and --headless (no HTTP server)
        vllm serve \
            --distributed-executor-backend mp \
            --nnodes "${NNODES}" \
            --node-rank "${NODE_RANK}" \
            --master-addr "${HEAD_NODE_IP}" \
            --master-port "${VLLM_MASTER_PORT}" \
            --headless \
            "${SCHEDULING_ARGS[@]}" \
            "${VLLM_EXTRA_ARGS[@]}"
    fi
fi
