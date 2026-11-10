# shared settings for the complete_port tools (source me)
CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${SB_IMAGE:-pufferai/puffertank:5.0}"
ART="$CP/artifacts"                       # generated: blobs, traces, references, binaries
SK="$(cd "$CP/../starter-kit" && pwd)"    # the unmodified simulator (the oracle)
mkdir -p "$ART/bin" "$ART/blobs" "$ART/traces" "$ART/refs"
# the six shipped building shapes (one scenario each) used by the gates
SHAPES="building_train_1/fairbanks_1 building_train_2/atlanta_3 building_train_4/atlanta_5 building_train_5/atlanta_6 building_train_7/atlanta_8 building_train_8/atlanta_1"
# GPU selection (P2): SB_GPU=<index> pins every tool to one device instead of all; tools/with_gpu.sh
# additionally takes the per-device flock, so two jobs never share a device by accident.
if [ -n "${SB_GPU:-}" ] && [ "${SB_GPU}" != all ]; then DEVS=(--gpus "device=$SB_GPU"); else DEVS=(--gpus all); fi
# run a command in the CUDA 13 container with complete_port mounted at /cp
dock() { docker run --rm "${DEVS[@]}" -v "$CP":/cp -w /cp "$IMAGE" bash -lc "$*"; }
