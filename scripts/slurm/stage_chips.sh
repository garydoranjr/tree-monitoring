# Sourced by the planet_temporal_*.sh jobs. stage_chips SRC DSTROOT copies the
# chip set SRC to DSTROOT/<name> on node-local disk and prints that path. A
# per-node lock keeps array tasks sharing a node from copying at once, and the
# copy is verified by checksum against SRC before use.

stage_chips() {
    local src="${1%/}" dstroot="$2"
    local dst="$dstroot/$(basename "$src")"
    mkdir -p "$dstroot"
    (
        flock 9
        rsync -a "$src"/ "$dst"/
        local diffs
        diffs=$(rsync -rc --dry-run --out-format='%n' "$src"/ "$dst"/ | grep -v '/$' || true)
        if [ -n "$diffs" ]; then
            echo "Checksum mismatch staging $src -> $dst:" >&2
            echo "$diffs" >&2
            exit 1
        fi
    ) 9>"$dstroot/.stage.lock" >&2
    echo "$dst"
}
