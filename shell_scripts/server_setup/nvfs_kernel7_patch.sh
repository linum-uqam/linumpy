#!/bin/bash
# Patch nvidia-fs 2.28.4 to build against Linux kernel 7.0 (Ubuntu 26.04).
#
# REMOVAL TRIGGER
# ---------------
# Delete this script as soon as nvidia-fs-dkms ships an upstream version
# that builds cleanly on the running kernel. Quick check:
#
#   sudo dkms add /usr/src/nvidia-fs-<NEW_VER>      # try the stock sources
#   sudo dkms build nvidia-fs/<NEW_VER>             # if this succeeds, drop the patch
#
# Two upstream API changes this patch works around:
#   1) struct vm_area_struct: __vm_flags removed, vm_flags is now a const
#      field of an anonymous union (writes still go through vm_flags_set).
#      Fix: read via vma->vm_flags.
#   2) struct blk_dma_iter::iter is now struct blk_map_iter (was
#      struct req_iterator). They share .iter (bvec_iter) and .bio in the
#      same order, so we add a configure probe HAVE_BLK_MAP_ITER and
#      typedef nvfs_req_iter_t accordingly. Helper signatures use the typedef.
#   3) create_nv.symvers.sh only decompressed *.ko.xz; Ubuntu 26.04 ships
#      kernel modules as *.ko.zst, so nm fails and Module.symvers ends up
#      empty. Without CRCs the runtime symbol-version check on
#      nvidia_p2p_dma_unmap_pages fails and nvidia-fs falls back to the
#      no-symver path -> GDS stays in compat mode. Fix: add a *.ko.zst
#      branch that decompresses with zstd before nm runs.
#
# USAGE (server-side, as root)
# ----------------------------
#   sudo bash nvfs_kernel7_patch.sh
#   sudo dkms remove nvidia-fs/2.28.4 --all 2>/dev/null || true
#   sudo dkms add /usr/src/nvidia-fs-2.28.4
#   sudo dkms build nvidia-fs/2.28.4
#   sudo dkms install nvidia-fs/2.28.4
#   sudo modprobe nvidia-fs
#   /usr/local/cuda/gds/tools/gdscheck -p
#
# The script is idempotent (creates *.orig backups, skips reapplying if the
# probe is already present). Safe to re-run after kernel header upgrades.

set -e
SRC=${NVFS_SRC:-/usr/src/nvidia-fs-2.28.4}
if [ ! -d "$SRC" ]; then
    echo "ERROR: $SRC not found. Set NVFS_SRC=/usr/src/nvidia-fs-<ver> if version differs." >&2
    exit 2
fi
cd "$SRC"

# Idempotency: keep one .orig backup
for f in nvfs-mmap.c nvfs-dma.c configure create_nv.symvers.sh; do
    [ -f "$f.orig" ] || cp -p "$f" "$f.orig"
done

# --- nvfs-mmap.c: drop ACCESS_PRIVATE(__vm_flags) --------------------------
# kernel 7.0 dropped the __vm_flags private name; vm_flags is the public
# (const) field name again.
python3 - <<'PY'
from pathlib import Path
p = Path("nvfs-mmap.c")
s = p.read_text()
old = "\tvm_flags = ACCESS_PRIVATE(vma, __vm_flags);"
new = "\tvm_flags = vma->vm_flags;"
assert old in s, "expected ACCESS_PRIVATE line not found"
p.write_text(s.replace(old, new, 1))
print("nvfs-mmap.c: patched vm_flags read")
PY

# --- configure: add HAVE_BLK_MAP_ITER probe --------------------------------
python3 - <<'PY'
from pathlib import Path
p = Path("configure")
s = p.read_text()
marker = 'if compile_prog "Checking if blk_rq_dma_map_iter_start is present..."; then\n        output_sym "HAVE_BLK_RQ_DMA_MAP_ITER_START"\nfi\n'
addition = '''
cat > $TEST_C <<EOF
#include <linux/blkdev.h>
#include <linux/blk-mq.h>
#include <linux/blk-mq-dma.h>
#include "test.h"

int test (void)
{
        struct blk_map_iter iter;
        (void)iter;
        return 0;
}

EOF
if compile_prog "Checking if struct blk_map_iter is present..."; then
        output_sym "HAVE_BLK_MAP_ITER"
fi
'''
if "HAVE_BLK_MAP_ITER" in s:
    print("configure: HAVE_BLK_MAP_ITER probe already present")
else:
    assert marker in s, "anchor not found in configure"
    s = s.replace(marker, marker + addition, 1)
    p.write_text(s)
    print("configure: added HAVE_BLK_MAP_ITER probe")
PY

# --- nvfs-dma.c: typedef nvfs_req_iter_t and rewire V2 path ----------------
python3 - <<'PY'
from pathlib import Path
p = Path("nvfs-dma.c")
s = p.read_text()

# Insert typedef just after the V2 #ifdef
anchor = "// V2 ops implementations for kernels with iterator API\n#ifdef HAVE_BLK_RQ_DMA_MAP_ITER_START\n"
typedef_block = '''
/*
 * Kernel 6.16+ (e.g. 7.0) replaced struct req_iterator inside struct
 * blk_dma_iter with struct blk_map_iter. Both layouts share .iter
 * (bvec_iter) and .bio in the same order, so the helpers below only
 * use those two fields.
 */
#ifdef HAVE_BLK_MAP_ITER
typedef struct blk_map_iter nvfs_req_iter_t;
#else
typedef struct req_iterator nvfs_req_iter_t;
#endif

'''
if "nvfs_req_iter_t" not in s:
    assert anchor in s, "V2 #ifdef anchor not found"
    s = s.replace(anchor, anchor + typedef_block, 1)
    print("nvfs-dma.c: inserted typedef")

# Replace `struct req_iterator` with the typedef *only* inside the V2 path
# (i.e. between the V2 #ifdef and the matching #endif). The V1 path lives
# above this block and still expects struct req_iterator.
v2_start = s.index("// V2 ops implementations for kernels with iterator API\n#ifdef HAVE_BLK_RQ_DMA_MAP_ITER_START\n")
# match closing #endif that terminates the V2 section: it's the last #endif in file
# but to be safe locate the next "#endif /* HAVE_BLK_RQ_DMA_MAP_ITER_START */"
# Try canonical first, then fall back to last #endif before EOF.
end_marker = "#endif /* HAVE_BLK_RQ_DMA_MAP_ITER_START */"
if end_marker in s:
    v2_end = s.index(end_marker, v2_start)
else:
    v2_end = s.rfind("#endif")

before = s[:v2_start]
v2 = s[v2_start:v2_end]
after = s[v2_end:]

new_v2 = v2.replace("struct req_iterator", "nvfs_req_iter_t")
if new_v2 != v2:
    s = before + new_v2 + after
    p.write_text(s)
    print("nvfs-dma.c: rewired V2 path to nvfs_req_iter_t")
else:
    print("nvfs-dma.c: no further substitution needed (already patched)")
PY

# --- create_nv.symvers.sh: handle *.ko.zst (Ubuntu 26.04) ------------------
python3 - <<'PY'
from pathlib import Path
p = Path("create_nv.symvers.sh")
s = p.read_text()
anchor = '''\tcase "$nvidia_mod" in\n\t\t*ko.xz)\n\t\t\t/bin/cp -fv $nvidia_mod .\n\t\t\tnvidia_mod=$(basename $nvidia_mod | sed -e "s/.xz//g")\n\t\t\txz -d ${nvidia_mod}.xz\n\t\t\t;;\n\tesac'''
addition = '''\tcase "$nvidia_mod" in\n\t\t*ko.xz)\n\t\t\t/bin/cp -fv $nvidia_mod .\n\t\t\tnvidia_mod=$(basename $nvidia_mod | sed -e "s/.xz//g")\n\t\t\txz -d ${nvidia_mod}.xz\n\t\t\t;;\n\t\t*ko.zst)\n\t\t\t/bin/cp -fv $nvidia_mod .\n\t\t\tnvidia_mod=$(basename $nvidia_mod | sed -e "s/.zst//g")\n\t\t\tzstd -d --rm ${nvidia_mod}.zst\n\t\t\t;;\n\tesac'''
if "*ko.zst" in s:
    print("create_nv.symvers.sh: zst branch already present")
else:
    assert anchor in s, "xz case anchor not found in create_nv.symvers.sh"
    p.write_text(s.replace(anchor, addition, 1))
    print("create_nv.symvers.sh: added zst decompression branch")
PY

echo "patch done"
