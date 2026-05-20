import nextflow.Nextflow
import org.slf4j.Logger
import org.slf4j.LoggerFactory

/**
 * Helpers for the 3D reconstruction pipeline.
 *
 * Auto-loaded by Nextflow from `workflows/reconst_3d/lib/`.  All methods are
 * static; impure ones take `params` (the Nextflow params Map) as an explicit
 * argument so they're easy to test and grep for.
 *
 * Categories:
 *   - Slice ID parsing      : extractSliceId, extractSliceIdInt, toSliceTuple,
 *                             extractSliceIdsString, detectSingleGaps,
 *                             parseDebugSlices
 *   - Path utilities        : normalizePath, joinPath, resolveSubjectName,
 *                             partitionSlicesAndTransforms
 *   - Slice config parsing  : parseSliceConfig
 *   - CLI flag builders     : pyramidArgs, annotatedScreenshotArgs,
 *                             diagEnabled, stack*Args (option builders for
 *                             linum_stack_slices_motor.py)
 *
 * Boolean parsing in parseSliceConfig is kept in lockstep with
 * `linumpy.io.slice_config._parse_bool`; edit there too when the canonical
 * schema changes.
 */
class Helpers {

    private static final Logger log = LoggerFactory.getLogger('Helpers')

    // -------------------------------------------------------------------------
    // Slice ID parsing
    // -------------------------------------------------------------------------

    /** Extract z## slice ID string from a filename; returns "unknown" if not found. */
    static String extractSliceId(filename) {
        def name = filename instanceof java.nio.file.Path ? filename.fileName.toString() : filename.toString()
        def matcher = name =~ /z(\d+)/
        return matcher ? matcher[0][1] : 'unknown'
    }

    /** Extract slice ID as integer; returns -1 if not found. */
    static int extractSliceIdInt(filename) {
        def id = extractSliceId(filename)
        return id == 'unknown' ? -1 : id.toInteger()
    }

    /** Return [slice_id, file] for a given file path. */
    static List toSliceTuple(file_path) {
        return [extractSliceId(file_path), file_path]
    }

    /** Sorted, comma-separated slice IDs from a list of files (e.g. "01,02,03,05"). */
    static String extractSliceIdsString(List fileList) {
        fileList
            .collect { f -> extractSliceId(f) }
            .findAll { s -> s != 'unknown' }
            .sort { s -> s.toInteger() }
            .join(',')
    }

    /**
     * Detect single-slice gaps in a sorted slice list.
     * Returns a list of [missingId, beforeId, afterId] tuples (zero-padded).
     */
    static List detectSingleGaps(List sliceList) {
        def gaps = []
        def sliceIds = sliceList
            .collect { f -> extractSliceIdInt(f) }
            .findAll { n -> n >= 0 }
            .sort()

        sliceIds.eachWithIndex { current, i ->
            if (i >= sliceIds.size() - 1) return
            def next = sliceIds[i + 1]
            def gap = next - current
            if (gap == 2) {
                def missingId = String.format('%02d', current + 1)
                def beforeId = String.format('%02d', current)
                def afterId = String.format('%02d', next)
                gaps.add([missingId, beforeId, afterId])
                log.info("Gap detected: slice ${missingId} (between ${beforeId} and ${afterId})")
            } else if (gap > 2) {
                log.warn("Multiple missing slices between ${current} and ${next} - cannot interpolate")
            }
        }
        return gaps
    }

    /**
     * Parse `params.debug_slices`; supports "25,26", "25-29", or "25,27-29".
     * Returns a Set of zero-padded slice IDs, or null if blank.
     */
    static Set parseDebugSlices(String debugSlicesStr) {
        if (!debugSlicesStr || debugSlicesStr.trim().isEmpty()) return null
        def sliceIds = [] as Set
        debugSlicesStr.split(',').each { rawPart ->
            def part = rawPart.trim()
            if (part.contains('-')) {
                def rangeParts = part.split('-')
                if (rangeParts.size() == 2) {
                    def start = rangeParts[0].trim().toInteger()
                    def end = rangeParts[1].trim().toInteger()
                    (start..end).each { n -> sliceIds.add(String.format('%02d', n)) }
                }
            } else {
                sliceIds.add(String.format('%02d', part.toInteger()))
            }
        }
        return sliceIds
    }

    // -------------------------------------------------------------------------
    // Path utilities
    // -------------------------------------------------------------------------

    /** Remove duplicate and trailing slashes from a path string. */
    static String normalizePath(String path) {
        return path.replaceAll('/+', '/').replaceAll('/$', '')
    }

    /** Join path components safely. */
    static String joinPath(String base, String filename) {
        return "${normalizePath(base)}/${filename}"
    }

    /**
     * Resolve subject_name from inputDir, with this fallback order:
     *   1. `overrideName` if non-empty (typically `params.subject_name`)
     *   2. `sub-XX` token anywhere in the path
     *   3. parent of common input dirnames (`mosaic-grids`, `mosaics`, ...)
     *   4. leaf directory name
     */
    static String resolveSubjectName(String inputDir, String overrideName) {
        if (overrideName) return overrideName
        def subMatch = inputDir.split('/').find { part -> part ==~ /sub-\w+/ }
        if (subMatch) return subMatch
        def f = new File(inputDir)
        def dirName = f.getName()
        if (dirName in ['mosaic-grids', 'mosaics', 'mosaic_grids', 'input', 'data']) {
            return f.getParentFile()?.getName() ?: dirName
        }
        return dirName
    }

    /**
     * Partition a flat list of staged files into [slices, transforms]: items
     * ending in `.ome.zarr` go to slices; everything else (excluding `*.json`
     * metrics) goes to transforms.
     */
    static List partitionSlicesAndTransforms(List items) {
        def slices = items.findAll { f -> f.getName().endsWith('.ome.zarr') }
        def transforms = items.findAll { f ->
            def n = f.getName()
            !n.endsWith('.ome.zarr') && !n.endsWith('.json')
        }
        return [slices, transforms]
    }

    // -------------------------------------------------------------------------
    // Slice config parsing
    // -------------------------------------------------------------------------

    /**
     * Parse slice_config.csv, returning [use: Set<String>, excluded: Set<String>].
     * Truthy values: true, 1, yes, y, t (case-insensitive).
     */
    static Map parseSliceConfig(String configPath) {
        def slicesToUse = [] as Set
        def slicesExcluded = [] as Set
        def f = new File(configPath)

        if (!f.exists()) {
            Nextflow.error("Slice config file not found: ${configPath}")
        }

        def truthy = ['true', '1', 'yes', 'y', 't'] as Set
        f.withReader { reader ->
            reader.readLine() // Skip header
            reader.eachLine { line ->
                def parts = line.split(',')
                if (parts.size() >= 2) {
                    def sliceId = parts[0].trim()
                    def use = parts[1].trim().toLowerCase()
                    if (truthy.contains(use)) slicesToUse.add(sliceId)
                    else slicesExcluded.add(sliceId)
                }
            }
        }
        return [use: slicesToUse, excluded: slicesExcluded]
    }

    // -------------------------------------------------------------------------
    // CLI flag builders (params-coupled)
    // -------------------------------------------------------------------------

    /** True when the named per-stage diagnostic flag (or `diagnostic_mode`) is set. */
    static boolean diagEnabled(Map params, String flag) {
        return params.diagnostic_mode || params[flag]
    }

    /** Annotated-screenshot CLI flags shared by `stack` and `correct_bias_field`. */
    static String annotatedScreenshotArgs(Map params, String sliceIdsStr) {
        def show_lines = params.annotated_show_lines ? '--show_lines' : ''
        def orient = params.ras_input_orientation?.toString()?.trim()?.replace("'", '') ?: ''
        def orientation = orient ? "--orientation ${orient}" : ''
        return "--slice_ids \"${sliceIdsStr}\" --label_every ${params.annotated_label_every} ${show_lines} ${orientation} --crop_to_tissue"
    }

    /**
     * Build pyramid-related CLI arguments from `params.pyramid_*` settings.
     * `nLevelsFlag` names the downstream flag (`--n_levels` for most scripts,
     * `--n-levels` for `linum_align_to_ras.py`).
     */
    static String pyramidArgs(Map params, String nLevelsFlag = '--n_levels') {
        def opts = ''
        if (params.pyramid_n_levels != null) {
            opts += " ${nLevelsFlag} ${params.pyramid_n_levels}"
        } else {
            def base_res = params.resolution > 0 ? params.resolution : 10
            def valid = params.pyramid_resolutions.findAll { r -> r >= base_res }.sort()
            if (!valid.contains(base_res)) valid = [base_res] + valid
            opts += ' --pyramid_resolutions ' + valid.collect { r -> r.toString() }.join(' ')
            opts += params.pyramid_make_isotropic ? ' --make_isotropic' : ' --no_isotropic'
        }
        return opts
    }

    // -------------------------------------------------------------------------
    // `stack` option builders
    //
    // Split by concern so each `if` group lives next to the related parameters
    // rather than as one 65-line imperative blob.
    // -------------------------------------------------------------------------

    static String stackBlendingArgs(Map params) {
        def opts = ''
        if (params.stack_blend_enabled) opts += ' --blend'
        if (params.blend_refinement_px > 0) opts += " --blend_refinement_px ${params.blend_refinement_px}"
        if (params.stack_blend_z_refine_vox > 0) opts += " --blend_z_refine_vox ${params.stack_blend_z_refine_vox}"
        if (params.blend_z_refine_min_confidence > 0) opts += " --blend_z_refine_min_confidence ${params.blend_z_refine_min_confidence}"
        return opts
    }

    static String stackZMatchingArgs(Map params) {
        def opts = ''
        opts += " --slicing_interval_mm ${params.registration_slicing_interval_mm}"
        opts += " --search_range_mm ${params.registration_allowed_drifting_mm}"
        opts += " --moving_z_first_index ${params.moving_slice_first_index}"
        if (params.use_expected_z_overlap) opts += ' --use_expected_overlap'
        if (params.z_overlap_min_corr > 0) opts += " --z_overlap_min_corr ${params.z_overlap_min_corr}"
        if (params.analyze_shifts) opts += ' --output_z_matches z_matches.csv'
        opts += ' --output_stacking_decisions stacking_decisions.csv'
        return opts
    }

    static String stackPairwiseTransformArgs(Map params) {
        if (!params.apply_pairwise_transforms) return ''
        def opts = ' --transforms_dir transforms'
        if (params.apply_rotation_only) opts += ' --rotation_only'
        opts += " --max_rotation_deg ${params.max_rotation_deg}"
        if (params.load_transform_min_zcorr > 0) opts += " --load_min_zcorr ${params.load_transform_min_zcorr}"
        if (params.load_transform_max_rotation > 0) opts += " --load_max_rotation ${params.load_transform_max_rotation}"
        if (params.skip_error_transforms) opts += ' --skip_error_transforms'
        if (params.skip_warning_transforms) opts += ' --skip_warning_transforms'
        opts += " --confidence_high ${params.transform_confidence_high}"
        opts += " --confidence_low ${params.transform_confidence_low}"
        return opts
    }

    /** Drives per-slice use/auto_excluded → motor-only fallback in stack. */
    static String stackSliceConfigArg(slice_config) {
        return slice_config.name != 'NO_SLICE_CONFIG' ? " --slice_config ${slice_config}" : ''
    }

    /**
     * Skipped when refine_manual_transforms baked manual corrections into
     * the transforms directory; passing them again would double-apply.
     */
    static String stackManualOverrideArg(Map params) {
        return (params.manual_transforms_dir && !params.refine_manual_transforms)
            ? " --manual_transforms_dir ${params.manual_transforms_dir}"
            : ''
    }

    static String stackCumulativeArgs(Map params) {
        if (!params.stack_accumulate_translations) return ''
        def opts = ' --accumulate_translations'
        if (params.stack_confidence_weight_translations) opts += ' --confidence_weight_translations'
        if (params.stack_max_cumulative_drift_px > 0) opts += " --max_cumulative_drift_px ${params.stack_max_cumulative_drift_px}"
        // > 0 filters clamped translations; 0 = keep all (preserves re-homing boundary corrections).
        if (params.stack_max_pairwise_translation > 0) opts += " --max_pairwise_translation ${params.stack_max_pairwise_translation}"
        return opts
    }

    static String stackSmoothingArgs(Map params) {
        def opts = ''
        if (params.stack_smooth_window > 0) opts += " --smooth_window ${params.stack_smooth_window}"
        if (params.stack_translation_smooth_sigma > 0) opts += " --translation_smooth_sigma ${params.stack_translation_smooth_sigma}"
        if (params.stack_translation_min_zcorr > 0) opts += " --translation_min_zcorr ${params.stack_translation_min_zcorr}"
        return opts
    }

    // -------------------------------------------------------------------------
    // GPU pinning
    // -------------------------------------------------------------------------

    /**
     * Bash block that pins the current task to the least-loaded physical GPU.
     *
     * In-process pinning via `cp.cuda.Device(N).use()` is leaky — zarr's GPU
     * buffer prototype and the kvikio/GDS path still allocate on device 0
     * regardless. Setting `CUDA_VISIBLE_DEVICES` in the shell hides the other
     * cards from the process so no library can route around the choice.
     *
     * The returned snippet maintains per-GPU PID directories under /tmp guarded
     * by `flock`. On entry it prunes dead PIDs, picks the GPU with the fewest
     * live forks, and registers `$$` in that GPU's directory. On shell exit
     * (`trap ... EXIT`) it removes its own PID file. SIGKILL skips the trap, so
     * the dead-PID prune at next acquire reclaims any leaked slot — this matters
     * because OOM-kills (exit 137) used to leave slots permanently incremented,
     * skewing the load balancer toward the surviving GPU.
     *
     * Pass `params` and a short `tag` used for the per-task log line.
     * When `params.use_gpu` is false this returns an empty string.
     */
    static String gpuPinBlock(params, String tag) {
        if (!params.use_gpu) return ''
        def n = params.gpu_count as int
        return """
        UID_=\$(id -u)
        GPU_DIR=/tmp/linumpy_nf_gpu_slots_\${UID_}
        mkdir -p \$GPU_DIR
        for i in \$(seq 0 \$((${n} - 1))); do mkdir -p \$GPU_DIR/\$i; done
        _gpu_acquire() {
            exec 9>\$GPU_DIR/.lock
            flock 9
            local best=0 best_n=999999 i n_alive pid_file pid
            for i in \$(seq 0 \$((${n} - 1))); do
                n_alive=0
                for pid_file in \$GPU_DIR/\$i/*; do
                    [ -f "\$pid_file" ] || continue
                    pid=\$(basename "\$pid_file")
                    if kill -0 "\$pid" 2>/dev/null; then
                        n_alive=\$((n_alive + 1))
                    else
                        rm -f "\$pid_file"
                    fi
                done
                if [ "\$n_alive" -lt "\$best_n" ]; then best=\$i; best_n=\$n_alive; fi
            done
            : > \$GPU_DIR/\$best/\$\$
            echo \$best
        }
        _gpu_release() {
            rm -f \$GPU_DIR/\$1/\$\$
        }
        export CUDA_VISIBLE_DEVICES=\$(_gpu_acquire)
        trap "_gpu_release \$CUDA_VISIBLE_DEVICES" EXIT
        echo "[${tag}] CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"
        """
    }
}
