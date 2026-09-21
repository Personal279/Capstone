package com.pes.facialparalysis.ml

/**
 * Turns real, already-computed facial-landmark asymmetry measurements into a plain-language
 * explanation of which facial regions the assessment's measurable asymmetry was concentrated in.
 *
 * IMPORTANT — what this is and isn't:
 * The severity model is a CNN that classifies the raw image directly. On-device TFLite does not
 * expose its internal gradients or activations, so a true Grad-CAM/saliency map over the model's
 * own internals is not currently possible without retraining/re-exporting the model to emit
 * intermediate tensors. Rather than fabricate a heatmap, this explanation is built from the same
 * MediaPipe facial landmarks already detected during grading and the region-asymmetry measurements
 * in [FaiCalculator] (previously computed but unused). It is a measurable, image-derived proxy —
 * facial asymmetry co-located with the graded face — not a literal readout of the CNN's internal
 * attention. Wording below is written to reflect that distinction and never claims clinical
 * certainty or diagnosis.
 */
object XaiExplainer {

    private val displayNames = mapOf(
        "eye" to "eye",
        "mouth" to "mouth",
        "brow" to "eyebrow",
        "cheek" to "cheek",
        "jaw" to "lower face / jaw"
    )

    data class RegionFinding(
        val region: String,
        val displayLabel: String,
        val side: String,
        val asymmetryPercent: Double
    )

    data class XaiResult(
        val findings: List<RegionFinding>,
        /** Bullet-style summary lines, e.g. "Left eye region". */
        val bulletLabels: List<String>,
        val summarySentence: String
    )

    /** Only surface a region if its asymmetry is at least this large, to avoid noise. */
    private const val MIN_NOTEWORTHY_PERCENT = 4.0
    private const val MAX_REGIONS_SHOWN = 3

    fun explain(regions: List<FaiCalculator.RegionAsymmetry>): XaiResult {
        val ranked = regions
            .sortedByDescending { it.asymmetryPercent }
            .map {
                RegionFinding(
                    region = it.region,
                    displayLabel = displayNames[it.region] ?: it.region,
                    side = it.smallerSide,
                    asymmetryPercent = it.asymmetryPercent
                )
            }

        val notable = ranked.filter { it.asymmetryPercent >= MIN_NOTEWORTHY_PERCENT }
            .take(MAX_REGIONS_SHOWN)
            .ifEmpty { ranked.take(1) }

        val bullets = notable.map { finding ->
            val sideLabel = finding.side.replaceFirstChar { it.uppercase() }
            "$sideLabel ${finding.displayLabel} region"
        }

        val summary = if (notable.isNotEmpty() && notable.first().asymmetryPercent >= MIN_NOTEWORTHY_PERCENT) {
            "The model's prediction was influenced by measurable facial asymmetry, most concentrated around: " +
                bullets.joinToString(", ") + ". These measurements come from facial landmarks detected in " +
                "the image and contributed to, but do not by themselves determine, the model's prediction."
        } else {
            "The image showed only minor measured asymmetry between the left and right side of the face " +
                "across all analyzed regions, consistent with the model's prediction."
        }

        return XaiResult(findings = ranked, bulletLabels = bullets, summarySentence = summary)
    }
}
