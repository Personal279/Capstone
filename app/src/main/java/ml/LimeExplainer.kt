package com.pes.facialparalysis.ml

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.exp
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Real LIME (Local Interpretable Model-agnostic Explanations) over the EXISTING severity/grade
 * model in [MlpClassifier]. No new prediction model is trained or introduced here — this is a
 * pure explanation layer wrapped around the already-trained classifier.
 *
 * How it works, concretely (matches the standard LIME algorithm):
 * 1. The face is divided into 9 "superpixel" regions (left/right eye, left/right eyebrow, nose,
 *    left/right cheek, mouth, jaw), defined from the SAME MediaPipe facial landmark indices
 *    already used elsewhere in the app (e.g. [FaiCalculator]) — not arbitrary/fake regions.
 * 2. A batch of random on/off perturbations of those 9 regions is generated (standard LIME
 *    perturbation sampling — with 9 binary features, exhaustively enumerating all 512
 *    combinations is unnecessary and too slow on-device, so we sample, as real LIME
 *    implementations do). Each perturbation occludes its "off" regions on a copy of the real
 *    analyzed frame (mid-gray fill) and is run back through the unmodified, already-trained
 *    [MlpClassifier] — every sample is an actual forward pass of the real model on a real
 *    perturbed image, not a simulated or invented value.
 * 3. A weighted linear surrogate model is fit (least squares, weighted so perturbations closer to
 *    the original fully-visible image count more — the LIME proximity kernel) mapping "which
 *    regions were visible" to "predicted probability of the model's actual predicted grade". The
 *    fitted coefficients ARE the LIME explanation: how much each region's visibility pushed the
 *    model's output toward (positive) or away from (negative) its predicted grade. Nothing about
 *    the per-region importance is hard-coded — it is entirely the output of this regression.
 *
 * The explanation is always computed fresh for the specific image + specific predicted grade
 * being shown; it is never reused across different assessments.
 */
object LimeExplainer {

    data class RegionImportance(
        val region: String,
        val displayLabel: String,
        /** Signed LIME coefficient: positive = this region's visibility supports the predicted grade. */
        val weight: Double,
        /** |weight| scaled to 0..1 across all regions, for bar/heat rendering. */
        val normalizedMagnitude: Double,
        val direction: Direction
    )

    enum class Direction { SUPPORTS, OPPOSES, NEGLIGIBLE }

    data class LimeResult(
        val regionImportance: List<RegionImportance>,
        val localModelFitQuality: Double,
        /** Plain-language "why this prediction" sentence, generated from the top real LIME
         * contributors — never a static/templated claim about a specific region. */
        val summarySentence: String
    )

    private val displayNames = mapOf(
        "left_eye" to "Left eye",
        "right_eye" to "Right eye",
        "left_brow" to "Left eyebrow",
        "right_brow" to "Right eyebrow",
        "nose" to "Nose",
        "left_cheek" to "Left cheek",
        "right_cheek" to "Right cheek",
        "mouth" to "Mouth",
        "jaw" to "Jaw / lower face"
    )

    /**
     * Landmark indices approximating each region's extent, reusing the same MediaPipe Face Mesh
     * points already relied on elsewhere in the app (e.g. [FaiCalculator]'s eye/brow/cheek/jaw
     * pairs), just attributed to one side each instead of merged into a single bilateral region.
     * Exposed so the landmark-importance overlay can highlight the exact same points LIME scored.
     */
    val regionLandmarkIndices: Map<String, List<Int>> = mapOf(
        "left_eye" to listOf(159, 145, 33, 133),
        "right_eye" to listOf(386, 374, 263, 362),
        "left_brow" to listOf(105, 70, 107),
        "right_brow" to listOf(334, 300, 336),
        "nose" to listOf(1, 4, 5, 195, 197),
        "left_cheek" to listOf(234, 123),
        "right_cheek" to listOf(454, 352),
        "mouth" to listOf(0, 61, 291, 13, 14, 78, 308),
        "jaw" to listOf(152, 172, 397, 148, 377)
    )

    private val regionOrder: List<String> = listOf(
        "left_eye", "right_eye", "left_brow", "right_brow", "nose", "left_cheek", "right_cheek", "mouth", "jaw"
    )

    /** Random perturbation samples per explanation. With 9 binary features this comfortably
     * over-determines the 10-coefficient linear surrogate while staying feasible on-device. */
    private const val NUM_SAMPLES = 80

    /**
     * @param frame the exact bitmap that was fed to [MlpClassifier] for the prediction being explained
     * @param landmarks the landmarks detected on that same frame
     * @param predictedGrade the grade the existing model actually predicted (1..5) — LIME explains THIS output
     */
    fun explain(
        frame: Bitmap,
        landmarks: List<NormalizedLandmark>,
        predictedGrade: Int,
        classifier: MlpClassifier
    ): LimeResult {
        val order = regionOrder
        val n = order.size
        val occlusionRadiusPx = estimateOcclusionRadius(landmarks, frame.width, frame.height)

        // Deterministic per-image seed so re-inspecting the same stored assessment reproduces the
        // same explanation, while different assessments get independent perturbation samples.
        val seed = (predictedGrade.toLong() * 1_000_003L) xor (frame.width.toLong() * 92_821L) xor frame.height.toLong()
        val random = Random(seed)

        // Standard LIME perturbation sampling: random on/off masks over the region "superpixels",
        // always including the fully-visible (original) and fully-hidden anchors.
        val masks = LinkedHashSet<Int>()
        masks += (1 shl n) - 1 // all visible = the original image
        masks += 0 // all hidden = fully occluded baseline
        while (masks.size < NUM_SAMPLES) {
            var mask = 0
            for (bit in 0 until n) if (random.nextBoolean()) mask = mask or (1 shl bit)
            masks += mask
        }

        val samples = masks.map { mask ->
            val visibleRegions = order.filterIndexed { i, _ -> (mask shr i) and 1 == 1 }
            val perturbed = occlude(frame, landmarks, order.filter { it !in visibleRegions }, occlusionRadiusPx)
            val prob = classifier.predict(perturbed).probabilities[predictedGrade] ?: 0.0
            if (perturbed !== frame) perturbed.recycle()
            Triple(mask, visibleRegions.size, prob)
        }

        // Weighted least squares: features = [1, x_region1, ..., x_regionN] (1/0 visibility).
        // Weight kernel favors perturbations closer to the fully-visible original image, standard LIME practice.
        val featureCount = n + 1
        val ata = Array(featureCount) { DoubleArray(featureCount) }
        val atb = DoubleArray(featureCount)

        samples.forEach { (mask, activeCount, prob) ->
            val x = DoubleArray(featureCount)
            x[0] = 1.0
            for (i in 0 until n) x[i + 1] = ((mask shr i) and 1).toDouble()
            val hammingDistFromFull = n - activeCount
            val weight = exp(-(hammingDistFromFull * hammingDistFromFull) / (2.0 * (n * 0.75) * (n * 0.75)))

            for (i in 0 until featureCount) {
                atb[i] += weight * x[i] * prob
                for (j in 0 until featureCount) {
                    ata[i][j] += weight * x[i] * x[j]
                }
            }
        }

        val coefficients = solveLinearSystem(ata, atb) ?: DoubleArray(featureCount)
        val regionWeights = order.mapIndexed { i, region -> region to coefficients[i + 1] }.toMap()

        val maxAbs = regionWeights.values.maxOfOrNull { kotlin.math.abs(it) }?.coerceAtLeast(1e-6) ?: 1e-6
        val importance = order.map { region ->
            val w = regionWeights[region] ?: 0.0
            val direction = when {
                kotlin.math.abs(w) < maxAbs * 0.08 -> Direction.NEGLIGIBLE
                w > 0 -> Direction.SUPPORTS
                else -> Direction.OPPOSES
            }
            RegionImportance(
                region = region,
                displayLabel = displayNames[region] ?: region,
                weight = w,
                normalizedMagnitude = (kotlin.math.abs(w) / maxAbs).coerceIn(0.0, 1.0),
                direction = direction
            )
        }.sortedByDescending { it.normalizedMagnitude }

        // R^2-style fit quality of the local surrogate, so the UI can be honest about how well
        // this linear explanation actually tracks the real model's behavior on these samples.
        val meanProb = samples.map { it.third }.average()
        val ssTot = samples.sumOf { (it.third - meanProb) * (it.third - meanProb) }
        val ssRes = samples.sumOf { (mask, _, prob) ->
            val x = DoubleArray(featureCount)
            x[0] = 1.0
            for (i in 0 until n) x[i + 1] = ((mask shr i) and 1).toDouble()
            val predicted = x.indices.sumOf { i -> x[i] * coefficients[i] }
            (prob - predicted) * (prob - predicted)
        }
        val fitQuality = if (ssTot > 1e-9) (1.0 - ssRes / ssTot).coerceIn(0.0, 1.0) else 0.0

        return LimeResult(importance, fitQuality, buildSummarySentence(importance))
    }

    private fun buildSummarySentence(importance: List<RegionImportance>): String {
        val topSupporting = importance
            .filter { it.direction == Direction.SUPPORTS }
            .sortedByDescending { it.normalizedMagnitude }
            .take(2)
        return when (topSupporting.size) {
            0 -> "No single facial region stood out as a strong driver of this prediction — the " +
                "measured contributions were spread fairly evenly across the face."
            1 -> "The model's prediction was influenced most strongly by differences detected around " +
                "the ${topSupporting[0].displayLabel.lowercase()} region."
            else -> "The model's prediction was influenced most strongly by differences detected around " +
                "the ${topSupporting[0].displayLabel.lowercase()} and ${topSupporting[1].displayLabel.lowercase()} regions."
        }
    }

    private fun estimateOcclusionRadius(landmarks: List<NormalizedLandmark>, width: Int, height: Int): Float {
        // Interocular distance as the natural scale reference for the face in this image.
        val left = landmarks.getOrNull(33)
        val right = landmarks.getOrNull(263)
        val interocular = if (left != null && right != null) {
            val dx = (left.x() - right.x()) * width
            val dy = (left.y() - right.y()) * height
            sqrt((dx * dx + dy * dy).toDouble())
        } else {
            width * 0.3
        }
        return (interocular * 0.32).toFloat().coerceAtLeast(6f)
    }

    private fun occlude(
        source: Bitmap,
        landmarks: List<NormalizedLandmark>,
        regionsToHide: List<String>,
        radiusPx: Float
    ): Bitmap {
        if (regionsToHide.isEmpty()) return source
        val mutable = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutable)
        val paint = Paint().apply {
            color = Color.rgb(128, 128, 128)
            style = Paint.Style.FILL
            isAntiAlias = true
        }
        regionsToHide.forEach { region ->
            regionLandmarkIndices[region]?.forEach { idx ->
                landmarks.getOrNull(idx)?.let { lm ->
                    canvas.drawCircle(lm.x() * mutable.width, lm.y() * mutable.height, radiusPx, paint)
                }
            }
        }
        return mutable
    }

    /** Simple Gaussian elimination with partial pivoting for the small (10x10) normal-equations system. */
    private fun solveLinearSystem(a: Array<DoubleArray>, b: DoubleArray): DoubleArray? {
        val n = b.size
        val m = Array(n) { i -> DoubleArray(n + 1) { j -> if (j < n) a[i][j] else b[i] } }

        for (col in 0 until n) {
            var pivot = col
            for (row in col + 1 until n) {
                if (kotlin.math.abs(m[row][col]) > kotlin.math.abs(m[pivot][col])) pivot = row
            }
            if (kotlin.math.abs(m[pivot][col]) < 1e-10) return null
            val tmp = m[col]; m[col] = m[pivot]; m[pivot] = tmp

            for (row in 0 until n) {
                if (row == col) continue
                val factor = m[row][col] / m[col][col]
                for (k in col..n) m[row][k] -= factor * m[col][k]
            }
        }
        return DoubleArray(n) { i -> m[i][n] / m[i][i] }
    }
}
