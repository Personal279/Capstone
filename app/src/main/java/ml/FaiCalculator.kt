package com.pes.facialparalysis.ml

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

object FaiCalculator {

    data class FaiBreakdown(
        val eye: Double,
        val mouth: Double,
        val brow: Double,
        val cheek: Double,
        val jaw: Double
    ) {
        /** Order must match training: [eye, mouth, brow, cheek, jaw] */
        fun toFeatureArray(): DoubleArray = doubleArrayOf(eye, mouth, brow, cheek, jaw)
    }

    /**
     * Distance in normalized landmark space, scaled by image width/height,
     * matching the Python dist(p1, p2) = np.linalg.norm(p1 - p2) on pixel coords.
     */
    private fun dist(
        a: NormalizedLandmark, b: NormalizedLandmark,
        width: Int, height: Int
    ): Double {
        val dx = (a.x() - b.x()) * width
        val dy = (a.y() - b.y()) * height
        return sqrt((dx * dx + dy * dy).toDouble())
    }

    /**
     * Computes the 5 asymmetry features. width/height must be the ORIGINAL
     * image/frame dimensions used when landmarks were detected (pixel scale
     * matters for dist(), matching the Python get_pt(landmarks, idx, w, h)).
     */
    fun computeBreakdown(
        landmarks: List<NormalizedLandmark>,
        width: Int,
        height: Int
    ): FaiBreakdown {
        // Eye: 159/145 (left) vs 386/374 (right)
        val lEye = dist(landmarks[159], landmarks[145], width, height)
        val rEye = dist(landmarks[386], landmarks[374], width, height)
        val eye = abs(lEye - rEye) / max(lEye, max(rEye, 1e-5)) * 100.0

        // Eyebrow: nose(4) -> 105 (left) vs nose(4) -> 334 (right)
        val lBrow = dist(landmarks[4], landmarks[105], width, height)
        val rBrow = dist(landmarks[4], landmarks[334], width, height)
        val brow = abs(lBrow - rBrow) / max(lBrow, max(rBrow, 1e-5)) * 100.0

        // Mouth: center(0) -> 61 (left) vs center(0) -> 291 (right)
        val lMouth = dist(landmarks[0], landmarks[61], width, height)
        val rMouth = dist(landmarks[0], landmarks[291], width, height)
        val mouth = abs(lMouth - rMouth) / max(lMouth, max(rMouth, 1e-5)) * 100.0

        // Cheek: nose_ref(1) -> 234 (left) vs nose_ref(1) -> 454 (right)
        val lCheek = dist(landmarks[1], landmarks[234], width, height)
        val rCheek = dist(landmarks[1], landmarks[454], width, height)
        val cheek = abs(lCheek - rCheek) / max(lCheek, max(rCheek, 1e-5)) * 100.0

        // Jaw: chin(152) -> 172 (left) vs chin(152) -> 397 (right)
        val lJaw = dist(landmarks[152], landmarks[172], width, height)
        val rJaw = dist(landmarks[152], landmarks[397], width, height)
        val jaw = abs(lJaw - rJaw) / max(lJaw, max(rJaw, 1e-5)) * 100.0

        return FaiBreakdown(eye, mouth, brow, cheek, jaw)
    }

    /** Returns the 5-dim feature array in training order: [eye, mouth, brow, cheek, jaw] */
    fun compute(landmarks: List<NormalizedLandmark>, width: Int, height: Int): DoubleArray =
        computeBreakdown(landmarks, width, height).toFeatureArray()

    /** One bilateral facial region: its name, asymmetry %, and which side measured smaller. */
    data class RegionAsymmetry(
        val region: String,
        val asymmetryPercent: Double,
        /** "left" or "right" — whichever side's raw landmark distance was smaller. Purely
         * geometric (not a claim about which side is clinically affected). */
        val smallerSide: String
    )

    /**
     * Same 5 regions as [computeBreakdown], but also reports which side's raw distance was
     * smaller, so callers can point at a specific left/right region rather than just the
     * bilateral pair. Used for XAI region explanations — every number here is a real
     * measurement from the detected landmarks, nothing invented.
     */
    fun computeDetailed(
        landmarks: List<NormalizedLandmark>,
        width: Int,
        height: Int
    ): List<RegionAsymmetry> {
        fun region(name: String, lIdx1: Int, lIdx2: Int, rIdx1: Int, rIdx2: Int): RegionAsymmetry {
            val l = dist(landmarks[lIdx1], landmarks[lIdx2], width, height)
            val r = dist(landmarks[rIdx1], landmarks[rIdx2], width, height)
            val pct = abs(l - r) / max(l, max(r, 1e-5)) * 100.0
            val smaller = if (l <= r) "left" else "right"
            return RegionAsymmetry(name, pct, smaller)
        }

        return listOf(
            region("eye", 159, 145, 386, 374),
            region("mouth", 0, 61, 0, 291),
            region("brow", 4, 105, 4, 334),
            region("cheek", 1, 234, 1, 454),
            region("jaw", 152, 172, 152, 397)
        )
    }
}