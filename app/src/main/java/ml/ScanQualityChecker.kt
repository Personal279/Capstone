package com.pes.facialparalysis.ml

import android.graphics.Bitmap
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark

/**
 * Real, on-device checks for a handful of scan-quality factors that could plausibly have
 * affected a prediction. Every entry in [ScanQualityResult.detectedFactors] corresponds to an
 * actual measurement made on the analyzed frame — nothing here is guessed after the fact to
 * explain a grade. Factors that are plausible-but-unverifiable on-device are kept separate in
 * [ScanQualityResult.unverifiedPossibleFactors] and must always be worded as "may" affect,
 * never as a claimed cause.
 */
object ScanQualityChecker {

    data class ScanQualityResult(
        val detectedFactors: List<String>,
        val unverifiedPossibleFactors: List<String>
    )

    private val alwaysPossibleUnverified = listOf(
        "Shadows across part of the face",
        "Facial expression at the moment of capture",
        "Camera lens or image quality",
        "Small errors in automatic landmark detection"
    )

    fun analyze(bitmap: Bitmap, landmarks: List<NormalizedLandmark>): ScanQualityResult {
        val detected = mutableListOf<String>()

        val brightness = averageBrightness(bitmap)
        if (brightness < 70.0) detected += "The image appeared dark or unevenly lit."
        if (brightness > 205.0) detected += "The image appeared overexposed / very bright."

        val bbox = landmarkBoundingBox(landmarks)
        val margin = 0.03
        if (bbox.minX < margin || bbox.minY < margin || bbox.maxX > 1 - margin || bbox.maxY > 1 - margin) {
            detected += "Part of the face may have been near the edge of the frame."
        }

        val yawRatio = estimateYawRatio(landmarks)
        if (yawRatio != null && yawRatio > 1.35) {
            detected += "The head appeared turned away from a straight frontal angle."
        }

        return ScanQualityResult(detectedFactors = detected, unverifiedPossibleFactors = alwaysPossibleUnverified)
    }

    private fun averageBrightness(bitmap: Bitmap): Double {
        val sampleStep = maxOf(1, (bitmap.width * bitmap.height) / 4000) // subsample for speed
        var sum = 0L
        var count = 0L
        var i = 0
        val w = bitmap.width
        val h = bitmap.height
        while (i < w * h) {
            val x = i % w
            val y = i / w
            val pixel = bitmap.getPixel(x, y)
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF
            sum += (0.299 * r + 0.587 * g + 0.114 * b).toLong()
            count++
            i += sampleStep
        }
        return if (count > 0) sum.toDouble() / count else 128.0
    }

    private data class BBox(val minX: Float, val minY: Float, val maxX: Float, val maxY: Float)

    private fun landmarkBoundingBox(landmarks: List<NormalizedLandmark>): BBox {
        var minX = 1f; var minY = 1f; var maxX = 0f; var maxY = 0f
        landmarks.forEach { lm ->
            if (lm.x() < minX) minX = lm.x()
            if (lm.y() < minY) minY = lm.y()
            if (lm.x() > maxX) maxX = lm.x()
            if (lm.y() > maxY) maxY = lm.y()
        }
        return BBox(minX, minY, maxX, maxY)
    }

    /** Ratio of nose-to-left-eye vs nose-to-right-eye distance; near 1.0 for a frontal face,
     * larger when the head is turned. A real geometric measurement, not a fabricated claim. */
    private fun estimateYawRatio(landmarks: List<NormalizedLandmark>): Double? {
        val nose = landmarks.getOrNull(1) ?: return null
        val leftEye = landmarks.getOrNull(33) ?: return null
        val rightEye = landmarks.getOrNull(263) ?: return null
        fun dist(a: NormalizedLandmark, b: NormalizedLandmark): Double {
            val dx = (a.x() - b.x()).toDouble()
            val dy = (a.y() - b.y()).toDouble()
            return kotlin.math.sqrt(dx * dx + dy * dy)
        }
        val dLeft = dist(nose, leftEye)
        val dRight = dist(nose, rightEye)
        val smaller = kotlin.math.min(dLeft, dRight).coerceAtLeast(1e-4)
        val larger = kotlin.math.max(dLeft, dRight)
        return larger / smaller
    }
}
