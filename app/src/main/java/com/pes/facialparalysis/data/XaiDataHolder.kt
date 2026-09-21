package com.pes.facialparalysis.data

import android.graphics.Bitmap
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.pes.facialparalysis.ml.FaiCalculator

/**
 * Transient, in-memory-only holder for the data the "View AI Explanation" screen needs:
 * the frame that was analyzed, its detected landmarks, and the region asymmetry breakdown.
 *
 * Never persisted and never logged — the underlying bitmap is sensitive biometric data.
 * Cleared alongside [CapturedImageHolder] when the user leaves the result flow.
 */
object XaiDataHolder {
    var bitmap: Bitmap? = null
    var landmarks: List<NormalizedLandmark>? = null
    var regions: List<FaiCalculator.RegionAsymmetry> = emptyList()
    var grade: Int? = null
    var probabilities: Map<Int, Double> = emptyMap()
    var summarySentence: String = ""
    var bulletLabels: List<String> = emptyList()

    fun clear() {
        bitmap?.recycle()
        bitmap = null
        landmarks = null
        regions = emptyList()
        grade = null
        probabilities = emptyMap()
        summarySentence = ""
        bulletLabels = emptyList()
    }
}
