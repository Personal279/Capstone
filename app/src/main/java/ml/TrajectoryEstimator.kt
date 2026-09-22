package com.pes.facialparalysis.ml

import com.pes.facialparalysis.data.AssessmentRecord

/**
 * Experimental, model-estimated trajectory of overall facial asymmetry over time. This is a
 * simple linear regression fit to the patient's own real assessment history — it does not
 * fabricate data points and refuses to project when there isn't enough history to do so
 * responsibly. It is explicitly NOT a medical prognosis; callers must present it as such.
 */
object TrajectoryEstimator {

    data class TrajectoryResult(
        val hasEnoughData: Boolean,
        val slopePerDay: Double? = null,
        val projectedFaiIn14Days: Double? = null,
        val fitQuality: Double? = null,
        val trendDescription: String? = null
    )

    private const val MIN_ASSESSMENTS = 3

    fun estimate(history: List<AssessmentRecord>): TrajectoryResult {
        if (history.size < MIN_ASSESSMENTS) return TrajectoryResult(hasEnoughData = false)

        val ordered = history.sortedBy { it.timestamp }
        val t0 = ordered.first().timestamp
        val xs = ordered.map { (it.timestamp - t0) / 86_400_000.0 } // days since baseline
        val ys = ordered.map { it.faiValue }

        val n = xs.size
        val meanX = xs.average()
        val meanY = ys.average()
        val sxy = xs.indices.sumOf { (xs[it] - meanX) * (ys[it] - meanY) }
        val sxxSafe = xs.sumOf { (it - meanX) * (it - meanX) }
        if (sxxSafe < 1e-9) return TrajectoryResult(hasEnoughData = false)

        val slope = sxy / sxxSafe
        val intercept = meanY - slope * meanX

        val predicted = xs.map { slope * it + intercept }
        val ssRes = ys.indices.sumOf { (ys[it] - predicted[it]) * (ys[it] - predicted[it]) }
        val ssTot = ys.sumOf { (it - meanY) * (it - meanY) }
        val r2 = if (ssTot > 1e-9) (1.0 - ssRes / ssTot).coerceIn(0.0, 1.0) else 0.0

        val lastX = xs.last()
        val projected = slope * (lastX + 14) + intercept

        val trend = when {
            kotlin.math.abs(slope) < 0.02 -> "Roughly stable measured asymmetry over the assessed period."
            slope < 0 -> "Decreasing measured asymmetry trend over the assessed period."
            else -> "Increasing measured asymmetry trend over the assessed period."
        }

        return TrajectoryResult(
            hasEnoughData = true,
            slopePerDay = slope,
            projectedFaiIn14Days = projected.coerceIn(0.0, 100.0),
            fitQuality = r2,
            trendDescription = trend
        )
    }
}
