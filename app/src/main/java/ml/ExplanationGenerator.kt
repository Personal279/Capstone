package com.pes.facialparalysis.ml

object ExplanationGenerator {

    fun generate(breakdown: FaiCalculator.FaiBreakdown, grade: Int): String {
        val components = listOf(
            "eye region" to breakdown.eye,
            "mouth/lip region" to breakdown.mouth,
            "eyebrow region" to breakdown.brow,
            "cheek region" to breakdown.cheek,
            "jawline" to breakdown.jaw
        )

        val sorted = components.sortedByDescending { it.second }
        val topRegion = sorted[0].first
        val secondRegion = sorted[1].first

        return if (grade == 1) {
            "No significant facial asymmetry was detected across the eye, mouth, eyebrow, cheek, or jaw regions, consistent with a normal facial presentation."
        } else {
            "The largest asymmetry was observed in the $topRegion, with secondary asymmetry in the $secondRegion. " +
                    "This pattern contributed most to the predicted severity grade."
        }
    }

}