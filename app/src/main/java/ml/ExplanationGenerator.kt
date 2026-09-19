package com.pes.facialparalysis.ml

object ExplanationGenerator {

    fun generate(grade: Int): String {
        return when (grade) {
            1 -> "No significant facial asymmetry was detected, consistent with a normal facial presentation."
            2 -> "Mild facial asymmetry was detected. Movement and symmetry are only slightly affected."
            3 -> "Moderate facial asymmetry was detected, with noticeable but not severe impairment in facial movement."
            4 -> "Moderately severe facial asymmetry was detected, indicating significant impairment in facial movement."
            else -> "Severe facial asymmetry was detected, indicating substantial impairment in facial movement and symmetry."
        }
    }
}