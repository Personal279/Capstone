package com.pes.facialparalysis.data

/**
 * In-memory view model over a patient's [AssessmentRecord] history. Not a persisted entity —
 * everything here is derivable from the assessments table, so no schema/table beyond the
 * additive columns on AssessmentRecord was needed to support it.
 */
data class DigitalTwinSnapshot(
    val patientId: Int,
    val baseline: AssessmentRecord?,
    val latest: AssessmentRecord?,
    val previous: AssessmentRecord?,
    /** Oldest first — the order a timeline reads in. */
    val history: List<AssessmentRecord>,
    val isDemoData: Boolean
) {
    val gradeChangeSincePrevious: Int?
        get() = if (latest != null && previous != null) latest.grade - previous.grade else null

    companion object {
        fun from(records: List<AssessmentRecord>, patientId: Int): DigitalTwinSnapshot {
            val ordered = records.sortedBy { it.timestamp }
            return DigitalTwinSnapshot(
                patientId = patientId,
                baseline = ordered.firstOrNull(),
                latest = ordered.lastOrNull(),
                previous = if (ordered.size >= 2) ordered[ordered.size - 2] else null,
                history = ordered,
                isDemoData = false
            )
        }

        /**
         * A clearly-labeled synthetic timeline for demoing the Digital Twin UI when a patient
         * doesn't yet have enough real assessments. Never written to the database — it exists
         * only for this in-memory snapshot, and every screen that renders it must show a
         * "DEMO DATA" badge alongside it.
         */
        fun demo(patientId: Int, seedGrade: Int): DigitalTwinSnapshot {
            val now = System.currentTimeMillis()
            val dayMs = 24L * 60 * 60 * 1000
            val gradeSteps = listOf(
                (seedGrade + 1).coerceAtMost(5),
                (seedGrade + 1).coerceAtMost(5),
                seedGrade,
                seedGrade
            )
            val demoHistory = gradeSteps.mapIndexed { index, grade ->
                val dayOffset = listOf(14, 7, 2, 0)[index]
                AssessmentRecord(
                    id = -(index + 1),
                    patientId = patientId,
                    timestamp = now - dayOffset * dayMs,
                    grade = grade,
                    gradeLabel = gradeLabelFor(grade),
                    faiValue = 18.0 - index * 2.5,
                    confidence = 0.72 + index * 0.05,
                    eyeAsymmetry = 20.0 - index * 3.0,
                    mouthAsymmetry = 22.0 - index * 4.0,
                    browAsymmetry = 10.0 - index * 1.5,
                    cheekAsymmetry = 12.0 - index * 2.0,
                    jawAsymmetry = 9.0 - index * 1.0,
                    topRegionsCsv = "Left mouth region,Left eye region",
                    xaiExplanation = "Demo data for illustration only."
                )
            }
            return DigitalTwinSnapshot(
                patientId = patientId,
                baseline = demoHistory.first(),
                latest = demoHistory.last(),
                previous = demoHistory[demoHistory.size - 2],
                history = demoHistory,
                isDemoData = true
            )
        }

        private fun gradeLabelFor(grade: Int) = when (grade) {
            1 -> "Normal"
            2 -> "Mild"
            3 -> "Moderate"
            4 -> "Moderately severe"
            else -> "Severe"
        }
    }
}
