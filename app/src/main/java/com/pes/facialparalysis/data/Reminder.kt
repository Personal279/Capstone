package com.pes.facialparalysis.data

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * A reminder created ONLY from an explicit doctor/user instruction (e.g. typed while adding a
 * Doctor's Note). The AI pipeline must never insert rows here on its own — this table has no
 * write path from any ML/grading code, only from user-driven UI actions.
 */
@Entity(tableName = "reminders")
data class Reminder(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val patientId: Int,
    val title: String,
    val notes: String = "",
    val hour: Int,
    val minute: Int,
    /** Comma-separated java.util.Calendar day constants, e.g. "2,4,6" for Mon/Wed/Fri. Empty = one-time. */
    val daysOfWeekCsv: String = "",
    /** Only used when [daysOfWeekCsv] is empty: epoch millis of the one-time trigger date (midnight local). */
    val oneTimeDateMillis: Long = 0L,
    val createdAt: Long,
    val isActive: Boolean = true
) {
    val isRecurring: Boolean get() = daysOfWeekCsv.isNotBlank()

    fun daysOfWeek(): List<Int> =
        if (daysOfWeekCsv.isBlank()) emptyList()
        else daysOfWeekCsv.split(",").mapNotNull { it.trim().toIntOrNull() }
}
