package com.pes.facialparalysis.data

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * A note entered explicitly by a doctor/user for a patient. The AI never writes to this table —
 * every row here is a human-authored instruction or observation, never an AI-generated directive.
 */
@Entity(tableName = "doctor_notes")
data class DoctorNote(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val patientId: Int,
    val text: String,
    val createdAt: Long,
    /** e.g. "Physiotherapy", "Medication", "Follow-up", "General". Free-form, human-chosen. */
    val category: String = "General",
    /** Set once a Reminder row has been created from this note, so the UI can link them. */
    val linkedReminderId: Int? = null
)
