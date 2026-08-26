package com.pes.facialparalysis.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "assessments")
data class AssessmentRecord(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val patientId: Int,
    val timestamp: Long,
    val grade: Int,
    val gradeLabel: String,
    val faiValue: Double,
    val confidence: Double
)