package com.pes.facialparalysis.data

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "assessments")
data class AssessmentRecord(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val patientId: Int,
    val timestamp: Long,
    val grade: Int,
    val gradeLabel: String,
    /** Overall facial asymmetry index — average of the 5 region asymmetry percentages below. */
    val faiValue: Double,
    val confidence: Double,
    val eyeAsymmetry: Double = 0.0,
    val mouthAsymmetry: Double = 0.0,
    val browAsymmetry: Double = 0.0,
    val cheekAsymmetry: Double = 0.0,
    val jawAsymmetry: Double = 0.0,
    val topRegionsCsv: String = "",
    val xaiExplanation: String = "",

    // --- Added in DB version 5: full XAI assessment so it can be reopened and exported later ---
    /** JSON object {"1":0.02,"2":0.10,...} of per-grade confidence. */
    @ColumnInfo(defaultValue = "''") val probabilitiesJson: String = "",
    /** LIME one-sentence summary. */
    @ColumnInfo(defaultValue = "''") val limeSummary: String = "",
    /** JSON object {"eye":0.31,"mouth":0.12,...} of LIME region importance. */
    @ColumnInfo(defaultValue = "''") val limeImportanceJson: String = "",
    @ColumnInfo(defaultValue = "0") val limeFitQuality: Double = 0.0
)