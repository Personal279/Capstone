package com.pes.facialparalysis.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "patients")
data class Patient(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val name: String,
    val createdAt: Long,
    /** Absolute path of the first image ever analysed for this patient. Written once. */
    val initialImagePath: String? = null,
    /** Absolute path of the image from the most recent test. Overwritten on every test. */
    val latestImagePath: String? = null
)