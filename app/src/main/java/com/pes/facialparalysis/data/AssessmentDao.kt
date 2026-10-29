package com.pes.facialparalysis.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface AssessmentDao {

    /** Returns the new row id so callers can open the record straight after saving. */
    @Insert
    suspend fun insert(record: AssessmentRecord): Long

    @Query("SELECT * FROM assessments WHERE patientId = :patientId ORDER BY timestamp DESC")
    fun getRecordsForPatient(patientId: Int): Flow<List<AssessmentRecord>>

    @Query("SELECT * FROM assessments WHERE patientId = :patientId ORDER BY timestamp ASC")
    fun getRecordsForPatientAscending(patientId: Int): Flow<List<AssessmentRecord>>

    /** For the record detail screen opened from History. */
    @Query("SELECT * FROM assessments WHERE id = :id")
    suspend fun getById(id: Int): AssessmentRecord?
}