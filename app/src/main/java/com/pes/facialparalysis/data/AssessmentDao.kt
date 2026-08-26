package com.pes.facialparalysis.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface AssessmentDao {

    @Insert
    suspend fun insert(record: AssessmentRecord)

    @Query("SELECT * FROM assessments WHERE patientId = :patientId ORDER BY timestamp DESC")
    fun getRecordsForPatient(patientId: Int): Flow<List<AssessmentRecord>>

    @Query("SELECT * FROM assessments WHERE patientId = :patientId ORDER BY timestamp ASC")
    fun getRecordsForPatientAscending(patientId: Int): Flow<List<AssessmentRecord>>
}