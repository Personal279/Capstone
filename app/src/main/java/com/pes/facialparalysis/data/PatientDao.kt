package com.pes.facialparalysis.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface PatientDao {

    @Insert
    suspend fun insert(patient: Patient): Long

    @Query("SELECT * FROM patients ORDER BY name ASC")
    fun getAllPatients(): Flow<List<Patient>>

    @Query("SELECT * FROM patients WHERE id = :id")
    suspend fun getById(id: Int): Patient?

    /** Stores the paths of the initial and latest image for a patient. */
    @Query("UPDATE patients SET initialImagePath = :initial, latestImagePath = :latest WHERE id = :id")
    suspend fun setImagePaths(id: Int, initial: String, latest: String)
}