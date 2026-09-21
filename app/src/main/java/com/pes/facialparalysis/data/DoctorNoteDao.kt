package com.pes.facialparalysis.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

@Dao
interface DoctorNoteDao {

    @Insert
    suspend fun insert(note: DoctorNote): Long

    @Update
    suspend fun update(note: DoctorNote)

    @Query("SELECT * FROM doctor_notes WHERE patientId = :patientId ORDER BY createdAt DESC")
    fun getNotesForPatient(patientId: Int): Flow<List<DoctorNote>>
}
