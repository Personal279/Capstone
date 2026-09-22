package com.pes.facialparalysis.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

@Dao
interface ReminderDao {

    @Insert
    suspend fun insert(reminder: Reminder): Long

    @Update
    suspend fun update(reminder: Reminder)

    @Query("SELECT * FROM reminders WHERE patientId = :patientId AND isActive = 1 ORDER BY createdAt DESC")
    fun getRemindersForPatient(patientId: Int): Flow<List<Reminder>>

    @Query("SELECT * FROM reminders WHERE id = :id LIMIT 1")
    suspend fun getById(id: Int): Reminder?

    @Query("UPDATE reminders SET isActive = 0 WHERE id = :id")
    suspend fun deactivate(id: Int)
}
