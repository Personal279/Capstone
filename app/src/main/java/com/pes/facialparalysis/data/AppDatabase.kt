package com.pes.facialparalysis.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

@Database(
    entities = [AssessmentRecord::class, Patient::class, DoctorNote::class, Reminder::class],
    version = 5,
    exportSchema = false
)
abstract class AppDatabase : RoomDatabase() {

    abstract fun assessmentDao(): AssessmentDao
    abstract fun patientDao(): PatientDao
    abstract fun doctorNoteDao(): DoctorNoteDao
    abstract fun reminderDao(): ReminderDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        /** 4 -> 5: image paths on patients, full XAI data on assessments. Keeps all existing rows. */
        val MIGRATION_4_5 = object : Migration(4, 5) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE patients ADD COLUMN initialImagePath TEXT")
                db.execSQL("ALTER TABLE patients ADD COLUMN latestImagePath TEXT")
                db.execSQL("ALTER TABLE assessments ADD COLUMN probabilitiesJson TEXT NOT NULL DEFAULT ''")
                db.execSQL("ALTER TABLE assessments ADD COLUMN limeSummary TEXT NOT NULL DEFAULT ''")
                db.execSQL("ALTER TABLE assessments ADD COLUMN limeImportanceJson TEXT NOT NULL DEFAULT ''")
                db.execSQL("ALTER TABLE assessments ADD COLUMN limeFitQuality REAL NOT NULL DEFAULT 0")
            }
        }

        fun getDatabase(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "facial_paralysis_db"
                )
                    .addMigrations(MIGRATION_4_5)
                    // Only versions that never had a migration path may still be wiped.
                    // Version 4 -> 5 is now preserved by MIGRATION_4_5.
                    .fallbackToDestructiveMigrationFrom(1, 2, 3)
                    .build()
                INSTANCE = instance
                instance
            }
        }
    }
}