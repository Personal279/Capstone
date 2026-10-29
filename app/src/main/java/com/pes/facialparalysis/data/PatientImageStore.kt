package com.pes.facialparalysis.data

import android.content.Context
import android.graphics.Bitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Keeps exactly two images per patient, in app-private storage (not visible to other apps,
 * and excluded from backup because the manifest sets allowBackup="false"):
 *   - initial.jpg : written once, on the patient's first analysed test, never overwritten
 *   - latest.jpg  : overwritten on every test
 */
object PatientImageStore {

    private fun dir(context: Context, patientId: Int): File =
        File(context.filesDir, "patient_images/$patientId").apply { mkdirs() }

    /**
     * Saves [bitmap] as the latest image, and also as the initial image if [existingInitialPath]
     * is null or its file is gone. Returns the (initialPath, latestPath) to store on the Patient.
     * Does NOT recycle the bitmap.
     */
    suspend fun save(
        context: Context,
        patientId: Int,
        bitmap: Bitmap,
        existingInitialPath: String?
    ): Pair<String, String> = withContext(Dispatchers.IO) {
        val folder = dir(context, patientId)

        val latest = File(folder, "latest.jpg")
        writeJpeg(bitmap, latest)

        val existing = existingInitialPath?.let { File(it) }
        val initialPath = if (existing != null && existing.exists()) {
            existing.absolutePath
        } else {
            val initial = File(folder, "initial.jpg")
            writeJpeg(bitmap, initial)
            initial.absolutePath
        }
        initialPath to latest.absolutePath
    }

    /** Call when a patient is deleted, or when the user withdraws consent to storage. */
    suspend fun deleteAll(context: Context, patientId: Int) = withContext(Dispatchers.IO) {
        File(context.filesDir, "patient_images/$patientId").deleteRecursively()
    }

    private fun writeJpeg(bitmap: Bitmap, file: File) {
        val tmp = File(file.parentFile, file.name + ".tmp")
        tmp.outputStream().use { bitmap.compress(Bitmap.CompressFormat.JPEG, 92, it) }
        tmp.renameTo(file) // atomic-ish: never leaves a half-written image
    }
}